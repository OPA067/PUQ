import numpy as np
import torch
import torch.nn as nn
from config.base_config import Config
from modules.loss import KLdivergence
from modules.module_dsm import DSM
from modules.module_ipot import IPOT_module
from modules.module_edl import EDL
from modules.text_transformer import text_transformer
from modules.video_transfomer import video_transformer

from prob_models.pie_model import PIENet
from prob_models.tensor_utils import l2_normalize, sample_gaussian_tensors
from prob_models.uncertainty_module import UncertaintyModuleImage, UncertaintyModuleText


class CLIPStochastic(nn.Module):
    def __init__(self, config: Config):
        super(CLIPStochastic, self).__init__()
        self.config = config
        self.n_text_samples = self.config.n_text_samples
        self.n_video_samples = self.config.n_video_samples

        '''
            self.alpha = self.config.alpha ==>> 1e0
            self.beta = self.config.beta   ==>> 1e-1
            self.gama = self.config.gama ==>> 1e-4
        '''
        self.alpha = self.config.alpha
        self.beta = self.config.beta
        self.gama = self.config.gama

        from transformers import CLIPModel
        if config.clip_arch == 'ViT-B/32':
            self.clip = CLIPModel.from_pretrained("./openai/clip-vit-base-patch32")
        elif config.clip_arch == 'ViT-B/16':
            self.clip = CLIPModel.from_pretrained("./openai/clip-vit-base-patch16")
        else:
            raise ValueError

        config.pooling_type = 'transformer'
        self.text_transformer = text_transformer(config)
        self.video_transformer = video_transformer(config)

        embed_dim = self.config.embed_dim
        self.pie_net_text = PIENet(1, embed_dim, embed_dim, embed_dim // 2)
        self.uncertain_net_text = UncertaintyModuleImage(embed_dim, embed_dim, embed_dim // 2)

        self.pie_net_video = PIENet(1, embed_dim, embed_dim, embed_dim // 2)
        self.uncertain_net_video = UncertaintyModuleText(embed_dim, embed_dim, embed_dim // 2)

        self.ipot = IPOT_module()

        self.loss_kl = KLdivergence()

        self.EDL = EDL(config)

        self.DSM = DSM(config)

    def probabilistic_text(self, text_pooled, text_token):
        output = {}
        out, attn, residual = self.pie_net_text(text_pooled, text_token)
        output['attention'] = attn
        output['residual'] = residual

        uncertain_out = self.uncertain_net_text(text_pooled, text_token)
        logsigma = uncertain_out['logsigma']
        output['logsigma'] = logsigma
        output['uncertainty_attention'] = uncertain_out['attention']

        out = l2_normalize(out)
        output['mean'] = out
        output['embedding'] = sample_gaussian_tensors(out, logsigma, self.n_text_samples)

        return output

    def probabilistic_video(self, video_pooled, video_features):
        output = {}
        out, attn, residual = self.pie_net_video(video_pooled, video_features)
        output['attention'] = attn
        output['residual'] = residual

        uncertain_out = self.uncertain_net_video(video_pooled, video_features)
        logsigma = uncertain_out['logsigma']
        output['logsigma'] = logsigma
        output['uncertainty_attention'] = uncertain_out['attention']

        out = l2_normalize(out)
        output['mean'] = out
        output['embedding'] = sample_gaussian_tensors(out, logsigma, self.n_video_samples)

        return output

    def forward(self, data, is_train=True):
        batch_size = data['video'].shape[0]
        text_data = data['text']
        video_data = data['video']
        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)

        word_features, text_features = self.clip.get_text_features(**text_data)
        _, video_features = self.clip.get_image_features(video_data)
        video_features = video_features.reshape(batch_size, self.config.num_frames, -1)

        if is_train:

            text_pool = self.text_transformer(text_features, video_features)
            text_pooled = torch.diagonal(text_pool, dim1=0, dim2=1).permute(1, 0)
            text_pooled = text_pooled / text_pooled.norm(dim=-1, keepdim=True)
            word_features = word_features / word_features.norm(dim=-1, keepdim=True)
            prob_text = self.probabilistic_text(text_pooled, word_features)
            prob_text_embedding = prob_text['embedding']    # [B, K, D]
            prob_text_logsigma = prob_text['logsigma']      # [B, D]

            video_pool = self.video_transformer(text_features, video_features)
            video_pooled = torch.diagonal(video_pool, dim1=0, dim2=1).permute(1, 0)
            video_pooled = video_pooled / video_pooled.norm(dim=-1, keepdim=True)
            video_features = video_features / video_features.norm(dim=-1, keepdim=True)
            prob_video = self.probabilistic_video(video_pooled, video_features)
            prob_video_embedding = prob_video['embedding']   # [B, K, D]
            prob_video_logsigma = prob_video['logsigma']     # [B, D]

            '''KL_divergence'''
            kl_loss = self.loss_kl(prob_video_embedding, prob_video_logsigma, prob_text_embedding, prob_text_logsigma)

            '''IPOT_Algorithm'''
            dim = self.config.embed_dim
            bs = prob_text_embedding.shape[0]
            bk = prob_text_embedding.shape[0] * prob_text_embedding.shape[1]
            dis_matrix, dis_loss = self.ipot(prob_text_embedding.view(-1, dim), prob_video_embedding.view(-1, dim), bk, bs, self.n_text_samples, self.n_video_samples)

            ''' About Other Distribution Supervision Methods (DSM)'''
            # dsm_loss = self.DSM(prob_text, prob_video)

            '''EDL(Evidential Deep Learning)'''
            sim_matrix = torch.matmul(text_pooled, video_pooled.t())
            edl_sim_loss = self.EDL(sim_matrix, status=True)
            edl_dis_loss = self.EDL(dis_matrix, status=False)

            return text_features, text_pooled, video_pool, self.alpha * dis_loss + self.beta * (edl_sim_loss + edl_dis_loss) + self.gama * kl_loss
        else:
            return text_features, video_features

