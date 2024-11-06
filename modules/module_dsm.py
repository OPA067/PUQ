import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.base_config import Config

class DSM(nn.Module):
    def __init__(self, config: Config):
        super(DSM, self).__init__()
        self.batch_size = config.batch_size
        self.embed_dim = config.embed_dim
        self.num_frames = config.num_frames
        self.n_text = config.n_text_samples
        self.n_video = config.n_video_samples


    def forward(self, prob_text, prob_video):

        K = prob_text.size(1)                           # ==>> self.n_text == self.n_video

        prob_text_embedding = prob_text['embedding']    # [B, K, D]
        prob_text_logsigma = prob_text['logsigma']      # [B, D]
        prob_text_mean = prob_text['mean']              # [B, D]

        prob_video_embedding = prob_video['embedding']  # [B, K, D]
        prob_video_logsigma = prob_video['logsigma']    # [B, D]
        prob_video_mean = prob_video['mean']            # [B, D]

        '''1.Cosine Distance'''
        # prob_text_embedding = torch.mean(prob_text_embedding, dim=1)
        # prob_video_embedding = torch.mean(prob_video_embedding, dim=1)
        # prob_text_embedding = prob_text_embedding / prob_text_embedding.norm(dim=-1, keepdim=True)
        # prob_video_embedding = prob_video_embedding / prob_video_embedding.norm(dim=-1, keepdim=True)
        # Cos_Dis_Matrix = 1 - torch.mm(prob_text_embedding, prob_video_embedding.t())  # [B, B]

        '''2.Monte-Carlo'''
        # assert prob_text_embedding.size(1) == self.n_text
        # assert prob_video_embedding.size(1) == self.n_video
        # score = []
        # for id1 in range(prob_text_embedding.size(0)):
        #     temp = []
        #     for id2 in range(prob_video_embedding.size(0)):
        #         q_embed = prob_text_embedding[id1, ...].contiguous()
        #         k_embed = prob_video_embedding[id2, ...].contiguous()
        #         sim_embed = torch.matmul(q_embed, k_embed.t())
        #         sim_mean = torch.mean(sim_embed.view(prob_text_embedding.size(1) * prob_video_embedding.size(1)))
        #         temp.append(sim_mean)
        #     score.append(torch.stack(temp))
        # score = torch.stack(score)  # [B, B]

        '''3.MIL'''

        '''4.Uniformity Loss'''
        # x = torch.cat([prob_text_embedding.view(-1, self.embed_dim), prob_video_embedding.view(-1, self.embed_dim)])
        # max_samples = 16384
        # t = 2
        # if len(x) ** 2 > max_samples:
        #     # prevent CUDA error: https://github.com/pytorch/pytorch/issues/22313
        #     indices = np.random.choice(len(x), int(np.sqrt(max_samples)))
        #     x = x[indices]
        # loss = torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

        '''5.KL-divergence'''

        '''6.2-Wasserstein'''
        # distance = torch.sqrt((prob_text_mean - prob_video_mean) ** 2 + (prob_text_logsigma - prob_video_logsigma) ** 2)

        '''7.MIL-NCE'''
        # sim_matrix = torch.mm(prob_text_embedding.view(-1, self.embed_dim), prob_video_embedding.view(-1, self.embed_dim).t())
        # la = np.ones((self.n_text, self.n_video))
        # mm_mask = np.eye(self.batch_size * self.n_text)
        # mm_mask = np.kron(mm_mask, la)
        # mm_mask = torch.tensor(mm_mask).float().bool()
        # mm_mask = mm_mask.to(sim_matrix.device)
        # sim_loss = - (F.log_softmax(sim_matrix, dim=1) * mm_mask).sum(1) / mm_mask.sum(1)
        # sim_loss = sim_loss.mean()