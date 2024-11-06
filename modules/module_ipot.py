import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class IPOT_module(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = 1

    def cost_matrix(self, x, y):

        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
        tmp1 = torch.matmul(x, y.T)
        cos_dis = 1 - tmp1

        return cos_dis

    def forward(self, t_prob, v_prob, n, bs, n_text, n_video):

        beta = self.beta

        sigma = torch.ones([n, 1], dtype=t_prob.dtype, device=t_prob.device) / n
        T = torch.ones([n, n], dtype=t_prob.dtype, device=t_prob.device)

        C = self.cost_matrix(t_prob, v_prob)

        A = torch.exp(-C / beta)

        for t in range(50):
            Q = A * T

            for k in range(1):
                delta = 1. / (n * torch.matmul(Q, sigma))
                sigma = 1. / (n * torch.matmul(Q.T, delta))

            tmp = torch.matmul(torch.diag(delta.squeeze()), Q)
            T = torch.matmul(tmp, torch.diag(sigma.squeeze()))

        '''distance_matrix in [B*K, B*K]'''
        distance_matrix = torch.matmul(C.T, T)
        # dis_loss = torch.trace(distance_matrix)
        distance_matrix = distance_matrix.reshape(bs, n_text, bs, n_video).mean(axis=(1, 3))
        dis_loss = torch.trace(distance_matrix)

        # '''Base on UATVR [https://github.com/bofang98/UATVR]'''
        # la = np.ones((n_text, n_video))
        # mm_mask = np.eye(8)
        # # 克洛克内积
        # mm_mask = np.kron(mm_mask, la)
        # mm_mask = torch.tensor(mm_mask).float().bool()
        # mm_mask = mm_mask.to(distance_matrix.device)
        # distance_matrix = (F.log_softmax(distance_matrix, dim=1) * mm_mask).sum(1)

        return distance_matrix, dis_loss

