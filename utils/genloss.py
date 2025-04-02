# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as tF
from .geomloss import SamplesLoss

class Cost:
    def __init__(self, factor=128) -> None:
        self.factor = factor

    def __call__(self, x, y):
        X, Y = x.clone(), y.clone()
        x_col = X.unsqueeze(-2) / self.factor
        y_row = Y.unsqueeze(-3) / self.factor
        C = torch.sum((x_col - y_row) ** 2, -1) ** 0.5 #torch.Size([1, 221184, 12])
        return C

per_cost = Cost(factor=128)
eps = 1e-8

class GeneralizedLoss(nn.modules.loss._Loss):
    def __init__(self, factor=100, reduction='mean') -> None:
        super().__init__()
        self.factor = factor #128
        self.reduction = reduction #mean
        self.tau = 5

        self.cost = per_cost
        self.blur = 0.01
        self.scaling = 0.8
        self.reach = 0.5
        self.p = 1
        self.uot = SamplesLoss(blur=self.blur, scaling=self.scaling, debias=False, backend='tensorized', cost=self.cost, reach=self.reach, p=self.p)
        self.pointLoss = nn.L1Loss(reduction=reduction)
        self.pixelLoss = nn.MSELoss(reduction=reduction)

        self.down = 1

    def forward(self, dens, dots):
        bs = dens.size(0) #torch.Size([2, 1, 384, 576])
        point_loss, pixel_loss, emd_loss = 0, 0, 0
        entropy = 0
        for i in range(bs):
            den = dens[i, 0] #torch.Size([384, 576])
            seq = torch.nonzero(dots[i, 0]) #torch.Size([12, 2])

            if seq.size(0) < 1 or den.sum() < eps:
                point_loss += torch.abs(den).mean()
                pixel_loss += torch.abs(den).mean()
                emd_loss += torch.abs(den).mean()
            else:
                A, A_coord = self.den2coord(den)
                A_coord = A_coord.reshape(1, -1, 2) #torch.Size([1, 221184, 2])
                A = A.reshape(1, -1, 1) #torch.Size([1, 221184, 1])

                B_coord = seq[None, :, :] #torch.Size([1, 12, 2])
                B = torch.ones(seq.size(0)).float().cuda().view(1, -1, 1) * self.factor #*128.0
                
                oploss, F, G = self.uot(A, A_coord, B, B_coord) #[2405.5364],[1, 221184, 1],[1, 12, 1]
                
                C = self.cost(A_coord, B_coord) #torch.Size([1, 221184, 12])
                PI = torch.exp((F.view(1, -1, 1) + G.view(1, 1, -1) - C).detach() / (self.blur ** self.p)) * A * B.view(1, 1, -1) #torch.Size([1, 221184, 12])
                entropy += torch.mean((1e-20+PI) * torch.log(1e-20+PI)) #tensor(-0.0039, device='cuda:0', grad_fn=<AddBackward0>)
                emd_loss += torch.mean(oploss)
                point_loss += self.pointLoss(PI.sum(dim=1).view(1, -1, 1), B)
                pixel_loss += self.pixelLoss(PI.sum(dim=2).detach().view(1, -1, 1), A)
                
        loss = (emd_loss + self.tau * (point_loss + pixel_loss) + self.blur * entropy) / bs
        return loss
    
    def den2coord(self, denmap):
        assert denmap.dim() == 2, f"denmap.shape = {denmap.shape}, whose dim is not 2"
        coord = torch.nonzero(denmap) #预测的dmap不为0的位置坐标。
        denval = denmap[coord[:, 0], coord[:, 1]] #不为0坐标处的值
        return denval, coord
    
