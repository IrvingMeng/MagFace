#!/usr/bin/env python
# Copyright (c) Aibee, Inc. and its affiliates. All Rights Reserved

import torch
import torch.distributed as dist

import numpy as np
import torchshard as ts


class ParallelMagLoss(torch.nn.Module):
    """
    Parallel Mag Loss.
    """
    def __init__(self, l_a, u_a, l_margin, u_margin, scale=64.0):
        super(ParallelMagLoss, self).__init__()
        self.l_a = l_a
        self.u_a = u_a
        self.l_margin = l_margin
        self.u_margin = u_margin
        self.large_value = 1 << 10
        self.scale = scale
        self.cut_off = np.cos(np.pi/2-self.l_margin)

    def calc_loss_G(self, x_norm):
        g = 1/(self.u_a**2) * x_norm + 1/(x_norm)
        return torch.mean(g)

    def forward(self, input, target, x_norm):
        cos_theta, cos_theta_m = input
        rank = ts.distributed.get_rank()
        world_size = ts.distributed.get_world_size()
        one_hot = torch.zeros((
            cos_theta.size(0),
            world_size * cos_theta.size(1)),
            device=cos_theta.device
        )
        one_hot.scatter_(1, target.view(-1, 1), 1.0)
        loss_g = self.calc_loss_G(x_norm)

        one_hot = ts.distributed.scatter(one_hot, dim=-1)
        output = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta

        # set parallel attribute
        parallel_dim = ts.get_parallel_dim(cos_theta)
        ts.register_parallel_dim(output, parallel_dim)

        loss = ts.nn.functional.parallel_cross_entropy(output, target)
        return loss, loss_g, one_hot #, valid_num
