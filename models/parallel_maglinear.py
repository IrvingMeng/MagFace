#!/usr/bin/env python
# Copyright (c) Aibee, Inc. and its affiliates. All Rights Reserved

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parameter import Parameter

from loguru import logger

import torchshard as ts


class ParallelMagLinear(torch.nn.Module):
    """
    Parallel fc for Mag loss
    """
    def __init__(self,
            inp_features,
            out_features,
            scale=64.0,
            useArcFace=1,
            easy_margin=True,
            local_rank_init=False            
        ):
        super(ParallelMagLinear, self).__init__()
        self.inp_features = inp_features
        self.out_features = out_features
        self.scale = scale
        self.useArcFace = useArcFace

        # weight
        self.weight = torch.Tensor(
            self.out_features,
            self.inp_features
        ).to(ts.distributed.get_rank())

        # info
        self.easy_margin = easy_margin        
        self.reset_parameters()
        self.slice_params()

    def reset_parameters(self):
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)

    def slice_params(self):
        self.weight = ts.distributed.scatter(self.weight, dim=0)
        # wrap into Parameter
        self.weight = Parameter(self.weight)

        # set parallel attr
        ts.register_parallel_dim(self.weight, -1)


    def forward(self, x, x_norm, m):
        """
        Here m is a function which generate adaptive margin
        """
        x = ts.distributed.copy(x.float())

        ada_margin = m(x_norm)
        # norm the weight
        weight_norm = F.normalize(self.weight, dim=-1)
        cos_theta = torch.mm(
            F.normalize(x),
            weight_norm.t()
        )

        if self.useArcFace:
            cos_m = torch.cos(ada_margin)
            sin_m = torch.sin(ada_margin)
            cos_theta = cos_theta.clamp(-1, 1)
            sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))

            cos_theta_m = cos_theta * cos_m - sin_theta * sin_m

            if self.easy_margin:
                cos_theta_m = torch.where(
                    cos_theta.float() > 0,
                    cos_theta_m.float(),
                    cos_theta.float()
                )
            else:
                mm = torch.sin(math.pi - ada_margin) * ada_margin
                threshold = torch.cos(math.pi - ada_margin)
                cos_theta_m = torch.where(
                    cos_theta.float() > threshold,
                    cos_theta_m.float(),
                    cos_theta.float() - mm
                )
        else:
            cos_theta_m = cos_theta - ada_margin

        # multiply the scale in advance
        cos_theta_m = self.scale * cos_theta_m
        cos_theta = self.scale * cos_theta

        # set parallel attribute
        ts.register_parallel_dim(cos_theta_m, -1)
        ts.register_parallel_dim(cos_theta, -1)

        return cos_theta, cos_theta_m

    def extra_repr(self):
        return 'inp_features={}, out_features={}, scale={}'.format(
            self.inp_features,
            self.out_features,
            self.scale
        )
