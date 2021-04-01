#!/usr/bin/env python
# Copyright (c) Aibee, Inc. and its affiliates. All Rights Reserved

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from .initialize import get_model_parallel_world_size
from .initialize import get_model_parallel_rank
from .initialize import get_model_parallel_group
from .mappings import copy_to_region
from .mappings import scatter_to_region
from .mappings import gather_from_region
from .utils import divide
from .utils import Utility
from loguru import logger

class ParallelMagLinear(torch.nn.Module):
    """
    Parallel fc for Mag loss
    """
    def __init__(self,
            inp_features,
            out_features,
            scale=64.0,
            easy_margin=True,
            local_rank_init=False
        ):
        super(ParallelMagLinear, self).__init__()
        self.inp_features = inp_features
        self.out_features = out_features
        self.scale = scale

        # divide
        self.world_size = get_model_parallel_world_size()
        self.out_features_per_partition = divide(
            self.out_features,
            self.world_size
        )

        # weight
        self.weight = torch.nn.Parameter(torch.Tensor(
            self.out_features_per_partition,
            self.inp_features
        ))

        # info
        self.easy_margin = easy_margin

        # init
        if local_rank_init:
            logger.warning('parallel weight initialization method: LOCAL rank')
            self.reset_local_rank_parameters()
        else:
            logger.warning('parallel weight initialization method: GLOBAL processes')
            self.reset_parameters()

    def reset_local_rank_parameters(self):
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.weight.model_parallel = True

    def reset_parameters(self):
        # init weight
        _weight = torch.empty(
            self.out_features,
            self.inp_features,
            dtype=self.weight.dtype,
            requires_grad=False
        )
        _weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)

        # scatter
        self.weight.data = scatter_to_region(_weight, dim=0)
        self.weight.model_parallel = True

        # del weight
        del _weight

    def forward(self, x, x_norm, m):
        """
        Here m is a function which generate adaptive margin
        """
        x = copy_to_region(x.float())

        ada_margin = m(x_norm)
        cos_m = torch.cos(ada_margin)
        sin_m = torch.sin(ada_margin)

        # norm the weight
        weight_norm = F.normalize(self.weight, dim=-1)
        cos_theta = torch.mm(
            F.normalize(x),
            weight_norm.t()
        )
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

        # multiply the scale in advance
        cos_theta_m = self.scale * cos_theta_m
        cos_theta = self.scale * cos_theta

        return cos_theta, cos_theta_m

    def extra_repr(self):
        return 'inp_features={}, out_features={}, scale={}'.format(
            self.inp_features,
            self.out_features,
            self.scale
        )