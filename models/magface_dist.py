#!/usr/bin/env python
import sys
sys.path.append("..")
from models import iresnet
from collections import OrderedDict
from termcolor import cprint
from torch.nn import Parameter
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import math
import torch
import torch.nn as nn
import os
import torch.nn.parallel as parallel
import torchshard as ts


def builder(args):
    model = MagFaceBuilder(args)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(args.gpu)
    model.features = parallel.DistributedDataParallel(
                    model.features.to(args.gpu),
                    device_ids=[args.gpu],
                    output_device=args.gpu
                )
    getattr(model, args.parallel_module_name).to(args.gpu)
    return model


def load_features(args):
    if args.arch == 'iresnet18':
        features = iresnet.iresnet18(
            pretrained=True,
            num_classes=args.embedding_size)
    elif args.arch == 'iresnet34':
        features = iresnet.iresnet34(
            pretrained=True,
            num_classes=args.embedding_size)
    elif args.arch == 'iresnet50':
        features = iresnet.iresnet50(
            pretrained=True,
            num_classes=args.embedding_size)
    elif args.arch == 'iresnet100':
        features = iresnet.iresnet100(
            pretrained=True,
            num_classes=args.embedding_size)
    else:
        raise ValueError()
    return features


class MagFaceBuilder(nn.Module):
    def __init__(self, args):
        super(MagFaceBuilder, self).__init__()
        self.features = load_features(args)
        self.l_margin = args.l_margin
        self.u_margin = args.u_margin
        self.l_a = args.l_a
        self.u_a = args.u_a

        self.parallel_module_name = args.parallel_module_name
        self.fc_parallel_type = None
        self.add_module(
            self.parallel_module_name,
            None
        )
        from .parallel_maglinear import ParallelMagLinear
        self.build_parallel_module(
                self.parallel_module_name,
                ParallelMagLinear(
                    args.embedding_size,
                    args.last_fc_size,
                    scale=args.arc_scale,
                    useArcFace=args.arc,
                    local_rank_init=False                 
                )
        )

    def build_parallel_module(self, module_name, module):
        setattr(self, module_name, module)

    def _margin(self, x):
        """generate adaptive margin
        """
        margin = (self.u_margin-self.l_margin) / \
            (self.u_a-self.l_a)*(x-self.l_a) + self.l_margin
        return margin

    def forward(self, x, target):
        x = self.features(x)
        x_norm = torch.norm(x, dim=1, keepdim=True).clamp(self.l_a, self.u_a)

        x = ts.distributed.gather(x, dim=0)
        x_norm = ts.distributed.gather(x_norm, dim=0)
        logits = getattr(self, self.parallel_module_name)(x,
                                                          x_norm,
                                                          self._margin)
        return logits, x_norm
