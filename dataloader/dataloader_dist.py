#!/usr/bin/env python
import sys
sys.path.append("..")

from utils import cv2_trans as transforms
from termcolor import cprint
import cv2
import torchvision
import torch.utils.data as data
import torch
import random
import numpy as np
import os
import warnings
import math


class MagTrainDataset(data.Dataset):
    def __init__(self, ann_file, transform=None, rank=None,num_replicas=None):
        self.ann_file = ann_file
        self.transform = transform
        self.num_replicas = num_replicas
        self.rank = rank
        self.init()

    def init(self):
        self.weight = {}
        self.im_names = []
        self.targets = []
  
        with open(self.ann_file) as f:
            lnum = sum(1 for _ in f)
            idxs = list(range(0, lnum, int(lnum/self.num_replicas)))
            # TODO: ugly code
            if len(idxs) == self.num_replicas:
                idxs.append(lnum)
            cidx = [idxs[self.rank], idxs[self.rank+1]
                    ] if self.num_replicas > 1 else [0, lnum]
        with open(self.ann_file) as f:
            for idx, line in enumerate(f):
                if idx < cidx[0] or idx > cidx[1]:
                    continue
                data = line.strip().split(' ')
                self.im_names.append(data[0])
                target = int(data[2])
                self.targets.append(target)

    def __getitem__(self, index):
        im_name = self.im_names[index]
        target = self.targets[index]
        img = cv2.imread(im_name)

        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.im_names)


class ParallelDistributedSampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    Note:
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        shuffle (optional): If true (default), sampler will shuffle the indices

    Warning:
        In distributed mode, calling the ``set_epoch`` method is needed to
        make shuffling work; each process will use the same random seed
        otherwise.

    Example::

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0))
        self.total_size = self.num_samples
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def train_loader(args):
    train_trans = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    train_dataset = MagTrainDataset(
        args.train_list,
        transform=train_trans,
        rank=args.rank,
        num_replicas=args.world_size
    )
    train_sampler = ParallelDistributedSampler(
                    train_dataset,
                    rank=args.rank,
                    num_replicas=args.world_size,
                    shuffle=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=(train_sampler is None),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=(train_sampler is None))

    return train_loader        