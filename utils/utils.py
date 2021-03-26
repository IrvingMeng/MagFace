"""Utility functions
    This file contains utility functions that are not used in the core library,
    but are useful for building models or training code using the config system.
"""
import logging
import os
import sys
import math
import shutil
import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn.functional as F
from termcolor import cprint
from loguru import logger


# classes
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


# functions
def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, args):
    decay = args.lr_drop_ratio if epoch in args.lr_drop_epoch else 1.0
    lr = args.lr * decay
    global current_lr
    current_lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    args.lr = current_lr
    return current_lr


def adjust_learning_rate_cosine(optimizer, epoch, args):
    """cosine learning rate annealing without restart"""
    lr = args.lr_min + (args.lr - args.lr_min) * \
        (1 + math.cos(math.pi * epoch / args.epochs)) / 2
    global current_lr
    current_lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return current_lr


def adjust_learning_rate_warmup(optimizer, epoch, args):
    """warmup learning rate gradually at the beginning of training according
       to paper: Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour
    """
    lr = args.lr * (epoch + 1) / args.warmup_epoch
    global current_lr
    current_lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return current_lr


def adjust_learning_rate_warmup_iter(optimizer, iter, args):
    """warmup learning rate per iteration gradually at the beginning of training according
       to paper: Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour
    """
    alpha = (iter + 0.0) / args.warmup_iter
    lr = ((1 - alpha) * args.warmup_factor + alpha) * args.lr
    global current_lr
    current_lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return current_lr


def accuracy(args, output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    if isinstance(output, tuple):
        output = output[0]

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def setup_logger(name, save_dir, distributed_rank, filename="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
