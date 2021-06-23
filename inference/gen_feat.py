#!/usr/bin/env python
import sys
sys.path.append("..")
sys.path.append("../../")

from utils import utils
from network_inf import builder_inf
import cv2
from termcolor import cprint
from torchvision import datasets
from torchvision import transforms
import torch.utils.data as data
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.data.distributed
import torchvision
import torch
import argparse
import numpy as np
import warnings
import time
import pprint
import os

# parse the args
cprint('=> parse the args ...', 'green')
parser = argparse.ArgumentParser(description='Trainer for posenet')
parser.add_argument('--arch', default='iresnet100', type=str,
                    help='backbone architechture')
parser.add_argument('--inf_list', default='', type=str,
                    help='the inference list')
parser.add_argument('--feat_list', type=str,
                    help='The save path for saveing features')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--embedding_size', default=512, type=int,
                    help='The embedding feature size')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--cpu-mode', action='store_true', help='Use the CPU.')
parser.add_argument('--dist', default=1, help='use this if model is trained with dist')
args = parser.parse_args()


class ImgInfLoader(data.Dataset):
    def __init__(self, ann_file, transform=None):
        self.ann_file = ann_file
        self.transform = transform
        cprint('=> preparing dataset for inference ...')
        self.init()

    def init(self):
        with open(self.ann_file) as f:
            self.imgs = f.readlines()

    def __getitem__(self, index):
        ls = self.imgs[index].strip().split()
        # change here
        img_path = ls[0]
        if not os.path.isfile(img_path):
            raise Exception('{} does not exist'.format(img_path))
            exit(1)
        img = cv2.imread(img_path)
        if img is None:
            raise Exception('{} is empty'.format(img_path))
            exit(1)
        _img = cv2.flip(img, 1)
        return [self.transform(img), self.transform(_img)], img_path

    def __len__(self):
        return len(self.imgs)


def main(args):
    cprint('=> torch version : {}'.format(torch.__version__), 'green')

    ngpus_per_node = torch.cuda.device_count()
    cprint('=> ngpus : {}'.format(ngpus_per_node), 'green')
    main_worker(ngpus_per_node, args)


def main_worker(ngpus_per_node, args):
    cprint('=> modeling the network ...', 'green')
    model = builder_inf(args)
    model = torch.nn.DataParallel(model)
    if not args.cpu_mode:
        model = model.cuda()

    cprint('=> building the dataloader ...', 'green')
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0., 0., 0.],
            std=[1., 1., 1.]),
    ])
    inf_dataset = ImgInfLoader(
        ann_file=args.inf_list,
        transform=trans
    )

    inf_loader = torch.utils.data.DataLoader(
        inf_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=False)

    cprint('=> starting inference engine ...', 'green')
    cprint('=> embedding features will be saved into {}'.format(args.feat_list))

    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')

    progress = utils.ProgressMeter(
        len(inf_loader),
        [batch_time, data_time],
        prefix="Extract Features: ")

    # switch to evaluate mode
    model.eval()

    fio = open(args.feat_list, 'w')
    with torch.no_grad():
        end = time.time()

        for i, (input, img_paths) in enumerate(inf_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # compute output
            embedding_feat = model(input[0])

            # embedding_feat = F.normalize(embedding_feat, p=2, dim=1)
            _feat = embedding_feat.data.cpu().numpy()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

            # write feat into files
            for feat, path in zip(_feat, img_paths):
                fio.write('{} '.format(path))
                for e in feat:
                    fio.write('{} '.format(e))
                fio.write('\n')
    # close
    fio.close()


if __name__ == '__main__':
    # parse the args
    cprint('=> parse the args ...', 'green')
    pprint.pprint(vars(args))
    main(args)
