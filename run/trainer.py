#!/usr/bin/env python
import sys
sys.path.append("..")
from dataloader import dataloader
from models import magface
from utils import utils
import numpy as np
from collections import OrderedDict
from termcolor import cprint
from torchvision import datasets
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torch
import argparse
import random
import warnings
import time
import pprint
import os


warnings.filterwarnings("ignore")


# parse the args
cprint('=> parse the args ...', 'green')
parser = argparse.ArgumentParser(description='Trainer for Magface')
parser.add_argument('--arch', default='resnet100', type=str,
                    help='backbone architechture')
parser.add_argument('--train_list', default='', type=str,
                    help='')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--embedding-size', default=512, type=int,
                    help='The embedding feature size')
parser.add_argument('--last-fc-size', default=1000, type=int,
                    help='The num of last fc layers for using softmax')


parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--lr-drop-epoch', default=[30, 60, 90], type=int, nargs='+',
                    help='The learning rate drop epoch')
parser.add_argument('--lr-drop-ratio', default=0.1, type=float,
                    help='The learning rate drop ratio')

parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--pth-save-fold', default='tmp', type=str,
                    help='The folder to save pths')
parser.add_argument('--pth-save-epoch', default=1, type=int,
                    help='The epoch to save pth')


# magface parameters
parser.add_argument('--l_a', default=10, type=float,
                    help='lower bound of feature norm')
parser.add_argument('--u_a', default=110, type=float,
                    help='upper bound of feature norm')
parser.add_argument('--l_margin', default=0.45,
                    type=float, help='low bound of margin')
parser.add_argument('--u_margin', default=0.8, type=float,
                    help='the margin slop for m')
parser.add_argument('--lambda_g', default=20, type=float,
                    help='the lambda for function g')
parser.add_argument('--arc-scale', default=64, type=int,
                    help='scale for arcmargin loss')
parser.add_argument('--vis_mag', default=1, type=int,
                    help='visualize the magnitude against cos')

args = parser.parse_args()


def main(args):
    # check the feasible of the lambda g
    s = 64
    k = (args.u_margin-args.l_margin)/(args.u_a-args.l_a)
    min_lambda = s*k*args.u_a**2*args.l_a**2/(args.u_a**2-args.l_a**2)
    color_lambda = 'red' if args.lambda_g < min_lambda else 'green'
    cprint('min lambda g is {}, currrent lambda is {}'.format(
        min_lambda, args.lambda_g), color_lambda)

    cprint('=> torch version : {}'.format(torch.__version__), 'green')
    ngpus_per_node = torch.cuda.device_count()
    cprint('=> ngpus : {}'.format(ngpus_per_node), 'green')

    main_worker(ngpus_per_node, args)


def main_worker(ngpus_per_node, args):
    global best_acc1

    cprint('=> modeling the network ...', 'green')
    model = magface.builder(args)
    model = torch.nn.DataParallel(model).cuda()
    # for name, param in model.named_parameters():
    #     cprint(' : layer name and parameter size - {} - {}'.format(name, param.size()), 'green')

    cprint('=> building the oprimizer ...', 'green')
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    pprint.pprint(optimizer)

    cprint('=> building the dataloader ...', 'green')
    train_loader = dataloader.train_loader(args)

    cprint('=> building the criterion ...', 'green')
    criterion = magface.MagLoss(
        args.l_a, args.u_a, args.l_margin, args.u_margin)

    global iters
    iters = 0

    cprint('=> starting training engine ...', 'green')
    for epoch in range(args.start_epoch, args.epochs):

        global current_lr
        current_lr = utils.adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        do_train(train_loader, model, criterion, optimizer, epoch, args)

        # save pth
        if epoch % args.pth_save_epoch == 0:
            state_dict = model.state_dict()

            utils.save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': state_dict,
                'optimizer': optimizer.state_dict(),
            }, False,
                filename=os.path.join(
                args.pth_save_fold, '{}.pth'.format(
                    str(epoch+args.start_epoch).zfill(5))
            ))
            cprint(' : save pth for epoch {}'.format(epoch + 1))


def do_train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.3f')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    learning_rate = utils.AverageMeter('LR', ':.4f')

    losses_id = utils.AverageMeter('L_ID', ':.3f')
    losses_mag = utils.AverageMeter('L_mag', ':.6f')
    progress_template = [batch_time, data_time, losses, losses_id, losses_mag,
                         top1, top5, learning_rate]

    progress = utils.ProgressMeter(
        len(train_loader),
        progress_template,
        prefix="Epoch: [{}]".format(epoch))
    end = time.time()

    # update lr
    learning_rate.update(current_lr)

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        global iters
        iters += 1

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output, x_norm = model(input, target)

        loss_id, loss_g, one_hot = criterion(output, target, x_norm)
        loss = loss_id + args.lambda_g * loss_g

        # measure accuracy and record loss
        acc1, acc5 = utils.accuracy(args, output[0], target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        losses_id.update(loss_id.item(), input.size(0))
        losses_mag.update(args.lambda_g*loss_g.item(), input.size(0))

        # compute gradient and do solver step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        if args.vis_mag:
            if (i > 10000) and (i % 100 == 0):
                x_norm = x_norm.detach().cpu().numpy()
                cos_theta = torch.masked_select(
                    output[0], one_hot.bool()).detach().cpu().numpy()
                logit = torch.masked_select(
                    F.softmax(output[0]), one_hot.bool()).detach().cpu().numpy()
                np.savez('{}/vis/epoch_{}_iter{}'.format(args.pth_save_fold, epoch, i),
                         x_norm, logit, cos_theta)


if __name__ == '__main__':

    pprint.pprint(vars(args))
    main(args)
