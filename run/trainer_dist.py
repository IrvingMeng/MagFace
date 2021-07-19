#!/usr/bin/env python
import sys
sys.path.append("..")
from dataloader import dataloader_dist
from models import magface_dist

from utils import utils
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import numpy as np
import torchshard as ts
from collections import OrderedDict
from termcolor import cprint
from torchvision import datasets
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.parallel as parallel
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
parser.add_argument('--fp16', default=0, type=int,
                    help='whether use fp16')                    


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
parser.add_argument('--arc', default=1,type=int,
                    help='1 means use Mag-ArcFace, 0 means Mag-CosFace' )
parser.add_argument('--vis_mag', default=1, type=int,
                    help='visualize the magnitude against cos')

args = parser.parse_args()


def init_seeds(seed=0, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def main_worker(gpu, args):
    # check the feasible of the lambda g
    # args.arc_scale = 64 if args.arc else 30
    args.arc_scale = 64
    s = args.arc_scale
    k = (args.u_margin-args.l_margin)/(args.u_a-args.l_a)
    min_lambda = s*k*args.u_a**2*args.l_a**2/(args.u_a**2-args.l_a**2)
    color_lambda = 'red' if args.lambda_g < min_lambda else 'green'
    ngpus_per_node = torch.cuda.device_count()
    
    args.gpu = gpu
    args.rank = args.nr * args.gpus + args.gpu
    torch.cuda.set_device(gpu)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=args.rank)
    init_seeds(0+args.rank)  
    
    # logs
    if args.rank == 0:
        cprint('min lambda g is {}, currrent lambda is {}'.format(
        min_lambda, args.lambda_g), color_lambda)
        cprint('=> torch version : {}'.format(torch.__version__), 'green')
        cprint('=> ngpus : {}'.format(ngpus_per_node), 'green')
    
    # init torchshard
    ts.distributed.init_process_group(group_size=args.world_size)

    global best_acc1
    if args.rank == 0:
        cprint('=> modeling the network ...', 'green')
    model = magface_dist.builder(args)
    # for name, param in model.named_parameters():
    #     cprint(' : layer name and parameter size - {} - {}'.format(name, param.size()), 'green')

    if args.rank == 0:
        cprint('=> building the oprimizer ...', 'green')
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    
    if args.rank == 0:
        pprint.pprint(optimizer)
        cprint('=> building the dataloader ...', 'green')
        cprint('=> building the criterion ...', 'green')

    grad_scaler = GradScaler(enabled=args.amp_mode)
    train_loader = dataloader_dist.train_loader(args)
    from models.parallel_magloss import ParallelMagLoss
    criterion = ParallelMagLoss(
        args.l_a, args.u_a, args.l_margin, args.u_margin)

    global iters
    iters = 0
    if args.rank == 0:
        cprint('=> starting training engine ...', 'green')
    for epoch in range(args.start_epoch, args.epochs):
        global current_lr
        current_lr = utils.adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        do_train(train_loader, model, criterion, optimizer, grad_scaler, epoch, args)

        # ts.collect_state_dict() needs to see all the process groups
        state_dict = model.state_dict()
        state_dict = ts.collect_state_dict(model, state_dict)
    
        # save pth
        if epoch % args.pth_save_epoch == 0 and args.rank == 0:
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': state_dict,
                'optimizer': optimizer.state_dict(),
             }, False,
             filename=os.path.join(
                args.pth_save_fold, '{}.pth'.format(str(epoch+1).zfill(5)))
             )            
            cprint(' : save pth for epoch {}'.format(epoch + 1))


def do_train(train_loader, model, criterion, optimizer, grad_scaler, epoch, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.3f')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    learning_rate = utils.AverageMeter('LR', ':.4f')
    throughputs = utils.AverageMeter('ThroughPut', ':.2f')

    losses_id = utils.AverageMeter('L_ID', ':.3f')
    losses_mag = utils.AverageMeter('L_mag', ':.6f')
    progress_template = [batch_time, data_time, throughputs, 'images/s',
                         losses, losses_id, losses_mag, 
                         top1, learning_rate]

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
        with autocast(enabled=args.amp_mode):
            output, x_norm = model(input, target)
        
        # x_norm is not needed to be gathered, as feature x is in each rank
        target = ts.distributed.gather(target, dim=0)

        # loss
        with autocast(enabled=args.amp_mode):
            loss_id, loss_g, one_hot = criterion(output, 
                                                 target,
                                                 x_norm)
        loss = loss_id + args.lambda_g * loss_g
        # compute gradient and do solver step
        optimizer.zero_grad()

        # backward
        grad_scaler.scale(loss).backward()
        # update weights
        grad_scaler.step(optimizer)
        grad_scaler.update() 

        # syn for logging
        torch.cuda.synchronize()   

        # measure elapsed time
        if args.rank == 0:
            duration = time.time() - end
            end = time.time()
            batch_time.update(duration)
            bs = args.batch_size
            throughputs.update(args.world_size * bs / duration)

        # measure accuracy and record loss
        output = ts.distributed.gather(output[0], dim=-1)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))

        losses_id.update(loss_id.item(), input.size(0))
        losses_mag.update(args.lambda_g*loss_g.item(), input.size(0))

        if i % args.print_freq == 0 and args.rank == 0:
            progress.display(i)
            debug_info(x_norm, args.l_a, args.u_a,
                           args.l_margin, args.u_margin)


def debug_info(x_norm, l_a, u_a, l_margin, u_margin):
    """
    visualize the magnitudes and magins during training.
    Note: modify the function if m(a) is not linear
    """
    mean_ = torch.mean(x_norm).detach().cpu().numpy()
    max_ = torch.max(x_norm).detach().cpu().numpy()
    min_ = torch.min(x_norm).detach().cpu().numpy()
    m_mean_ = (u_margin-l_margin)/(u_a-l_a)*(mean_-l_a) + l_margin
    m_max_ = (u_margin-l_margin)/(u_a-l_a)*(max_-l_a) + l_margin
    m_min_ = (u_margin-l_margin)/(u_a-l_a)*(min_-l_a) + l_margin
    print('  [debug info]: x_norm mean: {:.2f} min: {:.2f} max: {:.2f}'
          .format(mean_, min_, max_))
    print('  [debug info]: margin mean: {:.2f} min: {:.2f} max: {:.2f}'
          .format(m_mean_, m_min_, m_max_))

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    args.gpus = 8
    args.nodes = 1
    args.nr = 0
    args.parallel_module_name = 'parallel_fc'
    args.amp_mode = True if args.fp16 else False    
    
    args.world_size = args.gpus * args.nodes                
    os.environ['MASTER_ADDR'] = '172.20.10.62'              
    os.environ['NCCL_SOCKET_IFNAME'] = 'enp97s0f0' 
    os.environ['MASTER_PORT'] = '12355'                     
    pprint.pprint(vars(args))
    if args.batch_size % args.world_size is not 0:
        print('batch size {} is not a multiplier of world size {}'.format(
            args.batch_size, args.world_size
        ))
        exit(1)
    args.batch_size = int(args.batch_size / args.world_size)
    mp.spawn(main_worker, nprocs=args.gpus, args=(args,))    
