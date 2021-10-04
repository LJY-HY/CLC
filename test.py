import os
import sys
import argparse
import time
import math

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from datasets.CIFAR import CIFAR10_noisy,CIFAR100_noisy
from main_ce import set_loader
from utils.util import AverageMeter
from utils.util import adjust_learning_rate, warmup_learning_rate, accuracy
from utils.util import set_optimizer, save_model

from networks.ResNet import SupConResNet, LinearClassifier

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--correction',type = str,default = 'none',choices = ['correction','none'])
    parser.add_argument('--correction_th',type = float,default = 0.05,choices = [0.05,0.03,0.01])

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10','cifar100'], help='dataset')
    parser.add_argument('--corruption_type',type = str, default='unif',choices=['unif','flip','flip2'])
    parser.add_argument('--noise_rate', type = float, default = 0.0, choices = [0.0,0.2,0.4,0.6,0.8], help='using cosine annealing')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--ckpt_model', type=str, default='',
                        help='path to pre-trained CNN')
    parser.add_argument('--ckpt_linear', type=str, default='',
                        help='path to pre-trained CNN')
    parser.add_argument('--pretrained',action='store_true')
    parser.add_argument('--trial', type = int , default= 0)
    parser.add_argument('--gpu_id', default='0', type=int, 
                        help='id(s) for CUDA_VISIBLE_DEVICES')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))
    if 'SimCLR' in opt.ckpt_model:
        opt.method = 'SimCLR'
    else:
        opt.method = 'SupCon'
    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt


def set_model(opt):
    model = SupConResNet(name=opt.model, pretrained = opt.pretrained)
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt_model = torch.load(opt.ckpt_model, map_location='cpu')
    
    state_dict = ckpt_model['model']
    if opt.pretrained==False:
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict

    model = model.cuda()
    criterion = criterion.cuda()
    cudnn.benchmark = True

    if opt.pretrained==False:
        model.load_state_dict(state_dict)

    ckpt_linear = torch.load(opt.ckpt_linear, map_location='cpu')
    state_dict = ckpt_linear['model']
   
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v
    state_dict = new_state_dict
    classifier = classifier.cuda()
    cudnn.benchmark = True

    classifier.load_state_dict(state_dict)

    return model, classifier, criterion

def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def main():
    best_acc = 0
    opt = parse_option()

    opt.device = torch .device('cuda',opt.gpu_id)
    torch.cuda.set_device(opt.device)

    # build data loader
    _, val_loader = set_loader(opt)

    model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    # test
    adjust_learning_rate(opt, optimizer, 1)

    # eval for one epoch
    loss, val_acc = validate(val_loader, model, classifier, criterion, opt)

    print('best accuracy: {:.2f}'.format(val_acc))


if __name__ == '__main__':
    main()