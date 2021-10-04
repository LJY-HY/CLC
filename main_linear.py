import os
import sys
import argparse
import time
import math

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from main_ce import set_loader
from utils.util import AverageMeter
from utils.util import adjust_learning_rate, warmup_learning_rate, accuracy
from utils.util import set_optimizer, save_model, plot_loss, mixup, mixup_criterion

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
    parser.add_argument('--correction_th',type = float,default = 0.00,choices = [0.05,0.03,0.01,0.00])

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10','cifar100'], help='dataset')
    parser.add_argument('--corruption_type',type=str,default='unif',choices=['unif','flip','flip2'])
    parser.add_argument('--noise_rate', type = float, default = 0.0, choices = [0.0,0.2,0.4,0.5,0.6,0.8], help='using cosine annealing')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--gpu_id', default='0', type=int, 
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--pretrained',action='store_true')
    parser.add_argument('--correction_method',type=str,default='quadratic',choices=['quadratic','None'])
    parser.add_argument('--trial',type = int, default=0)
    parser.add_argument('--partial',action='store_true')
    parser.add_argument('--plotting',action='store_true')
    parser.add_argument('--mixup',action='store_true')
    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))
    if 'SimCLR' in opt.ckpt:
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

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    if opt.pretrained == False:
        state_dict = ckpt['model']
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
    model = model.cuda()
    classifier = classifier.cuda()
    criterion = criterion.cuda()
    cudnn.benchmark = True

    if opt.pretrained==False:
        model.load_state_dict(state_dict)

    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()
    classifier.train()

    if opt.partial:
        layer_count = 0
        for child in model.encoder.children():
            layer_count+=1
            if layer_count<int(len(list(model.encoder.children()))*0.75):
                child.eval()
                for param in child.parameters():
                    param.requires_grad=False

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    for idx, (images, labels, index, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        with torch.no_grad():
            XentLoss_ = nn.CrossEntropyLoss(reduction = 'none')
            loss_ = XentLoss_(output,labels)
            if epoch < opt.epochs and opt.correction=='correction':
                # self correction
                if opt.correction_method == 'quadratic':
                    corrected_labels = torch.where(loss_>sorted(loss_)[int(images.shape[0]*(1-opt.correction_th* (1-epoch/opt.epochs)**2)-1)], output.argmax(dim=1), labels)
                elif opt.correction_method == 'None':
                    corrected_labels = torch.where(loss_>sorted(loss_)[int(images.shape[0]*(1-opt.correction_th)-1)], output.argmax(dim=1), labels)
                for num, idxs in enumerate(index):
                    train_loader.dataset.train_labels[idxs] = corrected_labels[num]
            else:
                corrected_labels = labels
        if opt.mixup:
            mixed_images,corrected_labels_a, corrected_labels_b, lam = mixup(opt,images,corrected_labels)
            output = classifier(model.encoder(mixed_images))
            loss = mixup_criterion(output, corrected_labels_a, corrected_labels_b, lam)
        else:
            loss = criterion(output, corrected_labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        for param_group in optimizer.param_groups:
            lr_ = param_group['lr']
        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'lr {lr:.4f}'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, lr = lr_))
            sys.stdout.flush()
    return losses.avg, top1.avg


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
    train_loader, val_loader = set_loader(opt)
    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    # set save folder path
    opt.model_path = './save/Linear/{}_models'.format(opt.dataset)
    if opt.partial:
        opt.save_folder = os.path.join(opt.model_path, opt.model_name, opt.corruption_type,'partial')
        if opt.correction == 'none':
            opt.save_folder = os.path.join(opt.model_path, opt.model_name, opt.corruption_type,'partial_NoCorrection')
    elif opt.pretrained:
        opt.save_folder = os.path.join(opt.model_path, opt.model_name, opt.corruption_type,'pretrained')
    else:
        opt.save_folder = os.path.join(opt.model_path, opt.model_name, opt.corruption_type,'plain')

    if opt.partial:
        opt.save_folder_model = '/'.join(opt.ckpt.split('/')[:-1])
        opt.save_folder_model = os.path.join(opt.save_folder_model,opt.corruption_type,'partial')
        if opt.correction == 'none':
            opt.save_folder_model = '/'.join(opt.ckpt.split('/')[:-1])
            opt.save_folder_model = os.path.join(opt.save_folder_model,opt.corruption_type,'partial_NoCorrection')
        opt.save_folder_model = os.path.join(opt.save_folder_model, opt.correction_method+'_trial_'+str(opt.trial))
        if not os.path.isdir(opt.save_folder_model):
            os.makedirs(opt.save_folder_model)

    if opt.mixup:
        opt.save_folder = os.path.join(opt.save_folder,'mixup')
        opt.save_folder_model = os.path.join(opt.save_folder_model,'mixup')
    opt.save_folder = os.path.join(opt.save_folder,opt.correction_method+'_trial_'+str(opt.trial))
    
    if not os.path.isdir(opt.save_folder_model):
        os.makedirs(opt.save_folder_model)
    

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc))

        if opt.plotting:
            if epoch in [1,5,10,15,20,40,60,80]:
                plot_loss(opt,model,classifier,train_loader,epoch)

        # eval for one epoch
        loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
        if val_acc > best_acc:
            best_acc = val_acc
            save_file = os.path.join(opt.save_folder, 'linear_model_NR_{noise_rate}_th_{threshold}.pth'.format(noise_rate = opt.noise_rate,threshold=opt.correction_th))
            if opt.partial:
                save_file_model = os.path.join(opt.save_folder_model,'ckpt_epoch_1000_NR_{noise_rate}_th_{threshold}.pth'.format(noise_rate = opt.noise_rate, threshold = opt.correction_th))
                save_model(model,optimizer,opt,epoch,save_file_model)    
            save_model(classifier, optimizer, opt, epoch, save_file)


    print('best accuracy: {:.2f}'.format(best_acc))


if __name__ == '__main__':
    main()