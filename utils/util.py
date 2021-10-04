from __future__ import print_function
import os
import math
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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
            correct = correct.contiguous()
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

def mixup(opt,images,labels):
    lam = np.random.beta(1,1)
    bsz = images.shape[0]
    index = torch.randperm(bsz).cuda()
    mixed_images = lam*images + (1-lam)*images[index,:]
    labels_a, labels_b = labels, labels[index]
    return mixed_images, labels_a, labels_b, lam

def mixup_criterion(output, corrected_labels_a, corrected_labels_b, lam):
    criterion = torch.nn.CrossEntropyLoss()
    return lam*criterion(output,corrected_labels_a) + (1-lam)*criterion(output,corrected_labels_b)

def histogram(opt, file1_path, file2_path, epoch):
    if not os.path.exists('./figures'):
        os.makedirs('./figures')
    result_path = opt.dataset + '_NR_' + str(opt.noise_rate) + '_th_'+str(opt.correction_th)+'_corruption_type_'+opt.corruption_type+'_correction_method_'+opt.correction_method
    file1 = np.loadtxt(file1_path)
    file2 = np.loadtxt(file2_path)

    file1_min,file1_max = file1.min(), file1.max()
    file2_min,file2_max = file2.min(), file2.max()
    min_ = min(file1_min,file2_min)
    max_ = min(max(file1_max,file2_max),10)
    gap = (max_-min_)/100
    bins = np.arange(min_,max_+1,gap)
    file1_hist, bins = np.histogram(file1,bins)
    file2_hist, bins = np.histogram(file2,bins)

    feature_loss_dataset_epoch = file1_path.split('/')[2]
    feature = feature_loss_dataset_epoch.split('_')[0]
    plt.hist(file1,range=(min_,max_),bins=100,alpha=0.7,color='blue')
    plt.hist(file2,range=(min_,max_),bins=100,alpha=0.7,color = 'red')
    plt.axis([min_,max_,0,max(max(file1_hist),max(file2_hist))])
    plt.savefig('./figures/NR_'+str(opt.noise_rate)+'/epoch_'+str(epoch)+'_'+feature+'_'+result_path+'.png')
    plt.clf()

def histogram_multiple(opt, file1_path, file2_path, file3_path, file4_path, epoch):
    if not os.path.exists('./figures'):
        os.makedirs('./figures')
    result_path = opt.dataset + '_NR_' + str(opt.noise_rate) + '_th_'+str(opt.correction_th)+'_corruption_type_'+opt.corruption_type+'_correction_method_'+opt.correction_method
    file1 = np.loadtxt(file1_path)
    file2 = np.loadtxt(file2_path)
    file3 = np.loadtxt(file3_path)
    file4 = np.loadtxt(file4_path)

    file1_min,file1_max = file1.min(), file1.max()
    file2_min,file2_max = file2.min(), file2.max()
    file3_min,file3_max = file3.min(), file3.max()
    file4_min,file4_max = file4.min(), file4.max()

    min_ = min(file1_min,file2_min,file3_min,file4_min)
    max_ = min(max(file1_max,file2_max,file3_max,file4_max),10)
    gap = (max_-min_)/100
    bins = np.arange(min_,max_+1,gap)
    file1_hist, bins = np.histogram(file1,bins)
    file2_hist, bins = np.histogram(file2,bins)
    file3_hist, bins = np.histogram(file3,bins)
    file4_hist, bins = np.histogram(file4,bins)

    feature1_feature2_loss_dataset_epoch = file1_path.split('/')[2]
    feature1 = feature1_feature2_loss_dataset_epoch.split('_')[0]
    feature2 = feature1_feature2_loss_dataset_epoch.split('_')[1]
    plt.hist(file1,range=(min_,max_),bins=100,alpha=0.7,color='royalblue')
    plt.hist(file2,range=(min_,max_),bins=100,alpha=0.7,color='midnightblue')
    plt.hist(file3,range=(min_,max_),bins=100,alpha=0.7,color='lightcoral')
    plt.hist(file4,range=(min_,max_),bins=100,alpha=0.7,color='darkred')
    # plt.hist([file1,file2,file3,file4],range=(min_,max_),bins=100,alpha=0.7,histtype='barstacked')
    plt.axis([min_,max_,0,max(max(file1_hist),max(file2_hist),max(file3_hist),max(file4_hist))])
    plt.savefig('./figures/NR_'+str(opt.noise_rate)+'/epoch_'+str(epoch)+'_'+feature1+'_'+feature2+'_'+result_path+'.png')
    plt.clf()

def plot_loss(opt,model,classifier,train_loader,epoch):
    if not os.path.exists('./loss_value'):
        os.makedirs('./loss_value')
    clean_path = "./loss_value/CLEAN_loss_{dataset}_{epoch}.txt".format(dataset = opt.dataset,epoch=epoch)
    noisy_path = "./loss_value/NOISY_loss_{dataset}_{epoch}.txt".format(dataset = opt.dataset,epoch=epoch)
    easy_path = "./loss_value/EASY_loss_{dataset}_{epoch}.txt".format(dataset = opt.dataset,epoch=epoch)
    hard_path = "./loss_value/HARD_loss_{dataset}_{epoch}.txt".format(dataset = opt.dataset,epoch=epoch)

    clean_easy_path = "./loss_value/CLEAN_EASY_loss_{dataset}_{epoch}.txt".format(dataset = opt.dataset,epoch=epoch)
    clean_hard_path = "./loss_value/CLEAN_HARD_loss_{dataset}_{epoch}.txt".format(dataset = opt.dataset,epoch=epoch)
    noisy_easy_path = "./loss_value/NOISY_EASY_loss_{dataset}_{epoch}.txt".format(dataset = opt.dataset,epoch=epoch)
    noisy_hard_path = "./loss_value/NOISY_HARD_loss_{dataset}_{epoch}.txt".format(dataset = opt.dataset,epoch=epoch)

    clean_file = open(clean_path,'w')
    noisy_file = open(noisy_path,'w')
    easy_file = open(easy_path,'w')
    hard_file = open(hard_path,'w')
    
    clean_easy_file = open(clean_easy_path,'w')
    clean_hard_file = open(clean_hard_path,'w')
    noisy_easy_file = open(noisy_easy_path,'w')
    noisy_hard_file = open(noisy_hard_path,'w')

    for _, (images, labels, _, true_labels) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        with torch.no_grad():
            XentLoss_ = nn.CrossEntropyLoss(reduction = 'none')
            loss_ = XentLoss_(output,labels)

        for idx, loss_idx in enumerate(loss_):
            if labels[idx]==true_labels[idx]:
                clean_file.write("{}\n".format(loss_idx))
            else:
                noisy_file.write("{}\n".format(loss_idx))

        for idx, loss_idx in enumerate(loss_):
            if torch.argmax(output[idx])==true_labels[idx]:
                easy_file.write("{}\n".format(loss_idx))
            else:
                hard_file.write("{}\n".format(loss_idx))

        for idx, loss_idx in enumerate(loss_):
            if labels[idx]==true_labels[idx] and torch.argmax(output[idx])==true_labels[idx]:
                clean_easy_file.write("{}\n".format(loss_idx))
            elif labels[idx]==true_labels[idx] and torch.argmax(output[idx])!=true_labels[idx]:
                clean_hard_file.write("{}\n".format(loss_idx))
            elif labels[idx]!=true_labels[idx] and torch.argmax(output[idx])==true_labels[idx]:
                noisy_easy_file.write("{}\n".format(loss_idx))
            elif labels[idx]!=true_labels[idx] and torch.argmax(output[idx])!=true_labels[idx]:
                noisy_hard_file.write("{}\n".format(loss_idx))
    clean_file.close()
    noisy_file.close()
    easy_file.close()
    hard_file.close()

    clean_easy_file.close()
    clean_hard_file.close()
    noisy_easy_file.close()
    noisy_hard_file.close()

    histogram(opt,clean_path, noisy_path, epoch)
    histogram(opt,easy_path, hard_path, epoch)
    histogram_multiple(opt,clean_easy_path, clean_hard_path, noisy_easy_path, noisy_hard_path, epoch)