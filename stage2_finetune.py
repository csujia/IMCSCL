
# this is used for fine tune by clean data
# to test the quality of representations

from __future__ import print_function
import datetime
import os
import sys
import argparse
import time
import math
import numpy as np
import json
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from torchvision.datasets import CIFAR10, CIFAR100
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import tensorboard_logger as tb_logger
from util import store_setup_txt, Logger # 保存参数和输出日志
from util import AverageMeter, Recording, save_model
from util import adjust_learning_rate, warmup_learning_rate, accuracy, create_lr_scheduler
from util import set_optimizer_ce
from networks.resnet_big import SupConResNet, LinearClassifier, FineTune, SupCEResNet,SupConResNet2,FineTune3,FineTune4
# from 711_train_ce2 import set_loader
import pandas as pd
from tools import *
from MyDataset import MyDataset
from MyLoss import LDAMLoss, FocalLoss, IBLoss, IB_FocalLoss
from unbalanced_loss.focal_loss import MultiFocalLoss
from unbalanced_loss.poly import PolyLoss
# 评价
from sklearn.metrics import precision_recall_curve, average_precision_score,roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
#绘制热力图
from metric import plot_cm_hmap
try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

import warnings
warnings.filterwarnings("ignore")

def parse_option():
    parser = argparse.ArgumentParser('argument for fine tune on clean dataset')

    # noisy dataset and saved model for fine tune
    # please note that if you choose NoisySL pretrained model, you should keep other params same as the model folder
    # parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='dataset')
    parser.add_argument('--ckpt', type=str, default='/home/work/Jupyter_Notebook/Ve9eD0g/supcon/716/0708_byol2-main/BYOL-2/save/SimCLR/path_models/SimCLR_path_resnet50_lr_0.5_decay_0.0001_bsz_64_temp_0.5_trial_0/last.pth', help='path to pre-trained model')
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'])
    parser.add_argument('--sslorsl', type=str, default='ssl', choices=['ssl', 'sl'], help='separate recordings of SSL and SL')
    parser.add_argument('--appendix', type=str, default='ssl_pretrained or noisetype_rate_pretrained_try', help='appendix for recording')

    parser.add_argument('--dataset', type=str, default='path', choices=['cifar10', 'cifar100', 'path'],
                        help='dataset')
    parser.add_argument('--data_folder', type=str,
                        default='/home/work/Jupyter_Notebook/Ve9eD0g/Img/Train/Mal_ASMNgram_MAT_RGB',
                        help='path to custom dataset')  # 如果数据不是cifar10/100
    parser.add_argument('--mean', type=str, default='(0.98960114, 0.9664513, 0.94427776)',
                        help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, default='(0.052279416, 0.046389017, 0.034104187)',
                        help='std of dataset in path in form of str tuple')
    parser.add_argument('--mean2', type=str, default='(0.99018216, 0.9685766, 0.9477592)',
                        help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std2', type=str, default='(0.050820418, 0.044482365, 0.03263167)',
                        help='std of dataset in path in form of str tuple')
    parser.add_argument('--size', type=int, default=56,
                        help='parameter for RandomResizedCrop')  # 这里如果data image的size变了也要改动哦
    # train loader percent
    parser.add_argument('--percent', type=int, default=20, help='trainloader percent')
    # loss type and rule
    parser.add_argument('--train_rule', default='None', type=str,
                        help='data sampling strategy for train loader：CBReweight/IBReweight')
    # loss two stage
    parser.add_argument('--loss_typea', default="CE", type=str, help='loss type:CE,Focal,IB,IBFocal')
    parser.add_argument('--loss_typeb', default="Poly", type=str, help='loss type')
    # loss分阶段
    parser.add_argument('--start_ib_epoch', default=0, type=int, help='start epoch for IB Loss')

    # training
    parser.add_argument('--epochs', type=int, default=120, help='number of training epochs')
    parser.add_argument('--device', default='0', help="gpu used in training")
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    # parser.add_argument('--lr_decay_epochs', type=str, default='100,120', help='where to decay lr, can be a list')
    # parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')

    # other setting
    parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')
    parser.add_argument('--optimizer_type', default='sgd', choices=['sgd', 'adam','adamw'])
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')
    parser.add_argument('--warm', action='store_true', help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0', help='id for recording multiple runs')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=100, help='save frequency')

    opt = parser.parse_args()

    # set the path according to the environment
    # opt.data_folder = '/home/work/Jupyter_Notebook/Ve9eD0g/Img/Train/Mal_ASMNgram_MAT_RGB'
    cur_time = datetime.datetime.now().strftime('%y%m%d-%H_%M_%S')
    opt.model_name = 'SupFinetune_{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}'. \
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size, opt.trial)  # 在每个里面又分SupCE / SimCLR
    opt.model_name = './save/SupFineTune/{}_{}'.format(opt.model_name, cur_time)
    if not os.path.isdir(opt.model_name):
        os.makedirs(opt.model_name)
    opt.model_path = '{}_models'.format(opt.dataset)
    opt.tb_path = '{}_tensorboard'.format(opt.dataset)
    opt.log_path = '{}_Logs'.format(opt.dataset)
    opt.metric_path = '{}_Metric'.format(opt.dataset)

    opt.save_folder = '{}/{}'.format(opt.model_name, opt.model_path)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    opt.tb_folder = os.path.join(opt.model_name, opt.tb_path)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.log_folder = os.path.join(opt.model_name, opt.log_path)
    if not os.path.isdir(opt.log_folder):
        os.makedirs(opt.log_folder)

    opt.metric_folder = os.path.join(opt.model_name, opt.metric_path)
    if not os.path.isdir(opt.metric_folder):
        os.makedirs(opt.metric_folder)

    # opt.model_path = './save/clean_FT/{}/{}_models'.format(opt.sslorsl, opt.dataset)  # 储存模型的位置
    # opt.tb_path = './save/clean_FT/{}/{}_tensorboard'.format(opt.sslorsl, opt.dataset)  # 储存训练的tb结果的位置
    # opt.log_path = './save/clean_FT/{}/{}_logs'.format(opt.sslorsl, opt.dataset)  # log记录结果
    # opt.model_name = '{}_{}_{}'.format(opt.dataset, opt.model, opt.appendix)  # 模型的名字

    # gpus = opt.workers.split(',')
    # opt.workers = list([])
    # for it in gpus:
    #     opt.workers.append(int(it))
    server = "cuda:{}".format(int(opt.device))
    opt.device = torch.device(server if torch.cuda.is_available() else "cpu")

    # iterations = opt.lr_decay_epochs.split(',')
    # opt.lr_decay_epochs = list([])
    # for it in iterations:
    #     opt.lr_decay_epochs.append(int(it))

    # if opt.warm:
    #     opt.warmup_from = 0.01
    #     opt.warm_epochs = 10
    #     if opt.cosine:
    #         eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
    #         opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
    #                 1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
    #     else:
    #         opt.warmup_to = opt.learning_rate



    opt.n_cls = 9


    # # 最后创建文件夹 别管
    # opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    # if not os.path.isdir(opt.tb_folder):
    #     os.makedirs(opt.tb_folder)
    # opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    # if not os.path.isdir(opt.save_folder):
    #     os.makedirs(opt.save_folder)
    # opt.log_folder = os.path.join(opt.log_path, opt.model_name)
    # if not os.path.isdir(opt.log_folder):
    #     os.makedirs(opt.log_folder)

    return opt


# def set_loader(opt):
#     '''set dataloader with noisy dataset'''
#     if opt.dataset == 'cifar10':
#         mean = (0.4914, 0.4822, 0.4465)
#         std = (0.2023, 0.1994, 0.2010)
#     elif opt.dataset == 'cifar100':
#         mean = (0.5071, 0.4867, 0.4408)
#         std = (0.2675, 0.2565, 0.2761)
#     else:
#         raise ValueError('dataset not supported: {}'.format(opt.dataset))
#     normalize = transforms.Normalize(mean=mean, std=std)
#
#     train_transform = transforms.Compose([
#         transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         normalize,
#     ])
#
#     val_transform = transforms.Compose([
#         transforms.ToTensor(),
#         normalize,
#     ])
#
#     if opt.dataset == 'cifar10':
#         train_dataset = CIFAR10(root=opt.data_folder, train=True, download=True, transform=train_transform)
#         val_dataset = CIFAR10(root=opt.data_folder, train=False, download=True, transform=val_transform)
#     elif opt.dataset == 'cifar100':
#         train_dataset = CIFAR100(root=opt.data_folder, train=True, download=True, transform=train_transform)
#         val_dataset = CIFAR100(root=opt.data_folder, train=False, download=True, transform=val_transform)
#
#     # train_data, val_data, train_noisy_labels, val_noisy_labels, train_clean_labels, val_clean_labels = \
#     #     dataset_split(np.array(train_val_dataset.data), np.array(train_val_dataset.targets), opt.noise_rate,
#     #                   opt.noise_type, opt.split_per, opt.seed, opt.n_cls)
#     # train_dataset = Train_Dataset(train_data, train_noisy_labels, train_clean_labels, train_transform)
#     # val_dataset = Train_Dataset(val_data, val_noisy_labels, val_clean_labels, val_transform)
#
#     samples = list(range(0, int(len(train_dataset) * opt.percent / 100)))
#     subset = torch.utils.data.Subset(train_dataset, samples)
#     train_loader = torch.utils.data.DataLoader(dataset=subset, batch_size=opt.batch_size, shuffle=True,
#                                                num_workers=opt.num_workers, pin_memory=False)
#     val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=256, num_workers=8, shuffle=False,
#                                              pin_memory=False)
#
#     return train_loader, val_loader
def set_loader(opt):
    # construct training and validation data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     normalize,
    # ])
    #
    # val_transform = transforms.Compose([
    #     transforms.CenterCrop(opt.size),
    #     transforms.ToTensor(),
    #     normalize,
    # ])

    size = opt.size
    train_transform = transforms.Compose([
        transforms.Resize((int(size * (256 / 224)), int(size * (256 / 224)))),
        transforms.RandomCrop(size),

        transforms.RandomHorizontalFlip(),
        # RandomRotate(15, 0.3),
        # RandomGaussianBlur(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize((int(size * (256 / 224)), int(size * (256 / 224)))),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        normalize,
    ])




    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        train=False,
                                        transform=val_transform)
    elif opt.dataset == 'path1':
        train_dataset = datasets.ImageFolder(root=opt.data_folder+'/train100',
                                            transform=train_transform)
        val_dataset = datasets.ImageFolder(root=opt.data_folder+'/val',
                                        transform=val_transform)
    elif opt.dataset == 'path':
        train_dataset = MyDataset(opt.data_folder+'/train100', transform=train_transform)
        val_dataset = MyDataset(opt.data_folder+'/val', transform=val_transform)

        with open('class.txt', 'w') as file:
            file.write(str(train_dataset.class_to_id))
        with open('class.json', 'w', encoding='utf-8') as file:
            file.write(json.dumps(train_dataset.class_to_id))

    else:
        raise ValueError(opt.dataset)


    cls_per = train_dataset.num_per_cls_list

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=8, pin_memory=True)

    return train_loader, val_loader,cls_per


def set_model(opt):
    if opt.sslorsl == 'ssl':
        model = SupConResNet2(name=opt.model)

    else:
        model = SupCEResNet(name=opt.model, num_classes=opt.n_cls)
    # criterion = torch.nn.CrossEntropyLoss()
    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    model.load_state_dict(ckpt)
    # state_dict = ckpt['model']

    if torch.cuda.is_available():
        # if opt.sslorsl=='ssl':  # 因为之前是在多张卡上面训练的，这里读取的时候需要做一点点改动
        #     new_state_dict = {}
        #     for k, v in state_dict.items():
        #         k = k.replace("module.", "")
        #         new_state_dict[k] = v
        #     state_dict = new_state_dict
        model =  model.to(opt.device)
        classifier = classifier.to(opt.device)
        # criterion = criterion.to(opt.device)
        cudnn.benchmark = True




    return model, classifier


def train(train_loader, model, criterion, criterion_ib, optimizer,lr_scheduler, epoch, opt):
    """one epoch training for fine tuning"""
    model.train() # 这里的model指的是包括encoder和classifier的整个model

    losses = AverageMeter()
    top1 = AverageMeter()

    for idx, (images, labels) in enumerate(train_loader):

        images = images.to(opt.device, non_blocking=True)
        labels = labels.to(opt.device, non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        # output = model(images)
        # loss = criterion(output, labels)

        if epoch < opt.start_ib_epoch:
            if 'IB' in opt.loss_typea:
                pred, features = model(images)
                loss = criterion(pred, labels, features)

            else:
                pred, _ = model(images)
                loss = criterion(pred, labels)
        else:
            if 'IB' in opt.loss_typeb:
                pred, features = model(images)
                loss = criterion_ib(pred, labels, features)

            else:
                pred, _ = model(images)
                loss = criterion_ib(pred, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1 = accuracy(pred, labels, topk=(1, ))
        top1.update(acc1[0], bsz)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    return losses.avg, top1.avg, optimizer.param_groups[0]["lr"]# 输出的这个值才有意义


def validate(val_loader, model, criterion, criterion_ib, epoch, opt):
    """validation"""
    model.eval()
    val_list = []
    pred_list = []
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        for idx, (images, clean_labels) in enumerate(val_loader):  # 计算的是和clean label比较的结果
            for t in clean_labels:
                val_list.append(t.data.item())
            images = images.float().to(opt.device)
            clean_labels = clean_labels.to(opt.device)
            bsz = clean_labels.shape[0]

            # forward
            # output = model(images)
            # loss = criterion(output, clean_labels)

            if epoch < opt.start_ib_epoch:
                if 'IB' in opt.loss_typea:
                    pred, features = model(images)
                    loss = criterion(pred, clean_labels, features)

                else:
                    pred, _ = model(images)
                    loss = criterion(pred, clean_labels)
            else:
                if 'IB' in opt.loss_typeb:
                    pred, features = model(images)
                    loss = criterion_ib(pred, clean_labels, features)

                else:
                    pred, _ = model(images)
                    loss = criterion_ib(pred, clean_labels)
            _, yuce = torch.max(pred.data.cpu(), 1)
            for y in yuce:
                pred_list.append(y.data.item())
            # update metric
            losses.update(loss.item(), bsz)
            acc1 = accuracy(pred, clean_labels, topk=(1, ))  # acc5没有用到
            top1.update(acc1[0], bsz)

    return val_list, pred_list, losses.avg, top1.avg


# 加上计算label precision和noise detect的validation
# def validate_cal(val_loader, model, criterion, opt):
#     """validation"""
#     model.eval()
#
#     # losses = AverageMeter()
#     top1 = AverageMeter()
#     # ABCD: A:target=pred, target=clean; B:target=pred, target=noisy;
#     # C: target!=pred, target=clean; D: target!=pred, target=noisy
#     A = AverageMeter()
#     B = AverageMeter()
#     C = AverageMeter()
#     D = AverageMeter()
#
#     with torch.no_grad():
#         for idx, (images, labels, clean_labels) in enumerate(val_loader):
#             images = images.float().to(opt.device)
#             labels = labels.to(opt.device)
#             clean_labels = clean_labels.to(opt.device)
#             bsz = clean_labels.shape[0]
#
#             # forward
#             output = model(images)
#             # loss = criterion(output, clean_labels)
#             _, pred = torch.max(output.data, 1) # 预测的label
#
#             # update metric
#             # losses.update(loss.item(), bsz)
#             acc1, acc5 = accuracy(output, clean_labels, topk=(1, 5))
#             top1.update(acc1[0], bsz)
#             A.update(int(((pred == labels) & (labels == clean_labels)).sum()))
#             B.update(int(((pred == labels) & (labels != clean_labels)).sum()))
#             C.update(int(((pred != labels) & (labels == clean_labels)).sum()))
#             D.update(int(((pred != labels) & (labels != clean_labels)).sum()))
#
#     label_precision_rate = A.sum/(A.sum+B.sum)
#     clean_selection_num = A.sum
#     noise_detection_rate = D.sum/(C.sum+D.sum)
#     noise_detect_num = D.sum
#
#     return top1.avg, label_precision_rate, clean_selection_num, noise_detection_rate, noise_detect_num


def main():
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    opt = parse_option()
    best_acc = 0
    best_epoch=0
    # start recording
    Recording(opt, start=True)
    # 保存参数和输出日志
    store_setup_txt(f'{opt.log_folder}/setup-train-val.txt', opt)  # 记录args设置

    # 记录正常的 print 信息
    sys.stdout = Logger(f'{opt.log_folder}/log.log')
    # 记录 traceback 异常信息
    sys.stderr = Logger(f'{opt.log_folder}/log.log')
    # build data loader based on noisy dataset
    train_loader, val_loader,cls_num_list = set_loader(opt)

    # 构建权重值
    # cls_num_list = dataset_train.num_per_cls_list
    if opt.train_rule == 'None':
        per_cls_weights = None
    elif opt.train_rule == 'CBReweight':
        train_sampler = None
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(0)
    elif opt.train_rule == 'IBReweight':
        per_cls_weights = 1.0 / np.array(cls_num_list)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(0)
    else:
        raise RuntimeError(f"Sample rule {opt.train_rule} is not listed ")

    # 构建损失函数
    criterion = torch.nn.CrossEntropyLoss(weight=None).cuda(0)
    if opt.loss_typea == 'CE':
        criterion = torch.nn.CrossEntropyLoss(weight=None).cuda(0)
        # criterion_ib = torch.nn.CrossEntropyLoss(weight=None).cuda(0)
    elif opt.loss_typea == 'LDAM':
        criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda(0)
        # criterion = torch.nn.CrossEntropyLoss(weight=None).cuda(0)
        # criterion_ib = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda(0)
    elif opt.loss_typea == 'Focal':
        criterion = FocalLoss(weight=per_cls_weights, gamma=1).cuda(0)
        # criterion = torch.nn.CrossEntropyLoss(weight=None).cuda(0)
        # criterion_ib = FocalLoss(weight=per_cls_weights, gamma=1).cuda(0)

    elif opt.loss_typea == 'IB':
        criterion = IBLoss(weight=per_cls_weights, alpha=1000).cuda(0)
        # criterion = torch.nn.CrossEntropyLoss(weight=None).cuda(0)
        # criterion_ib = IBLoss(weight=per_cls_weights, alpha=1000).cuda(0)
    elif opt.loss_typea == 'IBFocal':
        criterion = IB_FocalLoss(weight=per_cls_weights, num_classes=opt.n_cls, alpha=1000, gamma=1).cuda(0)
        # criterion = torch.nn.CrossEntropyLoss(weight=None).cuda(0)
        # criterion_ib = IB_FocalLoss(weight=per_cls_weights, alpha=1000, gamma=1).cuda(0)
    else:
        raise RuntimeError(f"Loss type {opt.loss_typea} is not listed ")

    criterion_ib = torch.nn.CrossEntropyLoss(weight=None).cuda(0)
    if opt.loss_typeb == 'CE':
        # criterion = torch.nn.CrossEntropyLoss(weight=None).cuda(0)
        criterion_ib = torch.nn.CrossEntropyLoss(weight=None).cuda(0)
    elif opt.loss_typeb == 'LDAM':
        # criterion = torch.nn.CrossEntropyLoss(weight=None).cuda(0)
        criterion_ib = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda(0)
    elif opt.loss_typeb == 'Focal':
        # criterion = torch.nn.CrossEntropyLoss(weight=None).cuda(0)
        criterion_ib = MultiFocalLoss(num_class=opt.n_cls, gamma=2.0, reduction='mean').cuda(0)

    elif opt.loss_typeb == 'IB':
        # criterion = torch.nn.CrossEntropyLoss(weight=None).cuda(0)
        criterion_ib = IBLoss(weight=per_cls_weights, num_classes=opt.n_cls, alpha=1000).cuda(0)
    elif opt.loss_typeb == 'IBFocal':
        # criterion = torch.nn.CrossEntropyLoss(weight=None).cuda(0)
        criterion_ib = IB_FocalLoss(weight=per_cls_weights, num_classes=opt.n_cls, alpha=1000, gamma=1).cuda(0)
    elif opt.loss_typeb == 'Poly':
        criterion_ib = PolyLoss().cuda(0)

    else:
        raise RuntimeError(f"Loss type {opt.loss_typeb} is not listed ")

    # build model and criterion
    model_temp, classifier= set_model(opt)
    model = FineTune4(opt.model, model_temp, opt.n_cls)  # the model that need to optimize
    model.to(opt.device)

    # build optimizer
    optimizer = set_optimizer_ce(opt, model)  # 优化整个model的params
    another_lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), opt.epochs, warmup=True, warmup_epochs=10)
    # tensorboard
    train_writer = SummaryWriter(log_dir=os.path.join(opt.tb_folder, 'train'))
    valid_writer = SummaryWriter(log_dir=os.path.join(opt.tb_folder, 'valid'))
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    train_loss_list, val_loss_list, train_acc_list, val_acc_list, epoch_list = [], [], [], [], []
    train_lr_list = []
    epoch_list = []
    val_f1_list, val_rec_list, val_pre_list = [], [], []
    # training routine
    for epoch in range(1, opt.epochs + 1):
        # lr = adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        train_loss, train_acc,lr = train(train_loader, model, criterion,  criterion_ib, optimizer, another_lr_scheduler, epoch, opt)
        train_acc = train_acc.data.cpu().item()
        print('Train epoch {}, loss:{:.2f}, accuracy:{:.2f}, lr:{:3f}'.format(epoch, train_loss,  train_acc, lr))

        # tensorboard logger
        train_writer.add_scalar('loss', train_loss, epoch)
        train_writer.add_scalar('acc', train_acc, epoch)

        # eval for one epoch on test data
        val_list, pred_list,val_loss, val_acc=validate(val_loader, model, criterion,criterion_ib,epoch, opt)
        val_acc = val_acc.data.cpu().item()
        print('Val epoch {}, loss:{:.2f}, accuracy:{:.2f}'.format(epoch, val_loss, val_acc))
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch =epoch
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

            cm = confusion_matrix(val_list, pred_list)

            label_cm = val_loader.dataset.class_names

            cf_matrix_path = opt.metric_folder

            plot_cm_hmap(cm, label_cm, cf_matrix_path, epoch, Normalize=False, show=True)

        valid_writer.add_scalar('loss', val_loss, epoch)
        valid_writer.add_scalar('acc', val_acc, epoch)

        F1Score_val = round(f1_score(y_true=val_list, y_pred=pred_list, average='weighted'), 4)  # 也可以指定micro模式
        AccScore_val = round(accuracy_score(y_true=val_list, y_pred=pred_list), 4)
        RecScore_val = round(recall_score(y_true=val_list, y_pred=pred_list, average='weighted'), 4)  # 也可以指定micro模式
        PreScore_val = round(precision_score(y_true=val_list, y_pred=pred_list, average='weighted'), 4)
        epoch_list.append(epoch)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        train_lr_list.append(lr)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        val_f1_list.append(F1Score_val)
        val_rec_list.append(RecScore_val)
        val_pre_list.append(PreScore_val)

        plt.figure(3, dpi=150)
        plt.plot(epoch_list, train_loss_list, 'r-', label=u'Train Loss')
        # 显示图例
        plt.plot(epoch_list, val_loss_list, 'b-', label=u'Val Loss')
        plt.legend(["Train Loss", "Val Loss"], loc="upper right")
        plt.xlabel(u'epoch')
        plt.ylabel(u'loss')
        plt.title('Model Loss ')
        plt.savefig(opt.log_folder + "/loss.png", bbox_inches='tight', pad_inches=0)
        plt.close(3)
        plt.figure(4, dpi=150)
        plt.plot(epoch_list, train_acc_list, 'r-', label=u'Train Acc')
        plt.plot(epoch_list, val_acc_list, 'b-', label=u'Val Acc')
        plt.legend(["Train Acc", "Val Acc"], loc="lower right")
        plt.title("Model Acc")
        plt.ylabel("acc")
        plt.xlabel("epoch")
        plt.savefig(opt.log_folder + "/acc.png", bbox_inches='tight', pad_inches=0)
        plt.close(4)

        plt.figure(5, dpi=150)
        plt.plot(epoch_list, train_lr_list, 'b-', label=u'Train LR')

        plt.legend(["Train LR"], loc="lower right")
        plt.title("Model LR")
        plt.ylabel("Learning Rate")
        plt.xlabel("epoch")
        plt.savefig(opt.log_folder + "/lr.png", bbox_inches='tight', pad_inches=0)
        plt.close(5)

        df_result = pd.DataFrame({'epoch': epoch_list, 'train_acc': train_acc_list,
                                  'train_loss': train_loss_list,
                                  'train_lr': train_lr_list,
                                  'val_loss': val_loss_list,
                                  'val_acc': val_acc_list,
                                  'val_Recall': val_rec_list,
                                  'val_Precision': val_pre_list,
                                  'val_F1': val_f1_list,
                                  'best_acc': best_acc})
        df_result.to_csv(opt.log_folder + '/df_result.csv')
    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

    print('best epoch: {} best accuracy: {:.4f}'.format(best_epoch, best_acc))
    # print('best accuracy: {:.2f}'.format(best_acc))
    # end recording
    Recording(opt, start=False)


if __name__ == '__main__':
    main()