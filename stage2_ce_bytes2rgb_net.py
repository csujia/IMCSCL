# 标准的supervised cross entropy训练

from __future__ import print_function
import datetime
import os
import sys

#导入模型
from my_models.alexnet import AlexNet
from my_models.densenet import densenet121
from my_models.googlenet import googlenet
from my_models.MobileNet import mobilenet
from my_models.ResNet import resnet18,resnet50
from my_models.SEResNet import seresnet18
from my_models.shufflenetv1 import shufflenetv1
from my_models.vgg import vgg19
from my_models.xception import xception

import argparse
import time
import math
import json
import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
import pandas as pd
import numpy as np
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from util import AverageMeter,Recording, save_model
from util import adjust_learning_rate, warmup_learning_rate, accuracy,create_lr_scheduler
from util import set_optimizer_ce, save_model
from util import store_setup_txt, Logger # 保存参数和输出日志
from networks.resnet_big import SupCEResNet
from networks.mobileone import SupCEResNet_mobile
from MyDataset import MyDataset
from MyLoss import LDAMLoss, FocalLoss, IBLoss, IB_FocalLoss
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
    parser = argparse.ArgumentParser('argument for training')

    # dataset
    parser.add_argument('--network_name', type=str, default='google', choices=['cifar10', 'cifar100', 'path'],
                        help='dataset')
    parser.add_argument('--dataset', type=str, default='path', choices=['cifar10', 'cifar100','path'], help='dataset')
    parser.add_argument('--data_folder', type=str,
                        default='/home/work/Jupyter_Notebook/Ve9eD0g/Img/Train/Mal_bytes2rgb/',
                        help='path to custom dataset')  # 如果数据不是cifar10/100
    parser.add_argument('--mean', type=str, default='(0.5829389, 0.8363681, 0.3923308)',
                        help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, default='(0.28230786, 0.20856646, 0.290898)',
                        help='std of dataset in path in form of str tuple')
    parser.add_argument('--mean2', type=str, default='(0.99018216, 0.9685766, 0.9477592)',
                        help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std2', type=str, default='(0.050820418, 0.044482365, 0.03263167)',
                        help='std of dataset in path in form of str tuple')
    parser.add_argument('--size', type=int, default=112,
                        help='parameter for RandomResizedCrop')  # 这里如果data image的size变了也要改动哦

    # train loader percent
    parser.add_argument('--percent', type=int, default=20, help='trainloader percent')

    # loss type and rule
    parser.add_argument('--train_rule', default='None', type=str,
                        help='data sampling strategy for train loader：CBReweight/IBReweight')
    # loss two stage
    parser.add_argument('--loss_typea', default="CE", type=str, help='loss type:CE,Focal,IB,IBFocal')
    parser.add_argument('--loss_typeb', default="CE", type=str, help='loss type')
    # loss分阶段
    parser.add_argument('--start_ib_epoch', default=61, type=int, help='start epoch for IB Loss')

    # training
    parser.add_argument('--model', type=str, default='convmixer')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')

    parser.add_argument('--epochs', type=int, default=60, help='number of training epochs')
    parser.add_argument('--warm_epoch', type=int, default=10, help='number of warm epochs')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='100,120', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true', help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true', help='warm-up for large batch training')
    parser.add_argument('--optimizer_type', type=str, default='sgd', choices=['sgd', 'adam', 'lars','adamw',],
                        help='Optimizer to use')
    # other setting
    parser.add_argument('--trial', type=str, default='0', help='id for recording multiple runs')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=100, help='save frequency')

    opt = parser.parse_args()

    # set the path according to the environment
    # opt.data_folder = 'G:\\0zip\\Mal_ASMNgram_MAT_RGB\\'


    # iterations = opt.lr_decay_epochs.split(',')
    # opt.lr_decay_epochs = list([])
    # for it in iterations:
    #     opt.lr_decay_epochs.append(int(it)) # 出来[300, 400, 500]这种list
    cur_time = datetime.datetime.now().strftime('%y%m%d-%H_%M_%S')
    opt.model_name = 'SupCE_{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size, opt.trial)   # 在每个里面又分SupCE / SimCLR

    # if opt.cosine:
    #     opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    # if opt.batch_size > 256:
    #     opt.warm = True
    # if opt.warm:
    #     opt.model_name = '{}_warm'.format(opt.model_name)
    #     opt.warmup_from = 0.01
    #     opt.warm_epochs = 10
    #     if opt.cosine:
    #         eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
    #         opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
    #                 1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
    #     else:
    #         opt.warmup_to = opt.learning_rate
    # 建立文件夹
    opt.model_name = './save/SupCE/{}_{}'.format(opt.model_name, cur_time)
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

    opt.log_folder = os.path.join(opt.model_name,opt.log_path)
    if not os.path.isdir(opt.log_folder):
        os.makedirs(opt.log_folder)

    opt.metric_folder = os.path.join(opt.model_name, opt.metric_path)
    if not os.path.isdir(opt.metric_folder):
        os.makedirs(opt.metric_folder)

    opt.n_cls = 9


    return opt


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
        train_dataset = MyDataset(opt.data_folder+'/train02', transform=train_transform)
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
    '''regular resnet'''
    model = SupCEResNet_mobile(name=opt.model, num_classes=opt.n_cls)
    #criterion = torch.nn.CrossEntropyLoss()


    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        #criterion = criterion.cuda()

        cudnn.benchmark = True

    return model


def train(train_loader, model, criterion, criterion_ib, optimizer,lr_scheduler, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
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

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        # if (idx + 1) % opt.print_freq == 0:
        #     print('Train: [{0}][{1}/{2}]\t'
        #           'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'loss {loss.val:.3f} ({loss.avg:.3f})\t'
        #           'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
        #            epoch, idx + 1, len(train_loader), batch_time=batch_time,
        #            data_time=data_time, loss=losses, top1=top1))
        #     sys.stdout.flush()
    # print(' *Train Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg, optimizer.param_groups[0]["lr"]


def validate(val_loader, model, criterion, criterion_ib, epoch, opt):
    """validation"""
    model.eval()
    val_list = []
    pred_list = []
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            for t in labels:
                val_list.append(t.data.item())
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
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
            _, yuce = torch.max(pred.data.cpu(), 1)
            for y in yuce:
                pred_list.append(y.data.item())

            # update metric
            losses.update(loss.item(), bsz)
            acc1,  = accuracy(pred, labels, topk=(1,))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            #
            # if idx % opt.print_freq == 0:
            #     print('Test: [{0}/{1}]\t'
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #           'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            #            idx, len(val_loader), batch_time=batch_time,
            #            loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return val_list, pred_list, losses.avg, top1.avg


def main():
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    best_acc = 0
    best_epoch =0
    opt = parse_option()

    # 保存参数和输出日志
    store_setup_txt(f'{opt.log_folder}/setup-train-val.txt', opt)  # 记录args设置

    # 记录正常的 print 信息
    sys.stdout = Logger(f'{opt.log_folder}/log.log')
    # 记录 traceback 异常信息
    sys.stderr = Logger(f'{opt.log_folder}/log.log')
    # 创建保存数据路径
    # start recording
    # Recording(opt, start=True)
    # build data loader
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
        criterion_ib = FocalLoss(weight=per_cls_weights, num_classes=opt.n_cls, gamma=1).cuda(0)

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
        # build model and criterion
    if opt.network_name == 'alex':
        model = AlexNet(num_classes=opt.classes)
    elif opt.network_name == 'defense':
        model = densenet121(num_classes=opt.classes)
    elif opt.network_name == 'goole':
        model = googlenet(num_class=opt.classes)
    elif opt.network_name == 'mobile':
        model = mobilenet(num_classes=opt.classes)
    elif opt.network_name == 'resnet18':
        model = resnet18(num_classes=opt.classes)
    elif opt.network_name == 'resnet50':
        model = resnet50(num_classes=opt.classes)
    elif opt.network_name == 'shuffle':
        model = shufflenetv1(num_classes=opt.classes)
    elif opt.network_name == 'senet':
        model = seresnet18(num_classes=opt.classes)
    elif opt.network_name == 'vgg':
        model = vgg19(num_classes=opt.classes)
    elif opt.network_name == 'xception':
        model = xception(num_classes=opt.classes)
    elif opt.network_name == 'convmixer':
        model = set_model(opt)
        # model = set_model(opt)
        # model = set_model(opt)

    # build optimizer
    optimizer = set_optimizer_ce(opt, model)
    another_lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), opt.epochs, warmup=True, warmup_epochs=10)
    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    train_loss_list, val_loss_list, train_acc_list, val_acc_list, epoch_list = [], [], [], [], []
    train_lr_list = []
    epoch_list =[]
    val_f1_list, val_rec_list, val_pre_list = [], [], []
    # training routine
    for epoch in range(1, opt.epochs + 1):
        # adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        train_loss, train_acc,lr = train(train_loader, model, criterion, criterion_ib, optimizer, another_lr_scheduler, epoch, opt)
        time2 = time.time()
        train_acc = train_acc.data.cpu().item()
        print('Train epoch {}, loss:{:.2f}, accuracy:{:.2f}, lr:{:3f}'.format(epoch, train_loss, train_acc, lr))
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))


        # tensorboard logger
        logger.log_value('train_loss', train_loss, epoch)
        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('learning_rate', lr, epoch)

        # evaluation
        val_list, pred_list,val_loss, val_acc = validate(val_loader, model, criterion,criterion_ib,epoch, opt)
        val_acc = val_acc.data.cpu().item()
        logger.log_value('val_loss', val_loss, epoch)
        logger.log_value('val_acc', val_acc, epoch)

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


        # if epoch % opt.save_freq == 0:
        #     save_file = os.path.join(
        #         opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
        #     save_model(model, optimizer, opt, epoch, save_file)

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

    # print('best accuracy: {:.2f}'.format(best_acc))
    print('best epoch: {} best accuracy: {:.4f}'.format(best_epoch, best_acc))

    # end recording
    # Recording(opt, start=False)
if __name__ == '__main__':
    main()
