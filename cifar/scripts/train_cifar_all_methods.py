# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py
import sys
import time 
import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from models import resnet as RN
from  models import pyramidnet as PYRM
from  models import vgg as VGG
from models import wideresnet as WR 
from models import shufflenetv2 as SN
from models import mobilenetv2 as MN
from models import resnext as RNX
from models import densenet as DN
from models.iccv19_resnet import *
#from models.iccv19_resnet_ds import *
from models.preactresnet import CIFAR_ResNet18, CIFAR_ResNet34 
from tensorboardX import SummaryWriter
from loss_all_methods import SCELoss, label_smooth, generalized_cross_entropy, joint_optimization, boot_soft, boot_hard, Forward, Backward, DisturbLabel, PC
import random
import utils
import numpy as np
import cv2
import warnings
import random
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True

warnings.filterwarnings("ignore")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
                     

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10, CIFAR-100 and ImageNet-1k Training')
parser.add_argument('--net_type', default='pyramidnet', type=str,
                    help='networktype: resnet, and pyamidnet')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='start epoch to run')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1000, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--depth', default=32, type=int,
                    help='depth of the network (default: 32)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='to use basicblock for CIFAR datasets (default: bottleneck)')
parser.add_argument('--dataset', dest='dataset', default='imagenet', type=str,
                    help='dataset (options: cifar10, cifar100, and imagenet)')
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help='to print the status at every iteration')
#parser.add_argument('--alpha', default=300, type=float,
#                    help='number of new channel increases per depth (default: 300)')
parser.add_argument('--expname', default='TEST', type=str,
                    help='name of experiment')
parser.add_argument('--save_dir', default='./', type=str,
                    help='model saved dir')
parser.add_argument('--seed', type=int, default=2019, help='random seed')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default='./runs/pretrained_model/resnet50-19c8e357.pth', type=str, metavar='PATH',
					help='path to resnet50 pretrained pth.')
parser.add_argument('--dali_cpu', action='store_true',
                    help='Runs CPU based version of DALI pipeline.')
parser.add_argument('--width', default=None, type=int, help='the widen factor of wideresnet')

parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--phase", default=None, type=int)

parser.add_argument('--theta', default=0.5, type=float)
parser.add_argument('--method', default=None, type=str, help='ce, sce, ls, gce, jo, bootsoft, boothard, forward, backward, disturb')
parser.add_argument('--olsalpha', default=None, type=float)
parser.add_argument('--T', default=1.0, type=float, help='temprature to scale')

parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)

best_err1 = 100
best_err5 = 100
numberofclass = 1000

def main():
    global args, best_err1, best_err5, numberofclass 
    args = parser.parse_args()
    
    assert args.method in ['ce', 'ols', 'sce', 'ls', 'gce', 'jo', 'bootsoft', 'boothard', 'forward', 'backward', 'disturb', 'PC'], \
        "method must be the one of 'ce', 'sce', 'ls', 'gce', 'jo', 'bootsoft', 'boothard', 'forward', 'backward', 'disturb', 'PC'  "
    
    args.gpu = 0
    args.world_size = 1
    
    print(args)
    log_dir = '%s/runs/record_dir/%s/' % (args.save_dir, args.expname)
    writer = SummaryWriter(log_dir=log_dir)
    
    if args.seed is not None:
        print('set the same seed for all.....')
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)


    if args.dataset.startswith('cifar'):
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        if args.dataset == 'cifar100':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('./data', train=True, download=True, transform=transform_train),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('./data', train=False, transform=transform_test),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
            numberofclass = 100
        elif args.dataset == 'cifar10':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('./data', train=True, download=True, transform=transform_train),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('./data', train=False, transform=transform_test),
                batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)
            numberofclass = 10
        else:
            raise Exception('unknown dataset: {}'.format(args.dataset))
        
    elif args.dataset == 'imagenet':
        traindir = os.path.join('./data/ILSVRC1/train')
        valdir = os.path.join('./data/ILSVRC1/val1')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        jittering = utils.ColorJitter(brightness=0.4, contrast=0.4,
                                      saturation=0.4)
        lighting = utils.Lighting(alphastd=0.1,
                                  eigval=[0.2175, 0.0188, 0.0045],
                                  eigvec=[[-0.5675, 0.7192, 0.4009],
                                          [-0.5808, -0.0045, -0.8140],
                                          [-0.5836, -0.6948, 0.4203]])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                jittering,
                lighting,
                normalize,
            ]))

        train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=False, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False)
        numberofclass = 1000

    
    
    print("=> creating model '{}'".format(args.net_type))
    # define loss function (criterion) and optimizer
    solver = Solver()

    solver.model = solver.model.cuda()
    print('the number of model parameters: {}'.format(sum([p.data.nelement() for p in solver.model.parameters()])))
    cudnn.benchmark = True

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_err1 = checkpoint['best_err1']
            solver.model.load_state_dict(checkpoint['state_dict'])
            solver.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                 .format(args.resume, checkpoint['epoch']))
                 
     
    for epoch in range(args.start_epoch, args.epochs):  
        print('current os time = ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        adjust_learning_rate(solver.optimizer, epoch)
        # train for one epoch
        train_loss = solver.train(train_loader, epoch)
        # evaluate on validation set
        err1, err5, val_loss = solver.validate(val_loader, epoch)

        writer.add_scalar('training loss', train_loss, epoch)
        writer.add_scalar('testing loss', val_loss, epoch)
        writer.add_scalar('top1 error', err1, epoch)
        writer.add_scalar('top5 error', err5, epoch)

        # remember best prec@1 and save checkpoint
        is_best = err1 <= best_err1
        best_err1 = min(err1, best_err1)
        if is_best:
            best_err5 = err5
        
        print('Current best accuracy (top-1 and 5 error):', best_err1, best_err5)
        save_checkpoint({
            'epoch': epoch,
            'arch': args.net_type,
            'state_dict': solver.model.state_dict(),
            'best_err1': best_err1,
            'best_err5': best_err5,
            'optimizer': solver.optimizer.state_dict(),
        }, is_best)

    print('Best accuracy (top-1 and 5 error):', best_err1, best_err5)
    print('method = {}, expname = {}'.format(args.method, args.expname))
    loss_dir = "%s/runs/record_dir/%s/" % (args.save_dir, args.expname)
    writer.export_scalars_to_json(loss_dir + 'loss.json')
    writer.close()

class Solver():
    def __init__(self):
        super(Solver, self).__init__()
        global numberofclass 
        
        #define the network
        if args.net_type == 'resnet':
            self.model = RN.ResNet(dataset=args.dataset, depth=args.depth, num_classes=numberofclass, bottleneck=args.bottleneck)
            
        elif args.net_type == 'pyramidnet':
            self.model = PYRM.PyramidNet(args.dataset, args.depth, args.alpha, numberofclass,
                                    args.bottleneck)
                                    
        elif args.net_type == 'wideresnet':
            self.model = WR.WideResNet(depth=args.depth, num_classes=numberofclass, widen_factor=args.width)
            
        elif args.net_type == 'vggnet':
            self.model = VGG.vgg16(num_classes=numberofclass)
            
        elif args.net_type == 'mobilenet':
            self.model = MN.mobile_half(num_classes=numberofclass)
            
        elif args.net_type == 'shufflenet':
            self.model = SN.ShuffleV2(num_classes=numberofclass)
        
        elif args.net_type == 'densenet':
            self.model = DN.densenet_cifar(num_classes=numberofclass)
        
        elif args.net_type == 'resnext29-2':
            self.model = RNX.ResNeXt29_2x64d(num_classes=numberofclass)
        elif args.net_type == 'resnext29-4':
            self.model = RNX.ResNeXt29_4x64d(num_classes=numberofclass)
        elif args.net_type == 'resnext29-32':
            self.model = RNX.ResNeXt29_32x4d(num_classes=numberofclass)
        
        elif args.net_type == 'imagenetresnet18':
            self.model = multi_resnet18_kd(num_classes=numberofclass)
        elif args.net_type == 'imagenetresnet34':
            self.model = multi_resnet34_kd(num_classes=numberofclass)
        elif args.net_type == 'imagenetresnet50':
            self.model = multi_resnet50_kd(num_classes=numberofclass)
        elif args.net_type == 'imagenetresnet101':
            self.model = multi_resnet101_kd(num_classes=numberofclass)
        elif args.net_type == 'imagenetresnet152':
            self.model = multi_resnet152_kd(num_classes=numberofclass)
        else:
            raise Exception('unknown network architecture: {}'.format(args.net_type))

        

        
        self.optimizer = torch.optim.SGD(self.model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)
        self.loss_lams = torch.zeros(numberofclass, numberofclass, dtype=torch.float32).cuda()
        self.loss_lams.requires_grad = False 
        #define the loss function
        if args.method == 'ce':
            self.criterion = nn.CrossEntropyLoss()
        elif args.method == 'sce':
            if args.dataset == 'cifar10':
                self.criterion = SCELoss(alpha=0.1, beta=1.0, num_classes=numberofclass)
            else:
                self.criterion = SCELoss(alpha=6.0, beta=0.1, num_classes=numberofclass)
        elif args.method == 'ls':
            self.criterion = label_smooth(num_classes=numberofclass)
        elif args.method == 'gce':
            self.criterion = generalized_cross_entropy(num_classes=numberofclass)
        elif args.method == 'jo':
            self.criterion = joint_optimization(num_classes=numberofclass)
        elif args.method == 'bootsoft':
            self.criterion = boot_soft(num_classes=numberofclass)
        elif args.method == 'boothard':
            self.criterion = boot_hard(num_classes=numberofclass)
        elif args.method == 'forward':
            self.criterion = Forward(num_classes=numberofclass)
        elif args.method == 'backward':
            self.criterion = Backward(num_classes=numberofclass)
        elif args.method == 'disturb':
            self.criterion = DisturbLabel(num_classes=numberofclass)
        elif args.method == 'ols':
            self.criterion = nn.CrossEntropyLoss()
        elif args.method == 'PC':
            self.criterion = PC(100)
        self.criterion = self.criterion.cuda()
        

    def update_loss_lams(self, output, target):
        with torch.no_grad():
            logits = torch.softmax(output, dim=1)
            sort_args = torch.argsort(logits, dim=1, descending=True)
            for k in range(output.shape[0]):
                if target[k] != sort_args[k, 0]:
                    continue
                self.cur_epoch_lams[target[k]] += logits[k]
                self.cur_epoch_cnt[target[k]] += 1
    def update_loss_lams2(self, output, target):
        with torch.no_grad():
            logits = output / args.T
            sort_args = torch.argsort(logits, dim=1, descending=True)
            for k in range(output.shape[0]):
                if target[k] != sort_args[k, 0]:
                    continue
                self.cur_epoch_lams[target[k]] += logits[k]
                self.cur_epoch_cnt[target[k]] += 1
    
    
    def soft_cross_entropy(self, output, target):    
        target_prob = torch.zeros_like(output)
        batch = output.shape[0]
        for k in range(batch):
            target_prob[k] = self.loss_lams[target[k]]
        log_like = -torch.nn.functional.log_softmax(output, dim=1)
        loss = torch.sum(torch.mul(log_like, target_prob)) / batch 
        return loss
    def kd_loss(self, output, target):
        target_prob = torch.zeros_like(output)
        batch = output.shape[0]
        for k in range(batch):
            target_prob[k] = self.loss_lams[target[k]]
        ps = torch.nn.functional.log_softmax(output / args.T, dim=1)
        log_like = torch.nn.functional.kl_div(ps, target_prob, size_average=False)
        log_like = log_like * (args.T ** 2) / batch
        return log_like
    def sce_loss(self, output, onehot):
        log_like = -torch.nn.functional.log_softmax(output, dim=1)
        loss = torch.sum(torch.mul(log_like, onehot)) / output.shape[0]
        return loss
    
    def train(self, train_loader, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        global numberofclass 

        # switch to train mode
        self.model.train()
        end = time.time()
        current_LR = get_learning_rate(self.optimizer)[0]
        
        self.cur_epoch_lams = torch.zeros(numberofclass, numberofclass, dtype=torch.float32).cuda()
        self.cur_epoch_cnt = torch.zeros(numberofclass, dtype=torch.float32).cuda()
        self.cur_epoch_lams.requires_grad = False
        self.cur_epoch_cnt.requires_grad = False
        for i, (input, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.cuda()
            target = target.cuda()
            input_var = torch.autograd.Variable(input, requires_grad=True)
            
            
            # compute output
            output = self.model(input_var)
            
            if args.method == 'ols':
                self.update_loss_lams(output, target)
                args.T = 1.
                loss = 0.5 * self.criterion(output, target) + \
                          0.5 * self.soft_cross_entropy(output, target)
            elif args.method == 'ls':
                loss = self.criterion(output, target)
            elif args.method == 'PC':
                if epoch < 200:
                    loss = torch.nn.functional.cross_entropy(output, \
                            target)
                else:
                    loss = self.criterion(output, target)
            else:
                loss = self.criterion(output, target)
                
            #measure accuracy and record loss
            err1, err5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(err1.item(), input.size(0))
            top5.update(err5.item(), input.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
                         
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
    
                    
            if i % args.print_freq == 0 and args.verbose == True:
                print('Epoch: [{0}/{1}][{2}/{3}]\t'
                      'LR: {LR:.6f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                      'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                    epoch, args.epochs, i, len(train_loader), LR=current_LR, batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
            
        if args.method == 'ols':   
            for cls in range(numberofclass):
                if self.cur_epoch_cnt[cls].max() < 0.5:
                    self.loss_lams[cls] = 1. / numberofclass 
                else:
                    self.loss_lams[cls] = self.cur_epoch_lams[cls] / self.cur_epoch_cnt[cls]            
        return losses.avg

    def validate(self, val_loader, epoch):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        self.model.eval()
        end = time.time()

        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input = input.cuda()
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            with torch.no_grad():
                output = self.model(input_var)
            loss = self.criterion(output, target_var)
            # measure accuracy and record loss
            err1, err5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))

            top1.update(err1.item(), input.size(0))
            top5.update(err5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and args.verbose == True:
                print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                      'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                       epoch, args.epochs, i, 196, batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'.format(
            epoch, args.epochs, top1=top1, top5=top5, loss=losses))
        return top1.avg, top5.avg, losses.avg
    def save_scripts(self, val_loader, epoch, prop):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        self.model.eval()
        end = time.time()

        save_name = args.net_type + '_' + args.method + '_' + prop 
        embedding = []
        labels = []

        for i, (input, target) in enumerate(val_loader):
            if i > 80:
                break
            target = target.cuda()
            input = input.cuda()
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            with torch.no_grad():
                output, ebd = self.model(input_var)
                for j in range(ebd.shape[0]):
                    embedding.append(ebd[j].detach().cpu().numpy())
                    labels.append(target[j].detach().cpu().numpy())
            
            loss = self.criterion(output, target_var)
            # measure accuracy and record loss
            err1, err5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))

            top1.update(err1.item(), input.size(0))
            top5.update(err5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and args.verbose == True:
                print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                      'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                       epoch, args.epochs, i, 196, batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
        embedding = np.array(embedding)
        labels = np.array(labels)
        print('------',embedding.shape)
        for kk in range(10):
            print(embedding[kk].shape)
        np.save('./embeddings/' + save_name + '_' + 'emd.npy', embedding)
        np.save('./embeddings/' + save_name + '_' + 'labels.npy', labels)
 
        print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'.format(
            epoch, args.epochs, top1=top1, top5=top5, loss=losses))
        return top1.avg, top5.avg, losses.avg
    
    
    def visualize(self, dataloader, classes_id = [5, 2, 1], samples_per_class = 100):
        data = torch.zeros(samples_per_class * len(classes_id), 256, dtype=torch.float32)
        target = torch.zeros(samples_per_class * len(classes_id), dtype=torch.float32)
        cnt = torch.ones(len(classes_id), dtype=torch.long) * samples_per_class
        cur_cnt = 0
        with torch.no_grad():
            for i, (input, y) in enumerate(dataloader):
                output, attens = self.model(input)
                if cnt.sum() == 0:
                    break
                for kk in range(input.shape[0]):
                    if int(y[kk]) in classes_id and cnt[classes_id.index(y[kk])] > 0:
                        data[cur_cnt] = attens[kk]
                        target[cur_cnt] = y[kk]
                     
                        cnt[classes_id.index(y[kk])] = cnt[classes_id.index(y[kk])] - 1
                        cur_cnt += 1
                 
        return data, target

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = "%s/runs/record_dir/%s/" % (args.save_dir, args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '%s/runs/record_dir/%s/' % (args.save_dir, args.expname) + 'model_best.pth.tar')

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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.dataset.startswith('cifar') or args.dataset == ('tiny-imagenet'):
        lr = args.lr * (0.1 ** (epoch // (args.epochs * 0.5))) * (0.1 ** (epoch // (args.epochs * 0.75)))
    elif args.dataset == ('imagenet'):
        if args.epochs == 300:
            lr = args.lr * (0.1 ** (epoch // 75))
        elif args.epochs > 30:
            lr = args.lr * (0.1 ** (epoch // 30))
        #else:
        #    if epoch < 10:
        #        lr = args.lr
        #    elif epoch < 40:
        #        lr = args.lr * 0.1
        #    else:
        #        lr = args.lr * 0.01
            #lr = args.lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res


if __name__ == '__main__':
    main()
