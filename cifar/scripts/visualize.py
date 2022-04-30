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
from models import resnet_visualize as RN
from  models import pyramidnet as PYRM
from  models import vgg as VGG
from models import wideresnet as WR 
from models import resnet_distillation as RD
from tensorboardX import SummaryWriter
#from scripts.dali_dataloader_imagenet import *
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
                     

parser = argparse.ArgumentParser(description='Cutmix PyTorch CIFAR-10, CIFAR-100 and ImageNet-1k Training')
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

parser.add_argument("--local_rank", default=0, type=int)

parser.add_argument('--theta', default=0.5, type=float)

parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)

best_err1 = 100
best_err5 = 100
numberofclass = 1000

def main():
    global args, best_err1, best_err5, numberofclass 
    args = parser.parse_args()
    
    args.gpu = 0
    args.world_size = 1
    
    print(args)
    log_dir = '%s/runs/%s/' % (args.save_dir, args.expname)
    writer = SummaryWriter(log_dir=log_dir)
    
    if args.seed is not None:
        print('set the same seed for all.....')
        random.seed(args.seed)
        torch.manual_seed(args.seed)

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
                batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)
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
        
    #elif args.dataset == 'imagenet':
    #    traindir = os.path.join('./data/ILSVRC1/train')
    #    valdir = os.path.join('./data/ILSVRC1/val1')
        
        #pipe = HybridTrainPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=0, data_dir=traindir, crop=224, dali_cpu=False)#dali_cpu=args.dali_cpu)
        #pipe.build()
        #train_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))

        #pipe = HybridValPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=0, data_dir=valdir, crop=224, size=256)
        #pipe.build()
        #val_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))
        #numberofclass = 1000
        
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

    solver.model = torch.nn.DataParallel(solver.model).cuda()
    print(solver.model)
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
    
    ###visualize begin
    import numpy as np


    x = np.load('./data/cifar-10h/data/cifar10h-probs.npy')
    mcnt = torch.zeros(10, dtype=torch.long)
    mlams = torch.zeros(10, 10, dtype=torch.float32)
    for i in range(x.shape[0]):
        target = np.argmax(x[i])
        mlams[target] += torch.tensor(x[i], dtype=torch.float32)
        mcnt[target] += 1
    for i in range(10):
        mlams[i]  = mlams[i] / 1.0 / mcnt[i]
    
    #ditribution-1 mlams
    #distribution-2 solver.loss_lams
    #are there using MAE or KL divergence or soft_cross_entropy?
    
    ######################################load baseline model
    solver.model.load_state_dict(torch.load('./runs/resnet32_cifar10_baseline/model_best.pth.tar')['state_dict'])
    print('baseline model, single sample kl loss = ', solver.kl_loss_dataset(val_loader))
  
    data, target = solver.visualize(val_loader)
    writer.add_embedding(data, metadata=target, tag='val_baseline')
    data, target = solver.visualize(train_loader)
    writer.add_embedding(data, metadata=target, tag='train_baseline')
    
    adjust_learning_rate(solver.optimizer, 299)
    solver.train(train_loader, 299)
    with open(log_dir+'/baseline_lams_trainset.txt', 'w') as f:
        for kk in range(numberofclass):
            tmp = ''
            for pp in range(numberofclass):
                xx = float(solver.loss_lams[kk, pp])
                tmp = tmp + ' ' + str(xx)
            f.writelines(tmp + '\n')
            
    solver.train(val_loader, 299)
    with open(log_dir+'/baseline_lams_valset.txt', 'w') as f:
        for kk in range(numberofclass):
            tmp = ''
            for pp in range(numberofclass):
                xx = float(solver.loss_lams[kk, pp])
                tmp = tmp + ' ' + str(xx)
            f.writelines(tmp + '\n')
    
    log_like = -torch.log(solver.loss_lams.cpu())
    target_prob = mlams
    loss = torch.sum(torch.mul(log_like, target_prob)) / 10
    print('baseline lams -> human prob soft cross entropy = {}'.format(loss)) 
    
    kl_loss = mlams * (torch.log(mlams) - torch.log(solver.loss_lams.cpu()))
    kl_loss = kl_loss.sum() / 10.
    print('baseline lams -> human prob KL loss = {}'.format(kl_loss)) 
    
    ################################
    ######################################load ours model
    solver.model.load_state_dict(torch.load('./runs/resnet32_cifar10_ours_method/model_best.pth.tar')['state_dict'])
    print('ours model, single sample kl loss = ', solver.kl_loss_dataset(val_loader))

    data, target = solver.visualize(val_loader)
    writer.add_embedding(data, metadata=target, tag='val_ours')
    data, target = solver.visualize(train_loader)
    writer.add_embedding(data, metadata=target, tag='train_ours')
    
    adjust_learning_rate(solver.optimizer, 299)
    solver.train(train_loader, 299)
    with open(log_dir+'/ours_lams_trainset.txt', 'w') as f:
        for kk in range(numberofclass):
            tmp = ''
            for pp in range(numberofclass):
                xx = float(solver.loss_lams[kk, pp])
                tmp = tmp + ' ' + str(xx)
            f.writelines(tmp + '\n')
            
    solver.train(val_loader, 299)
    with open(log_dir+'/ours_lams_valset.txt', 'w') as f:
        for kk in range(numberofclass):
            tmp = ''
            for pp in range(numberofclass):
                xx = float(solver.loss_lams[kk, pp])
                tmp = tmp + ' ' + str(xx)
            f.writelines(tmp + '\n')
    
    log_like = -torch.log(solver.loss_lams.cpu())
    target_prob = mlams
    loss = torch.sum(torch.mul(log_like, target_prob)) / 10
    print('ours lams -> human prob soft cross entropy = {}'.format(loss)) 
    
    kl_loss = mlams * (torch.log(mlams) - torch.log(solver.loss_lams.cpu()))
    kl_loss = kl_loss.sum() / 10.
    print('ours lams -> human prob KL loss = {}'.format(kl_loss)) 
    ########################################
    ########################################    
    return

    ###visualize end
    
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
    loss_dir = "%s/runs/%s/" % (args.save_dir, args.expname)
    writer.export_scalars_to_json(loss_dir + 'loss.json')
    writer.close()

class Solver():
    def __init__(self):
        super(Solver, self).__init__()
        global numberofclass 
        if args.net_type == 'resnet':
            ##model = RN.ResNet(args.dataset, args.depth, numberofclass, args.bottleneck)  # for ResNet
            ##model = RN.resnet50(pretrained=False, num_classes=100)  # for ResNet
            ##model = RN.wide_resnet50_2(pretrained=False, num_classes=100)  # for ResNet
            self.model = RN.ResNet(dataset=args.dataset, depth=args.depth, num_classes=numberofclass, bottleneck=args.bottleneck)
            #self.model = RD.resnet56(num_classes=100) 
        elif args.net_type == 'pyramidnet':
            self.model = PYRM.PyramidNet(args.dataset, args.depth, args.alpha, numberofclass,
                                    args.bottleneck)
        elif args.net_type == 'wideresnet':
            self.model = WR.WideResNet(depth=40, num_classes=numberofclass, widen_factor=4)
            #model = WR.WideResNet(depth=28, num_classes=100, widen_factor=10) 
        elif args.net_type == 'vggnet':
            self.model = VGG.vgg16_bn(pretrained=False, dataset=args.dataset, num_classes=numberofclass)
        else:
            raise Exception('unknown network architecture: {}'.format(args.net_type))

        self.criterion = nn.CrossEntropyLoss().cuda()

        
        self.optimizer = torch.optim.SGD(self.model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)

        self.loss_lams = torch.zeros(numberofclass, numberofclass, dtype=torch.float32).cuda()

    def update_loss_lams(self, output, target):
        with torch.no_grad():
            logits = torch.softmax(output, dim=1)
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

        for i, (input, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.cuda()
            target = target.cuda()
            input_var = torch.autograd.Variable(input, requires_grad=True)
            with torch.no_grad():
                # compute output
                if True:
                    output, _ = self.model(input_var)
                    self.update_loss_lams(output, target)
                    #print(self.soft_cross_entropy(output, target), self.criterion(output, target))
                    loss = torch.tensor(0.)               
                else:      
                    output = self.model(input)
                    loss = torch.tensor(0.)
                
            #measure accuracy and record loss
            err1, err5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(err1.item(), input.size(0))
            top5.update(err5.item(), input.size(0))

                         
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
    
    def visualize(self, dataloader, classes_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], samples_per_class = 1000):
        data = torch.zeros(samples_per_class * len(classes_id), 256, dtype=torch.float32)
        target = torch.zeros(samples_per_class * len(classes_id), dtype=torch.float32)
        cnt = torch.ones(len(classes_id), dtype=torch.long) * samples_per_class
        cur_cnt = 0
        with torch.no_grad():
            for i, (input, y) in enumerate(dataloader):
                input = input.cuda()
                y = y.cuda()
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
        
    def kl_loss(self, pred_output, gt_output):
        pred = torch.nn.functional.log_softmax(pred_output, dim=1)
        gt = gt_output
        loss = torch.nn.functional.kl_div(pred, gt, size_average=False) / pred_output.shape[0]
        return loss
        
    def kl_loss_dataset(self, dataloader):
        probs = np.load('./data/cifar-10h/data/cifar10h-probs.npy')
        avg_loss = AverageMeter()
        
        cnt = 0
        with torch.no_grad():
            for i, (input, y) in enumerate(dataloader):
                input = input.cuda()
                y = y.cuda()
                output, attens = self.model(input)
                
                ####all samples
                #pred_output = output 
                #gt_output = torch.FloatTensor(probs[cnt : cnt + output.shape[0]]).cuda()
                #tmp_loss = self.kl_loss(pred_output, gt_output)
                #avg_loss.update(tmp_loss.item())
                #cnt += output.shape[0]
                ####all samples
                
                ####only correct samples
                pred_output, gt_output = [], []
                for kk in range(output.shape[0]):
                    if torch.argmax(output[kk]) == y[kk]:
                        pred_output.append(output[kk].cpu().numpy())
                        gt_output.append(probs[cnt + kk])
                pred_output = np.array(pred_output, dtype=np.float32)
                gt_output = np.array(gt_output, dtype=np.float32)
                
                pred_output = torch.FloatTensor(pred_output).cuda()
                gt_output = torch.FloatTensor(gt_output).cuda()
                tmp_loss = self.kl_loss(pred_output, gt_output)
                avg_loss.update(tmp_loss.item())
                cnt += output.shape[0]
                ####only correct samples
                
        return avg_loss.avg
                

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = "%s/runs/%s/" % (args.save_dir, args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '%s/runs/%s/' % (args.save_dir, args.expname) + 'model_best.pth.tar')

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
