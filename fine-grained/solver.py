import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
from data.dataset import DataSet
from utils import *
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
import time
import os
from network import resnet, vgg_cub, vgg_std, res2net, mobilenetv2
import random
import torch.nn.functional as F

from tensorboardX import SummaryWriter

def tf_kd_loss(outputs, labels):
    T = 40 # mobilenetv2
    multipler = 1.0 # mobilenetv2
    alpha = 0.95    # mobilenetv2

    # T = 6 #resnet50
    # multipler = 1.0 #resnet50
    # alpha = 0.95 #resnet50

    correct_prob = 0.99
    loss_CE = F.cross_entropy(outputs, labels)
    K = outputs.size(1)
    teacher_soft = torch.ones_like(outputs).cuda()
    teacher_soft = teacher_soft * (1 - correct_prob) / (K-1)
    for i in range(outputs.shape[0]):
        teacher_soft[i, labels[i]] = correct_prob
    loss_soft_regu = nn.KLDivLoss()(F.log_softmax(outputs, dim=1), F.softmax(teacher_soft/T, dim=1)) * multipler
    KD_loss = (1. - alpha) * loss_CE + alpha * loss_soft_regu
    return KD_loss
    

                
class label_smooth(torch.nn.Module):
    def __init__(self, num_classes):
        super(label_smooth, self).__init__()
        self.num_classes = num_classes
        self.softmax = torch.nn.Softmax(dim=1)
        self.alpha = 0.1
        self.device = 'cuda'
    def forward(self, output, target):
        probs = self.softmax(output)
        label_one_hot = torch.nn.functional.one_hot(target, self.num_classes).float().to(self.device)
        label_one_hot = label_one_hot * (1. - self.alpha) + self.alpha / float(self.num_classes)
        loss = torch.sum(-label_one_hot * torch.log(probs), dim=1).mean()
        return loss


class Solver():
    def __init__(self, args, train_transform=None, test_transform=None, val_transform=None):
        self.args = args
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.val_transform = val_transform
        
        self.loss_lams = torch.zeros(args.num_classes, args.num_classes, dtype=torch.float32).cuda()
        self.loss_lams[:, :] = 1. / args.num_classes
        self.loss_lams.requires_grad = False

        if self.args.arch == 'resnet50':
            self.net = resnet.resnet50(num_classes=args.num_classes)
        elif self.args.arch == 'res2net50':
            self.net = res2net.res2net50_26w_4s(num_classes=args.num_classes)
        elif self.args.arch == 'resnet101':
            self.net = resnet.resnet101()
        elif self.args.arch == 'resnet18':
            self.net = resnet.resnet18()
        elif self.args.arch == 'resnet34':
            self.net = resnet.resnet34()
        elif self.args.arch == 'vgg16':
            self.net = vgg_cub.vgg16()
        elif self.args.arch == 'vgg16_bn':
            self.net = vgg_cub.vgg16_bn()
        elif self.args.arch == 'vgg19':
            self.net = vgg_cub.vgg19()
        elif self.args.arch == 'vgg19_bn':
            self.net = vgg_cub.vgg19_bn()
        elif self.args.arch == 'vgg16_std':
            self.net = vgg_std.vgg16()
        elif self.args.arch == 'vgg16_bn_std':
            self.net = vgg_std.vgg16_bn()
        elif self.args.arch == 'mobilenetv2':
            self.net = mobilenetv2.mobilenet_v2(num_classes=args.num_classes)

        if self.args.load_model is not None:
            self.net.load_state_dict(torch.load(self.args.load_model), strict=True)
            print('load model from %s' % self.args.load_model)
        elif self.args.pretrained_model is not None:
            self.net.load_state_dict(torch.load(self.args.pretrained_model), strict=False)
            print('load pretrained model form %s' % self.args.pretrained_model)
        else:
            print('not load any model, will train from scrach!')

        if args.expname is None:
            args.expname = 'runs/{}_{}_{}'.format(args.arch, args.dataset, args.method)
        os.makedirs(args.expname, exist_ok=True)
    
    def soft_cross_entropy(self, output, target):    
        target_prob = torch.zeros_like(output)
        batch = output.shape[0]
        for k in range(batch):
            target_prob[k] = self.loss_lams[target[k]]
        log_like = -torch.nn.functional.log_softmax(output, dim=1)
        loss = torch.sum(torch.mul(log_like, target_prob)) / batch 
        return loss

    def PairwiseConfusion(self, features):
        batch_size = features.size(0)
        if float(batch_size) % 2 != 0:
            raise Exception('Incorrect batch size provided')
        batch_left = features[:int(0.5*batch_size)]
        batch_right = features[int(0.5*batch_size):]
        loss  = torch.norm((batch_left - batch_right).abs(),2, 1).sum() / float(batch_size)
        return loss
    
    def update_loss_lams(self, output, target):
        with torch.no_grad():
            logits = torch.softmax(output, dim=1)
            sort_args = torch.argsort(logits, dim=1, descending=True)
            for k in range(output.shape[0]):
                if target[k] != sort_args[k, 0]:
                    continue
                self.cur_epoch_lams[target[k]] += logits[k]
                self.cur_epoch_cnt[target[k]] += 1    
    
    def train(self):
        self.cur_epoch_lams = torch.zeros(self.args.num_classes, self.args.num_classes, dtype=torch.float32).cuda()
        self.cur_epoch_cnt = torch.zeros(self.args.num_classes, dtype=torch.float32).cuda()
        
        self.net = self.net.cuda()
        self.net.train()
        loss_recoder = AvgMeter()
        dataset = DataSet(self.args.img_path, self.args.img_txt_train, self.train_transform)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=8)

        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd)
        if self.args.method == 'baseline':
            self.criterion = nn.CrossEntropyLoss().cuda()
        elif self.args.method == 'ls':
            self.criterion = label_smooth(self.args.num_classes).cuda()
        elif self.args.method == 'ols':
            self.criterion = nn.CrossEntropyLoss().cuda()
        elif self.args.method == 'tfkd':
            self.criterion = tf_kd_loss

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[45, 80], gamma=0.1)

        f = open('{}/log.txt'.format(self.args.expname), 'w')
        writer = SummaryWriter(self.args.expname)
        
        cur_loss = 0
        best_top1, best_top5 = 0.0, 0.0
        istep = 0
        for epoch in range(self.args.epochs):
            self.cur_epoch_lams = torch.zeros(self.args.num_classes, self.args.num_classes, dtype=torch.float32).cuda()
            self.cur_epoch_cnt = torch.zeros(self.args.num_classes, dtype=torch.float32).cuda()
            for i, (x, y) in enumerate(dataloader):
                x = x.cuda()
                y = y.cuda()
                y = y.squeeze(1)
                target = y
                optimizer.zero_grad()
         
                output = self.net(x)
                if self.args.method == 'ols':
                    self.update_loss_lams(output, target)
                    loss = self.soft_cross_entropy(output, target) * 1.0 + self.criterion(output, target) * 1.0
                else:
                    if self.args.method == 'ls':
                        loss = self.criterion(output, target)
                    elif self.args.method == 'tf_kd':
                        loss = self.criterion(output, target)
                    else:
                        loss = self.criterion(output, target)
                
                loss.backward()
                optimizer.step()

                loss_recoder.update(loss.item(), 1)
                cur_loss += loss.item()
                time_now = datetime.now().strftime('%H:%M:%S')
                
                #record loss
                writer.add_scalar("loss", loss.item(), istep)
                istep+=1
                
                if (i+1) % (len(dataset)//2//self.args.batch_size) == 0:
                    print('%s [epoch %d/%d, iter %d/%d] lr = %f cur_loss = %f avg_loss = %f' % (time_now, epoch, self.args.epochs, i, len(dataloader), optimizer.param_groups[0]['lr'], cur_loss/100, loss_recoder.avg))
                    cur_loss = 0
                    
            if self.args.method == 'ols':   
                for cls in range(self.args.num_classes):
                    if self.cur_epoch_cnt[cls].max() < 0.5:
                        self.loss_lams[cls] = 1. / self.args.num_classes 
                    else:
                        # It is empirically found that adding restrictions here can be better for fine-grained classification
                        # Not for ImageNet or CIFAR
                        if self.loss_lams[cls].max() >= 0.88:
                            continue
                        else:
                            self.loss_lams[cls] = self.cur_epoch_lams[cls] / self.cur_epoch_cnt[cls]    

            scheduler.step()
            top1, top5 = self.test()
            writer.add_scalar("Top-1 ACC", top1, epoch)
            writer.add_scalar("Top-5 ACC", top5, epoch)
            torch.save(self.net.state_dict(), f'./{self.args.expname}/{epoch}.pth')
            if top1 > best_top1:
                best_top1, best_top5 = top1, top5
                torch.save(self.net.state_dict(), '{}/best.pth'.format(self.args.expname))
            print('Currently Best top-1 = {}, top-5 = {}'.format(best_top1, best_top5))
            f.writelines('Currently Best top-1 = {}, top-5 = {}'.format(best_top1, best_top5))
        f.close()
        writer.close()        
        

    def test(self, epochs=99):

        dataset = DataSet(self.args.img_path, self.args.img_txt_test, self.test_transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)

        criterion = nn.CrossEntropyLoss().cuda()
        # test the top-1 acc and top-5 acc
        top1, top5 = self.validate(dataloader, criterion, self.args)

        # please refer to [ICML 2019] Making Convolutional Networks Shift-Invariant Again
        # https://arxiv.org/pdf/1904.11486.pdf
        # test the robustness acc to shift transform, which is not considered in our paper
        val_dataset = DataSet(self.args.img_path, self.args.img_txt_test, self.val_transform)
        if (epochs+1) % 10 == 0: 
           print('start to validate shift:')
           val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)
           validate_shift(val_dataloader, self.net, self.args)

        # please refer to [ICML 2019] Making Convolutional Networks Shift-Invariant Again
        # https://arxiv.org/pdf/1904.11486.pdf
        # test the robustness acc to diagonal transform, which is not considered in our paper  
        if (epochs+1) % 10 == 0:
           print('start to validate diagonal:')
           torch.cuda.empty_cache()
           val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8)
           validate_diagonal(val_dataloader, self.net, self.args)

        self.net.train()
        return top1, top5
    
    # the function to test the model ensemble performance
    def test_ensemble(self, epochs=99):
        pth = self.args.ensemble
        cnt = len(pth)
        print(f'>>>>>>>>>>>ensemble {cnt} model<<<<<<<<<<<<<')
        net = []
        for i in range(cnt):
            net.append(resnet.resnet50(num_classes=self.args.num_classes))
        for i in range(cnt):
            print(pth[i])
            net[i].load_state_dict(torch.load(pth[i]), strict=True)
        
        for i in range(cnt):
            net[i].cuda()
            net[i].eval()
        
        dataset = DataSet(self.args.img_path, self.args.img_txt_test, self.test_transform)

        val_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)
        
        top1 = AvgMeter()
        top5 = AvgMeter()
       
        
        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                target = target.squeeze(1)
                # compute output
                output = net[0](input)
                output = torch.softmax(output, dim=1)
                for k in range(1, cnt):
                    output += torch.softmax(net[k](input), dim=1)
                
                
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1[0], input.size(0))
                top5.update(acc5[0], input.size(0))

                if i % self.args.print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                           i, len(val_loader), top1=top1, top5=top5))

            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
        print(f'Top-1 ACC: {top1.avg}, Top-5 ACC: {top5.avg}')

    def validate(self, val_loader, criterion, args):
        batch_time = AvgMeter()
        losses = AvgMeter()
        top1 = AvgMeter()
        top5 = AvgMeter()
        model = self.net

        # switch to evaluate mode
        model.eval()
        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(val_loader):
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                target = target.squeeze(1)
                # compute output
                output = model(input)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(acc1[0], input.size(0))
                top5.update(acc5[0], input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                           i, len(val_loader), batch_time=batch_time, loss=losses,
                           top1=top1, top5=top5))

            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
        self.net.train()
        return top1.avg, top5.avg

def validate_shift(val_loader, model, args):
    batch_time = AvgMeter()
    consist = AvgMeter()

    # switch to evaluate mode

    with torch.no_grad():
        end = time.time()
        for ep in range(args.epochs_shift):
            for i, (input, target) in enumerate(val_loader):
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                off0 = np.random.randint(32,size=2)
                off1 = np.random.randint(32,size=2)
                output0 = model(input[:,:,off0[0]:off0[0]+224,off0[1]:off0[1]+224])
                output1 = model(input[:,:,off1[0]:off1[0]+224,off1[1]:off1[1]+224])

                cur_agree = agreement(output0, output1).type(torch.FloatTensor).cuda()

                # measure agreement and record
                consist.update(cur_agree.item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    print('Ep [{0}/{1}]:\t'
                        'Test: [{2}/{3}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Consist {consist.val:.4f} ({consist.avg:.4f})\t'.format(
                        ep, args.epochs_shift, i, len(val_loader), batch_time=batch_time, consist=consist))

        print(' * Consistency {consist.avg:.3f}'
            .format(consist=consist))

    return consist.avg

def validate_diagonal(val_loader, model, args):
    batch_time = AvgMeter()
    prob = AvgMeter()
    top1 = AvgMeter()
    top5 = AvgMeter()

    # switch to evaluate mode

    D = 33
    diag_probs = np.zeros((len(val_loader.dataset),D))
    diag_probs2 = np.zeros((len(val_loader.dataset),D)) # save highest probability, not including ground truth
    diag_corrs = np.zeros((len(val_loader.dataset),D))

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            target = target.squeeze(1)
            inputs = []
            for off in range(D):
                inputs.append(input[:,:,off:off+224,off:off+224])
            inputs = torch.cat(inputs, dim=0)
            probs = torch.nn.Softmax(dim=1)(model(inputs))
            corrs = probs.argmax(dim=1).cpu().data.numpy() == target.item()
            outputs = 100.*probs[:,target.item()]
            acc1, acc5 = accuracy(probs, target.repeat(D), topk=(1, 5))

            probs[:,target.item()] = 0
            probs2 = 100.*probs.max(dim=1)[0].cpu().data.numpy()

            diag_probs[i,:] = outputs.cpu().data.numpy()
            diag_probs2[i,:] = probs2
            diag_corrs[i,:] = corrs

            # measure agreement and record
            prob.update(np.mean(diag_probs[i,:]), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Prob {prob.val:.4f} ({prob.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, prob=prob, top1=top1, top5=top5))

    print(' * Prob {prob.avg:.3f} Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(prob=prob,top1=top1, top5=top5))

    np.save(os.path.join(args.out_dir,'diag_probs'),diag_probs)
    np.save(os.path.join(args.out_dir,'diag_probs2'),diag_probs2)
    np.save(os.path.join(args.out_dir,'diag_corrs'),diag_corrs)

def validate_save(val_loader, mean, std, args):
    import matplotlib.pyplot as plt
    import os
    for i, (input, target) in enumerate(val_loader):
        img = (255*np.clip(input[0,...].data.cpu().numpy()*np.array(std)[:,None,None] + mean[:,None,None],0,1)).astype('uint8').transpose((1,2,0))
        plt.imsave(os.path.join(args.out_dir,'%05d.png'%i),img)

# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
def save_checkpoint(state, is_best, epoch, out_dir='./'):
    torch.save(state, os.path.join(out_dir,'checkpoint.pth.tar'))
    if(epoch % 10 == 0):
        torch.save(state, os.path.join(out_dir,'checkpoint_%03d.pth.tar'%epoch))
    if is_best:
        shutil.copyfile(os.path.join(out_dir,'checkpoint.pth.tar'), os.path.join(out_dir,'model_best.pth.tar'))

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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def agreement(output0, output1):
    pred0 = output0.argmax(dim=1, keepdim=False)
    pred1 = output1.argmax(dim=1, keepdim=False)
    agree = pred0.eq(pred1)
    agree = 100.*torch.mean(agree.type(torch.FloatTensor).cuda())
    return agree