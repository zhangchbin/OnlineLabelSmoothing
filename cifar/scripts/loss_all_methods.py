import torch
import numpy as np

import torch.nn as nn

class SCELoss(torch.nn.Module):
    '''
    2019 - iccv - Symmetric cross entropy for robust learning with noisy labels
    '''
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, pred, target):
        pred = self.softmax(pred)
        label_one_hot = torch.nn.functional.one_hot(target, self.num_classes).float().to(self.device)

        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)

        ce = (-1*torch.sum(label_one_hot * torch.log(pred), dim=1)).mean()
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1)).mean()
        loss = self.alpha * ce + self.beta * rce

        return loss
                
class label_smooth(torch.nn.Module):
    def __init__(self, num_classes):
        super(label_smooth, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_classes = num_classes
        self.softmax = torch.nn.Softmax(dim=1)
        self.alpha = 0.1
    def forward(self, output, target):
        probs = self.softmax(output)
        label_one_hot = torch.nn.functional.one_hot(target, self.num_classes).float().to(self.device)
        label_one_hot = label_one_hot * (1. - self.alpha) + self.alpha / float(self.num_classes)
        loss = torch.sum(-label_one_hot * torch.log(probs), dim=1).mean()
        return loss

class generalized_cross_entropy(torch.nn.Module):
    '''
    2018 - nips - Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels.
    '''
    def __init__(self, num_classes):
        super(generalized_cross_entropy, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.q = 0.7
        self.num_classes = num_classes
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, output, target):
        probs = self.softmax(output)
        label_one_hot = torch.nn.functional.one_hot(target, self.num_classes).float().to(self.device)
        
        loss = (1 - torch.pow(torch.sum(label_one_hot * probs, dim=1), self.q)) / self.q
        return loss.mean()

class joint_optimization(torch.nn.Module):
    '''
    2018 - cvpr - Jonit Optimization framework for learning with noisy labels
    '''
    def __init__(self, num_classes):
        super(joint_optimization, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_classes = num_classes
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, output, target):
        probs = self.softmax(output)
        avg_probs = torch.mean(probs, dim=0)
        p = (torch.ones(self.num_classes, dtype=torch.float32) / float(self.num_classes)).cuda()
        label_one_hot = torch.nn.functional.one_hot(target, self.num_classes).float().to(self.device)
        
        l_c = - torch.mean(torch.sum(torch.nn.functional.log_softmax(output, dim=1) * label_one_hot, dim=1))
        l_p = - torch.sum(torch.log(avg_probs) * p)
        l_e = - torch.mean(torch.sum(torch.nn.functional.log_softmax(output, dim=1) * probs, dim=1))
        
        loss = l_c + 1.2 * l_p + 0.8 * l_e
        return loss

class boot_soft(torch.nn.Module):
    '''
    2015 - iclrws - Training deep neural networks on noisy with bootstrapping
    '''
    def __init__(self, num_classes):
        super(boot_soft, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_classes = num_classes
        self.softmax = torch.nn.Softmax(dim=1)
        self.beta = 0.95
    def forward(self, output, target):
        probs = self.softmax(output)
        label_one_hot = torch.nn.functional.one_hot(target, self.num_classes).float().to(self.device)
        
        probs = torch.clamp(probs, 1e-7, 1. - 1e-7)
        return -torch.sum((self.beta * label_one_hot + (1-self.beta) * probs) * torch.log(probs), dim=1).mean()

class boot_hard(torch.nn.Module):
    '''
    2015 - iclrws - Training deep neural networks on noisy with bootstrapping
    '''
    def __init__(self, num_classes):
        super(boot_hard, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_classes = num_classes
        self.softmax = torch.nn.Softmax(dim=1)
        self.beta = 0.8
    def forward(self, output, target):
        probs = self.softmax(output)
        label_one_hot = torch.nn.functional.one_hot(target, self.num_classes).float().to(self.device)
        
        probs = torch.clamp(probs, 1e-7, 1. - 1e-7)
        pred_labels = torch.nn.functional.one_hot(torch.argmax(probs, 1), self.num_classes)
        return -torch.sum((self.beta * label_one_hot + (1.-self.beta) * pred_labels) * torch.log(probs), dim=1).mean()

class Forward(torch.nn.Module):
    '''
    Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach
    '''
    def __init__(self, num_classes):
        super(Forward, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_classes = num_classes
        self.softmax = torch.nn.Softmax(dim=1)
        self.beta = 0.8
        self.p = torch.eye(self.num_classes, dtype=torch.float32).cuda()
        self.p.requires_grad = False
    
    def forward(self, output, target):
        probs = self.softmax(output)
        label_one_hot = torch.nn.functional.one_hot(target, self.num_classes).float().to(self.device)
        
        probs = torch.clamp(probs, 1e-7, 1. - 1e-7)
        return -torch.sum(label_one_hot * torch.log(torch.matmul(probs, self.p)), dim=1).mean()
        
class Backward(torch.nn.Module):
    '''
    Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach
    '''
    def __init__(self, num_classes):
        super(Backward, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_classes = num_classes
        self.softmax = torch.nn.Softmax(dim=1)
        self.beta = 0.8
        self.p = torch.eye(self.num_classes, dtype=torch.float32).cuda()
        self.p.requires_grad = False
    
    def forward(self, output, target):
        probs = self.softmax(output)
        label_one_hot = torch.nn.functional.one_hot(target, self.num_classes).float().to(self.device)
        p_inv = torch.inverse(self.p)
        probs = torch.clamp(probs, 1e-7, 1. - 1e-7)
        return -torch.sum(torch.matmul(label_one_hot, p_inv) * torch.log(probs), dim=1).mean()

class DisturbLabel(torch.nn.Module):
    '''
    2016 - cvpr - DisturbLabel: Regularizing CNN on the Loss Layer
    '''
    def __init__(self, num_classes):
        super(DisturbLabel, self).__init__()
        self.noisy_rate = 0.1
        self.num_classes = num_classes
        self.bound = (num_classes - 1.) / float(num_classes) * self.noisy_rate
    def forward(self, output, target):
        batchsize = output.shape[0]
        new_target = target.clone()
        for kk in range(batchsize):
            r = torch.rand(1)
            if r < self.bound:
                dlabel = torch.randint(low=0, high=self.num_classes, size=(1,))
                while new_target[kk] == dlabel[0]:
                    dlabel = torch.randint(low=0, high=self.num_classes, size=(1,))
                new_target[kk] = dlabel[0]
        return torch.nn.functional.cross_entropy(output, new_target)

class PC(torch.nn.Module):
    def __init__(self, num_classes):
        super(PC, self).__init__()
        self.lamda = 10.
    def forward(self, output, target):
        batch_size = output.shape[0]
        batch_left = output[:batch_size // 2]
        batch_right = output[batch_size // 2:]
        loss1 = torch.nn.functional.cross_entropy(output, target)
        loss2 = 0.
        loss2 = torch.norm((batch_left - batch_right).abs(), 2, dim=0).sum()
        loss2 = loss2 / batch_size
        loss = loss1 + 10 * loss2
        return loss

