'''ResNeXt in PyTorch.
See the paper "Aggregated Residual Transformations for Deep Neural Networks" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                 stride=stride, padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1,
                 stride=stride, bias=False)
def branchBottleneck(channel_in, channel_out, kernel_size): 
    middle_channel = channel_out // 4
    return nn.Sequential(
        nn.Conv2d(channel_in, middle_channel, kernel_size=1, stride=1),
        nn.BatchNorm2d(middle_channel),
        nn.ReLU(),
        nn.Conv2d(middle_channel, middle_channel, kernel_size=kernel_size,stride=kernel_size),
        nn.BatchNorm2d(middle_channel),
        nn.ReLU(),
        nn.Conv2d(middle_channel, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU()
        )

class Block(nn.Module):
    '''Grouped convolution block.'''
    expansion = 2

    def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=1):
        super(Block, self).__init__()
        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(group_width, self.expansion*group_width, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*group_width)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*group_width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*group_width, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*group_width)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNeXt(nn.Module):
    def __init__(self, num_blocks, cardinality, bottleneck_width, num_classes=10):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(num_blocks[0], 1)
        self.layer2 = self._make_layer(num_blocks[1], 2)
        self.layer3 = self._make_layer(num_blocks[2], 2)
        # self.layer4 = self._make_layer(num_blocks[3], 2)
        self.linear = nn.Linear(cardinality*bottleneck_width*8, num_classes)

        num_c = cardinality * bottleneck_width * 2
        self.bottleneck1_1 = branchBottleneck(num_c, num_c * 4, kernel_size=8)
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.middle_fc1 = nn.Linear(num_c * 4, num_classes)

        self.bottleneck2_1 = branchBottleneck(num_c * 2, num_c * 4, kernel_size=4)
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.middle_fc2 = nn.Linear(num_c * 4, num_classes)
 
 
    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_planes, self.cardinality, self.bottleneck_width, stride))
            self.in_planes = Block.expansion * self.cardinality * self.bottleneck_width
        # Increase bottleneck_width by 2 after each stage.
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
       
        middle_out1 = self.bottleneck1_1(out)
      
        middle_out1 = self.avgpool1(middle_out1)
        middle_out1 = torch.flatten(middle_out1, 1)
        middle_out1 = self.middle_fc1(middle_out1)        

        out = self.layer2(out)   
        middle_out2 = self.bottleneck2_1(out)
        middle_out2 = self.avgpool2(middle_out2)
        middle_out2 = torch.flatten(middle_out2, 1)
        middle_out2 = self.middle_fc2(middle_out2)        

        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, middle_out1, middle_out2


def ResNeXt29_2x64d(num_classes):
    return ResNeXt(num_blocks=[3,3,3], cardinality=2, bottleneck_width=64, num_classes=num_classes)

def ResNeXt29_4x64d(num_classes):
    return ResNeXt(num_blocks=[3,3,3], cardinality=4, bottleneck_width=64, num_classes=num_classes)

def ResNeXt29_8x64d(num_classes):
    return ResNeXt(num_blocks=[3,3,3], cardinality=8, bottleneck_width=64, num_classes=num_classes)

def ResNeXt29_32x4d(num_classes):
    return ResNeXt(num_blocks=[3,3,3], cardinality=32, bottleneck_width=4, num_classes=num_classes)

