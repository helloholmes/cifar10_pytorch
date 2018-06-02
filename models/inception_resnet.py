# coding:utf-8
'''
python 3.5
pytorch 0.4.0
visdom 0.1.7
torchnet 0.0.2
auther: helloholmes
'''
import os
import torch
import torchvision as tv
from torch import nn
from tqdm import tqdm
from torchnet import meter
from torch.nn import functional as F

class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%Y_%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)

class BasicConv2d(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(inchannel, outchannel, kernel_size=kernel_size,
                            stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(outchannel,
                                eps=0.001,
                                momentum=0.1,
                                affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Mixed_A(nn.Module):
    def __init__(self):
        # input channel: 192
        # output channel: 320
        super(Mixed_A, self).__init__()
        self.branch0 = BasicConv2d(192, 96, 1, 1)

        self.branch1 = nn.Sequential(
            BasicConv2d(192, 48, 1, 1),
            BasicConv2d(48, 64, 5, 1, 2))

        self.branch2 = nn.Sequential(
            BasicConv2d(192, 64, 1, 1),
            BasicConv2d(64, 96, 3, 1, 1),
            BasicConv2d(96, 96, 3, 1, 1))

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(192, 64, 1, 1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class Block35(nn.Module):
    def __init__(self, scale=1.0):
        # input channel: 320
        # output channel: 320
        super(Block35, self).__init__()
        self.scale = scale

        self.branch0 = BasicConv2d(320, 32, 1, 1)

        self.branch1 = nn.Sequential(
            BasicConv2d(320, 32, 1, 1),
            BasicConv2d(32, 32, 3, 1, 1))

        self.branch2 = nn.Sequential(
            BasicConv2d(320, 32, 1, 1),
            BasicConv2d(32, 48, 3, 1, 1),
            BasicConv2d(48, 64, 3, 1, 1))

        self.conv2d = nn.Conv2d(128, 320, 1, 1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out

class Mixed_B(nn.Module):
    def __init__(self):
        # input channel: 320
        # output channel: 1088
        super(Mixed_B, self).__init__()
        self.branch0 = BasicConv2d(320, 384, 3, 2)

        self.branch1 = nn.Sequential(
            BasicConv2d(320, 256, 1, 1),
            BasicConv2d(256, 256, 3, 1, 1),
            BasicConv2d(256, 384, 3, 2))

        self.branch2 = nn.AvgPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out

class Block17(nn.Module):
    def __init__(self, scale=1.0):
        # input channel: 1088
        # output channel: 1088
        super(Block17, self).__init__()
        self.scale = scale 

        self.branch0 = BasicConv2d(1088, 192, 1, 1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 128, 1, 1),
            BasicConv2d(128, 160, (1, 7), 1, (0, 3)),
            BasicConv2d(160, 192, (7, 1), 1, (3, 0)))

        self.conv2d = nn.Conv2d(384, 1088, 1, 1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out

class Mixed_C(nn.Module):
    def __init__(self):
        # input channel: 1088
        # output channel: 2080
        super(Mixed_C, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(1088, 256, 1, 1),
            BasicConv2d(256, 384, 3, 2))

        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 256, 1, 1),
            BasicConv2d(256, 288, 3, 2))

        self.branch2 = nn.Sequential(
            BasicConv2d(1088, 256, 1, 1),
            BasicConv2d(256, 288, 3, 1, 1),
            BasicConv2d(288, 320, 3, 2))

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class Block8(nn.Module):
    def __init__(self, scale=1.0, noReLU=False):
        # input channel: 2080
        # output channel: 2080
        super(Block8, self).__init__()
        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(2080, 192, 1, 1)

        self.branch1 = nn.Sequential(
            BasicConv2d(2080, 192, 1, 1),
            BasicConv2d(192, 224, (1, 3), 1, (0, 1)),
            BasicConv2d(224, 256, (3, 1), 1, (1, 0)))

        self.conv2d = nn.Conv2d(448, 2080, 1, 1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out

class Inception_ResNet(BasicModule):
    def __init__(self):
        super(Inception_ResNet, self).__init__()      
        self.pre = nn.Sequential(
            BasicConv2d(3, 32, 3, 1, 1),
            BasicConv2d(32, 32, 3, 1, 1),
            BasicConv2d(32, 64, 3, 2, 1),
            BasicConv2d(64, 128, 1, 1),
            BasicConv2d(128, 192, 3, 1, 1),)
        self.mixed_a = Mixed_A()
        self.repeat_a = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),)
        self.mixed_b = Mixed_B()
        self.repeat_b = nn.Sequential(
            Block17(scale=0.1),
            Block17(scale=0.1),
            Block17(scale=0.1),
            Block17(scale=0.1),
            Block17(scale=0.1),
            Block17(scale=0.1),
            Block17(scale=0.1),
            Block17(scale=0.1),
            Block17(scale=0.1),)
        self.mixed_c = Mixed_C()
        self.repeat_c = nn.Sequential(
            Block8(scale=0.2),
            Block8(scale=0.2),
            Block8(scale=0.2),
            Block8(scale=0.2),
            Block8(scale=0.2),)
        self.last_block = Block8(0.2, True)
        self.features = nn.Sequential(
            self.pre,
            self.mixed_a,
            self.repeat_a,
            self.mixed_b,
            self.repeat_b,
            self.mixed_c,
            self.repeat_c,
            self.last_block)
        self.avg_pool = nn.AvgPool2d(3, count_include_pad=False)
        self.linear = nn.Linear(2080, 10)

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

if __name__ == '__main__':
    m = Inception_ResNet()
    data = torch.randn((1, 3, 32, 32))
    print(m(data).size())
    # print(m)