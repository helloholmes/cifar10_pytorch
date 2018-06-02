# coding:utf-8
'''
python 3.5
pytorch 0.4.0
visdom 0.1.7
torchnet 0.0.2
auther: helloholmes
'''
import torch
import time
import torch.nn.functional as F
import torchvision.models as models
from torch import nn

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

class Mixed(nn.Module):
    def __init__(self):
        # input channel: 64
        # output channel: 160
        super(Mixed, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = BasicConv2d(64, 96, 3, 2)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out

class Inception_A(nn.Module):
    def __init__(self):
        # input channel: 384
        # output channel: 384
        super(Inception_A, self).__init__()
        self.branch0 = BasicConv2d(384, 96, 1, 1)

        self.branch1 = nn.Sequential(
            BasicConv2d(384, 64, 1, 1),
            BasicConv2d(64, 96, 3, 1, 1))

        self.branch2 = nn.Sequential(
            BasicConv2d(384, 64, 1, 1),
            BasicConv2d(64, 96, 3, 1, 1),
            BasicConv2d(96, 96, 3, 1, 1))

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(384, 96, 1, 1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class Reduction_A(nn.Module):
    def __init__(self):
        # input channel: 384
        # output channel: 1024
        super(Reduction_A, self).__init__()
        self.branch0 = BasicConv2d(384, 384, 3, 2)

        self.branch1 = nn.Sequential(
            BasicConv2d(384, 192, 1, 1),
            BasicConv2d(192, 224, 3, 1, 1),
            BasicConv2d(224, 256, 3, 2))

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out

class Inception_B(nn.Module):
    def __init__(self):
        # input channel: 1024
        # output channel: 1024
        super(Inception_B, self).__init__()
        self.branch0 = BasicConv2d(1024, 384, 1, 1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1024, 192, 1, 1),
            BasicConv2d(192, 224, (1, 7), 1, (3, 0)),
            BasicConv2d(224, 256, (7, 1), 1, (0, 3)),)

        self.branch2 = nn.Sequential(
            BasicConv2d(1024, 192, 1, 1),
            BasicConv2d(192, 192, (7, 1), 1, (3, 0)),
            BasicConv2d(192, 224, (1, 7), 1, (0, 3)),
            BasicConv2d(224, 224, (7, 1), 1, (3, 0)),
            BasicConv2d(224, 256, (1, 7), 1, (0, 3)),)

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(1024, 128, 1, 1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class Reduction_B(nn.Module):
    def __init__(self):
        # input channel: 1024
        # output channel: 1536
        super(Reduction_B, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(1024, 192, 1, 1),
            BasicConv2d(192, 192, 3, 2))

        self.branch1 = nn.Sequential(
            BasicConv2d(1024, 256, 1, 1),
            BasicConv2d(256, 256, (1, 7), 1, (0, 3)),
            BasicConv2d(256, 320, (7, 1), 1, (3, 0)),
            BasicConv2d(320, 320, 3, 2))

        self.branch2 = nn.MaxPool2d(3, 2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out

class Inception_with_auxiliary_loss(BasicModule):
    def __init__(self):
        super(Inception_with_auxiliary_loss, self).__init__()
        self.f1 = nn.Sequential(
            BasicConv2d(3, 32, 3, 1, 1),
            BasicConv2d(32, 32, 3, 1, 1),
            BasicConv2d(32, 64, 3, 1, 1),
            Mixed(),
            BasicConv2d(160, 384, 3, 1, 1),)
        self.f2 = nn.Sequential(
            Inception_A(),
            Inception_A(),
            Inception_A(),)
        self.f3 = nn.Sequential(
            Reduction_A(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Reduction_B())

        self.l1 = nn.Sequential(
            nn.AvgPool2d(5, count_include_pad=False),
            nn.Conv2d(384, 256, 1, 1),)
        self.l1_ = nn.Sequential(
            nn.Linear(2304, 10),
            nn.Softmax(1))

        self.l2 = nn.Sequential(
            nn.AvgPool2d(5, count_include_pad=False),
            nn.Conv2d(384, 256, 1, 1),)
        self.l2_ = nn.Sequential(
            nn.Linear(2304, 10),
            nn.Softmax(1))

        self.avg_pool = nn.AvgPool2d(3, count_include_pad=False)
        self.linear = nn.Linear(1536, 10)
        self.softmax = nn.Softmax(1)

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

    def predict(self, x):
        x = self.f1(x)
        loss1 = self.l1(x)
        loss1 = loss1.view(loss1.size(0), -1)
        loss1 = self.l1_(loss1)
        x = self.f2(x)
        loss2 = self.l2(x)
        loss2 = loss2.view(loss2.size(0), -1)
        loss2 = self.l2_(loss2)
        x = self.f3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        loss3 = self.softmax(x)
        return loss1, loss2, loss3

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.softmax(x)
        return x

if __name__ == '__main__':
    m = Inception_with_auxiliary_loss()
    data = torch.randn((8, 3, 32, 32))
    l1, l2, l3 = m.predict(data)
    print(l1.size(), l2.size(), l3.size())