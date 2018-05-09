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

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
                                nn.BatchNorm2d(outchannel),
                                nn.ReLU(True),
                                nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
                                nn.BatchNorm2d(outchannel))
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)

class ResNet(BasicModule):
    def __init__(self):
        super(ResNet, self).__init__()
        self.pre = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(True),
                                nn.MaxPool2d(3, 2, 1))

        self.layer1 = self._make_layer(64, 128, 3)
        self.layer2 = self._make_layer(128, 256, 4, stride=1)
        self.layer3 = self._make_layer(256, 512, 6, stride=1)
        self.layer4 = self._make_layer(512, 512, 3, stride=1)

        self.fc = nn.Linear(512, 10)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        shortcut = nn.Sequential(nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
                                nn.BatchNorm2d(outchannel))
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

if __name__ == '__main__':
    m = ResNet()
    data = torch.randn((1, 3, 32, 32))
    print(m(data).size())
