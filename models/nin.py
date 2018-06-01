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


class NIN(BasicModule):
    # the input size: (batch, 3, 32, 32)
    def __init__(self):
        super(NIN, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 192, 5, 1, 1),
                                    nn.BatchNorm2d(192),
                                    nn.ReLU(True),
                                    nn.Conv2d(192, 160, 1),
                                    nn.BatchNorm2d(160),
                                    nn.ReLU(True),
                                    nn.Conv2d(160, 96, 1),
                                    nn.BatchNorm2d(96),
                                    nn.ReLU(True),
                                    nn.MaxPool2d(3, 3),
                                    nn.Dropout(p=0.5),
                                    nn.Conv2d(96, 192, 5, 1, 1),
                                    nn.BatchNorm2d(192),
                                    nn.ReLU(True),
                                    nn.Conv2d(192, 192, 1),
                                    nn.BatchNorm2d(192),
                                    nn.ReLU(True),
                                    nn.Conv2d(192, 192, 1),
                                    nn.BatchNorm2d(192),
                                    nn.ReLU(True),
                                    nn.MaxPool2d(3, 3),
                                    nn.Dropout(p=0.5),
                                    nn.Conv2d(192, 192, 3, 1, 1),
                                    nn.BatchNorm2d(192),
                                    nn.ReLU(True),
                                    nn.Conv2d(192, 192, 1),
                                    nn.BatchNorm2d(192),
                                    nn.ReLU(True),
                                    nn.Conv2d(192, 192, 1),
                                    nn.BatchNorm2d(192),
                                    nn.ReLU(True),
                                    nn.Conv2d(192, 10, 1),
                                    nn.BatchNorm2d(10),
                                    nn.ReLU(True),
                                    nn.AvgPool2d(2, 2),
                                    nn.Softmax(dim=1)
                                    )

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)