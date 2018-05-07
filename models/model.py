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

class LeNet(BasicModule):
    def __init__(self):
        # the input size: (batch, 3, 32, 32)
        super(LeNet, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 6, 3),
                                    nn.ReLU(True),
                                    nn.MaxPool2d(2, 2),
                                    nn.Conv2d(6, 16, 5),
                                    nn.ReLU(True),
                                    nn.MaxPool2d(2, 2))
        self.classifier = nn.Sequential(nn.Linear(400, 120),
                                        nn.ReLU(True),
                                        nn.Linear(120, 84),
                                        nn.ReLU(True),
                                        nn.Linear(84, 10))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x