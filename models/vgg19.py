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

class VGG19(BasicModule):
    def __init__(self):
        # the input size: (batch, 3, 32, 32)
        super(VGG19, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(True),
                                    nn.Conv2d(64, 64, 3, 1, 1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(True),
                                    nn.MaxPool2d(2, 2),
                                    nn.Conv2d(64, 128, 3, 1, 1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(True),
                                    nn.Conv2d(128, 128, 3, 1, 1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(True),
                                    nn.MaxPool2d(2, 2),
                                    nn.Conv2d(128, 256, 3, 1, 1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(True),
                                    nn.Conv2d(256, 256, 3, 1, 1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(True),
                                    nn.Conv2d(256, 256, 3, 1, 1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(True),
                                    nn.Conv2d(256, 256, 3, 1, 1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(True),
                                    nn.MaxPool2d(2, 2),
                                    nn.Conv2d(256, 512, 3, 1, 1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True),
                                    nn.Conv2d(512, 512, 3, 1, 1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True),
                                    nn.Conv2d(512, 512, 3, 1, 1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True),
                                    nn.Conv2d(512, 512, 3, 1, 1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True),
                                    nn.MaxPool2d(2, 2),
                                    nn.Conv2d(512, 512, 3, 1, 1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True),
                                    nn.Conv2d(512, 512, 3, 1, 1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True),
                                    nn.Conv2d(512, 512, 3, 1, 1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True),
                                    nn.Conv2d(512, 512, 3, 1, 1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True)
                                    )

        self.classifier = nn.Sequential(nn.Linear(2048, 4096),
                                        nn.ReLU(True),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(4096, 4096),
                                        nn.ReLU(True),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(4096, 10))

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
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x