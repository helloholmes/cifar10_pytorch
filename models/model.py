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

class VGG19(BasicModule):
    def __init__(self):
        # the input size: (batch, 3, 32, 32)
        super(VGG19, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1),
                                    nn.ReLU(True),
                                    nn.Conv2d(64, 64, 3, 1, 1),
                                    nn.ReLU(True),
                                    nn.MaxPool2d(2, 2),
                                    nn.Conv2d(64, 128, 3, 1, 1),
                                    nn.ReLU(True),
                                    nn.Conv2d(128, 128, 3, 1, 1),
                                    nn.ReLU(True),
                                    nn.MaxPool2d(2, 2),
                                    nn.Conv2d(128, 256, 3, 1, 1),
                                    nn.ReLU(True),
                                    nn.Conv2d(256, 256, 3, 1, 1),
                                    nn.ReLU(True),
                                    nn.Conv2d(256, 256, 3, 1, 1),
                                    nn.ReLU(True),
                                    nn.Conv2d(256, 256, 3, 1, 1),
                                    nn.ReLU(True),
                                    nn.MaxPool2d(2, 2),
                                    nn.Conv2d(256, 512, 3, 1, 1),
                                    nn.ReLU(True),
                                    nn.Conv2d(512, 512, 3, 1, 1),
                                    nn.ReLU(True),
                                    nn.Conv2d(512, 512, 3, 1, 1),
                                    nn.ReLU(True),
                                    nn.Conv2d(512, 512, 3, 1, 1),
                                    nn.ReLU(True),
                                    nn.MaxPool2d(2, 2),
                                    nn.Conv2d(512, 512, 3, 1, 1),
                                    nn.ReLU(True),
                                    nn.Conv2d(512, 512, 3, 1, 1),
                                    nn.ReLU(True),
                                    nn.Conv2d(512, 512, 3, 1, 1),
                                    nn.ReLU(True),
                                    nn.Conv2d(512, 512, 3, 1, 1),
                                    nn.ReLU(True)
                                    )

        self.classifier = nn.Sequential(nn.Linear(2048, 4096),
                                        nn.ReLU(True),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(4096, 4096),
                                        nn.ReLU(True),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(4096, 10))

    def forward(self, x):
        x = self.features(x)        
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class VGG19_pretrain(BasicModule):
    # the input size: (batch, 3, 32, 32)
    def __init__(self):
        super(VGG19_pretrain, self).__init__()
        model = models.vgg19(pretrained=True)
        self.features = model.features
        # for param in self.features.parameters():
        #     param.requires_grad = False
        self.classifier = nn.Sequential(nn.Linear(512, 4096),
                                        nn.ReLU(True),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(4096, 4096),
                                        nn.ReLU(True),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(4096, 10))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class NIN(BasicModule):
    # the input size: (batch, 3, 32, 32)
    def __init__(self):
        super(NIN, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 192, 5, 1, 1),
                                    nn.ReLU(True),
                                    nn.Conv2d(192, 160, 1),
                                    nn.ReLU(True),
                                    nn.Conv2d(160, 96, 1),
                                    nn.ReLU(True),
                                    nn.MaxPool2d(3, 3),
                                    nn.Dropout(p=0.5),
                                    nn.Conv2d(96, 192, 5, 1, 1),
                                    nn.ReLU(True),
                                    nn.Conv2d(192, 192, 1),
                                    nn.ReLU(True),
                                    nn.Conv2d(192, 192, 1),
                                    nn.ReLU(True),
                                    nn.MaxPool2d(3, 3),
                                    nn.Dropout(p=0.5),
                                    nn.Conv2d(192, 192, 3, 1, 1),
                                    nn.ReLU(True),
                                    nn.Conv2d(192, 192, 1),
                                    nn.ReLU(True),
                                    nn.Conv2d(192, 192, 1),
                                    nn.ReLU(True),
                                    nn.Conv2d(192, 10, 1),
                                    nn.ReLU(True),
                                    nn.AvgPool2d(2, 2),
                                    nn.Softmax(dim=1)
                                    )

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

if __name__ == '__main__':
    data = torch.randn((128, 3, 32, 32))
    m = NIN()
    print(m(data).size())
