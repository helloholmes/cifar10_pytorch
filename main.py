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
from models import model
from torchvision import transforms as T
from torch.utils.data import DataLoader
from utils.visualize import Visualizer
from config import DefaultConfig

def dataloader(root, batch_size, num_workers):
    transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))])
    trainset = tv.datasets.CIFAR10(root=root, train=True,
                                download=False, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                        shuffle=True, num_workers=num_workers)

    testset = tv.datasets.CIFAR10(root=root, train=False,
                                download=False, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)

    return trainloader, testloader

def train(opt):
    model_train = getattr(model, opt.model)()
    vis = Visualizer(opt.env)

    if opt.use_gpu:
        model_train.cuda()

    trainloader, testloader = dataloader(opt.root, opt.batch_size, opt.num_workers)

    criterion = torch.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = torch.optim.SGD(model_train.parameters(),
                                lr=lr,
                                momentum=0.9,
                                weight_decay=opt.weight_decay,
                                nesterov=True)

    # meter
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(10)
    previous_loss = 1e100

    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        confusion_matrix.reset()

        for ii, (data, label) in tqdm(enumerate(trainloader)):
            if opt.use_gpu:
                data = data.cuda()
                label = label.cuda()

            optimizer.zero_grad()
            score = model_train(data)
            loss = criterion(score, label)
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.item())
            confusion_matrix.add(score.data, label.data)

            if ii % opt.print_freq:
                vis.plot('loss', loss_meter.value()[0])

        test_cm, test_accuracy = test(model_train, testloader, opt)
        vis.plot('test_accuracy', test_accuracy)

        # updata learning rate
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            for param in optimizer.param_groups:
                param['lr'] = lr

        previous_loss = loss_meter.value()[0]

    model_train.save(opt.save_model_path+opt.model+'_'+str(epoch))

def test(model_train, dataloader, opt):
    model_train.eval()
    confusion_matrix = meter.ConfusionMeter(10)
    total_num = 0
    correct_num = 0
    for ii, (data, label) in tqdm(enumerate(dataloader)):
        if opt.use_gpu:
            data = data.cuda()

        score = model_train(data)

        confusion_matrix.add(score.data.squeeze(), label.type(torch.LongTensor))
        _, predict = torch.max(score.data, 1)
        total_num += label.size(0)
        correct_num += (predict.cpu() == label).sum()

    model_train.train()
    accuracy = 100 * float(correct_num) / float(total_num)

    return confusion_matrix, accuracy

if __name__ == '__main__':

    opt = DefaultConfig()
    opt.parse({'model': 'VGG19', 'max_epoch': 30, 'weight_decay': 1e-4, 'lr': 0.1})

    train(opt)
