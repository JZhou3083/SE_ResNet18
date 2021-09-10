import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchbearer

from torchsummary import summary
import torchvision
from torchvision import transforms 
from torchvision.datasets import ImageFolder

import os
import argparse


from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchbearer.callbacks import TensorBoard


import matplotlib
import matplotlib.pyplot as plt

import numpy as np


from torchvision.models import resnet18
from torch.utils.data import random_split


#### hyperparas
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()
## load model
args.resume=True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

##### data 
print('==> Preparing data..')
preprocess_input = {
    'train': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
batch_size = 128
# Data set
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=preprocess_input['train'])
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=preprocess_input['test'])

# dataloader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#### Blocks to build SE_ResNet18
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
# SE module
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

## SE blocks
class SEBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out += residual
        out = self.relu(out)

        return out

## SEResnet module 
class SEResNet(nn.Module):
    def __init__(self, block, n_size, num_classes=10, reduction=16):
        super(SEResNet, self).__init__()
        self.inplane = 64
        self.conv1 = nn.Conv2d(
            3, self.inplane, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1,dilation=1)
        self.layer1 = self._make_layer(
            block, 64, blocks=n_size, stride=1, reduction=reduction)
        self.layer2 = self._make_layer(
            block, 128, blocks=n_size, stride=2, reduction=reduction)
        self.layer3 = self._make_layer(
            block, 256, blocks=n_size, stride=2, reduction=reduction)
        self.layer4 = self._make_layer(
            block, 512, blocks=n_size, stride=2, reduction=reduction)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, reduction):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, stride, reduction))
            self.inplane = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
def model_create(SE=True,reduct=8):
    """Construct the model
    """    
    if SE:
      model=SEResNet(SEBasicBlock,2,num_classes=10)
    else:
      model=resnet18(pretrained=False)
    
    model.avgpool = nn.AdaptiveAvgPool2d((1,1))
    model.fc = nn.Linear(512, 10)  
    return model

print('==> Building model..')
### resnet-18 training

#create the model 
ResNet18=model_create(SE=False)
SE_ResNet18 = model_create(SE=True)

### model sumarry
#print("==>No Model structure: ")
#summary(ResNet18,(3,32,32))
#print("==>SE Model structure: ")
#summary(SE_ResNet18,(3,32,32))


## resnet18 first 
ResNet18.to(device)
if device =="cuda":
    ResNet18=torch.nn.DataParallel(ResNet18)
    cudnn.benchmark=True
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/resnet18.pth')
    ResNet18.load_state_dict(checkpoint['ResNet18'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

### test

# define the loss function and the optimiser
loss_function = nn.CrossEntropyLoss()

# optimiser, learning rate scheduler
optimiser = optim.SGD(ResNet18.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=300)# get outof local minima 

from torchbearer import Trial
ResNet18.eval()
trial_res18=Trial(ResNet18,criterion=loss_function,metrics=['cat_acc']).to(device)
trial_res18.with_test_generator(testloader)

predictions = trial_res18.predict(data_key=torchbearer.TEST_DATA)
predicted_classes = predictions.argmax(1).cpu()
true_classes = list(testset.targets)

from sklearn import metrics
print("Confusion natrix of ResNet-18:")
print(metrics.confusion_matrix(true_classes, predicted_classes))




### SE-resnet18 now 

SE_ResNet18.to(device)
if device =="cuda":
    SE_ResNet18=torch.nn.DataParallel(SE_ResNet18)
    cudnn.benchmark=True
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/se_resnet18.pth')
    SE_ResNet18.load_state_dict(checkpoint['SE_ResNet18'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

### test

# define the loss function and the optimiser
loss_function = nn.CrossEntropyLoss()

# optimiser, learning rate scheduler
optimiser = optim.SGD(SE_ResNet18.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=300)# get outof local minima 

from torchbearer import Trial
SE_ResNet18.eval()
trial_SE_ResNet18=Trial(SE_ResNet18,criterion=loss_function,metrics=['cat_acc']).to(device)
trial_SE_ResNet18.with_test_generator(testloader)

predictions = trial_SE_ResNet18.predict(data_key=torchbearer.TEST_DATA)
predicted_classes = predictions.argmax(1).cpu()
true_classes = list(testset.targets)

from sklearn import metrics
print("Confusion natrix of SE-ResNet-18:")
print(metrics.confusion_matrix(true_classes, predicted_classes))
