import torch as t
import os
import numpy as np
from PIL import Image
from torchvision import transforms 
from torch.utils.data import Dataset
import pandas as pd
import re
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.autograd import Variable
import random
import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import shutil

 
 
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']
 
 
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
 
 
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

#import moxing as mox
#mox.file.shift('os', 'mox')
posi1 = random.randint(1, 100)
posi2 = random.randint(1, 100)
size = 224

#data_dir = 's3://classifier-gar/train_data/'
#train_dir = 's3://classifier-gar/trainset/'
#test_dir = 's3://classifier-gar/testset/'

data_dir = 'D:/garbage/try0/try/'
train_dir = 'D:/garbage/try0/trainset_try/'
test_dir = 'D:/garbage/try0/testset_try/'


def data_divide(data_dir, train_dir, test_dir):
    for i in range(1000):#数据集最后一项的数字，不是数据集总数
        imgpath = data_dir + "img_" + str(i) + ".jpg"
        txtpath = data_dir + "img_" + str(i) + ".txt"
        if mox.file.exists(imgpath):
            if random.randint(0, 9) < 3:#30%概率数据选中为
                #mox.file.copy(imgpath, test_dir)
                #mox.file.copy(txtpath, test_dir)
                shutil.copy(imgpath, test_dir)
                shutil.copy(txtpath, test_dir)
                print("No." + str(i) + " has been divided into testset\n")
            else:
                #mox.file.copy(imgpath, train_dir)
                #mox.file.copy(txtpath, train_dir)
                shutil.copy(imgpath, train_dir)
                shutil.copy(txtpath, train_dir)
                print("No." + str(i) + " has been divided into trainset\n")

transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ])                
                

if posi1 % 20 != 0:
    if posi2 % 5 == 0:
        size = 224
    elif posi2 % 5 == 1:
        size = 245
    elif posi2 % 5 == 2:
        size = 274
    elif posi2 % 5 == 3:
        size = 316
    elif posi2 % 5 == 4:
        size = 354

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.05, contrast=0.1, saturation=0.3, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])



class DataClassify(Dataset):
    def __init__(self, root, transforms=None, mode=None):
        #存放图像地址
        self.imgs = [x.path for x in mox.file.scan_dir(root) if
            x.name.endswith(".jpg")]
        self.labels = [y.path for y in mox.file.scan_dir(root) if
            y.name.endswith(".txt")]
        self.transforms = transforms
        
    def __getitem__(self, index):
        #读取图像数据并返回
        img_path = self.imgs[index]
        label = int(re.sub('\D','',mox.file.read(self.labels[index])[-4:]))
        data = Image.open(img_path)
        if self.transforms:
            data = self.transforms(data)
        return data, label
    
    def __len__(self):
        return len(self.imgs)
def ListToTensor(lister):
    nparr = np.asarray(lister)
    tens = t.from_numpy(nparr)
    return tens



class BasicBlock(nn.Module):
    expansion = 1
 
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        residual = x
        out = self.relu(x)
 
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out = out + residual
        out = self.relu(out)
 
        return out
class Bottleneck(nn.Module):
    expansion = 4
 
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        residual = x
 
        out = self.relu(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out = out + residual
        out = self.relu(out)
 
        return out
class ResNet(nn.Module):
 
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
 
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
 
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
 
        return nn.Sequential(*layers)
 
    def forward(self, x):
        x = self.relu(x)
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
def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model
 
 
def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model
 
 
def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model
 
 
def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model
 
 
def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model



net = resnet50()
print(net)
coefficient = 5#超参数，学习率衰减系数，与模型有关，需要调参

def dloss(loss_list, lr=0.0001):
    if len(loss_list) <=2:
        return 0.001#超参数,初始学习率，与训练样本集有关，与模型关系较小，确定训练集后即可固定
    if (loss_list[-2] - loss_list[-1])/loss_list[-2] > 0.02:
        return lr
    return lr/coefficient


epochs = 2
Loss_list = []   #
Accurate_list = []

batch_size = 64
data_divide(data_dir, train_dir, test_dir)
trainset = DataClassify(train_dir, transforms=transform)
testset = DataClassify(test_dir)
#dataset = DataClassify('s3://classifier-gar/train_data/', transforms=transform)
#testset = DataClassify('s3://classifier-gar/try/')

data_loader_train = t.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
testloader = t.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)
#testloader = t.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)

net = resnet50()
cost = t.nn.CrossEntropyLoss()
optimizer = t.optim.Adam(list(net.parameters()), lr=dloss(Loss_list, lr=0.0001),weight_decay=1e-8)


def evaluate(model, testloader):#testset需要根据接口调整下面的部分代码，必须调整的代码会给出注释
    model.eval()#这个必须有，train里面也有对应的那个，好像可以不加，但我建议加上
    correct = 0.0
    run_correct = 0.0
    total = len(testloader)#test.dataset or something else
    for data in iter(testloader):#这一代码块是照着训练部分抄的，但事实上test这里应该是全集，
    #按照训练代码，这里只是一个batch，所以这里应该需要修改，函数传进来的testloader本身就是数据集全集
    #然后遍历所有batch之后再输出correct
        X_test, X_label = data
        label = []
        for la in X_label:
            label.append(la)
        label = ListToTensor(label)
        X_test = Variable(X_test)
        X_label = Variable(label)
        with torch.no_grad():
            logit = model(X_test)
            _, pred = t.max(F.softmax(outputs, dim=1).data, 1)
            run_correct = run_correct + (pred == X_label.data).sum()
            corr = (1.*run_correct).item()
    print("test_accura: ")#这里建议
    print(corr/total)
    print("-----------------------------------------------")



net.train()

for i, (X_data, X_label) in enumerate(data_loader_train):
    #X_data, X_label
    optimizer.zero_grad()
    output = net(X_data)
    loss = F.nll_loss(output, X_label)
    loss.backward()
    Loss_list.append(loss.item())
    optimizer.step()

    if i % 10 == 0:
        print(i, loss.item())
        
evaluate(net, testloader)



"""        
net.eval()
test_loss, correct = 0, 0

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss = test_loss + F.nll_loss(output, target, size_average=False).item()
        pred = output.argmax(1, keepdim=True)
        correct = correct + pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_data)
acc = correct / len(test_data)
print(acc, test_loss)

"""

"""

for epoch in range(epochs):
    print("epoch " + str(epoch+1) + " start training...")
    run_loss = 0.0
    run_correct = 0.0
    total = 0
    
    num = 0
    for data in iter(data_loader_train):
        X_train, X_label = data
        total += len(X_train)
        label = []
        for la in X_label:
            label.append(la)
        label = ListToTensor(label)
        
        X_train = Variable(X_train)
        X_label = Variable(label)
        optimizer.zero_grad()
        outputs = net(X_train)
        _, pred = t.max(F.softmax(outputs, dim=1).data, 1)
        loss = cost(outputs, X_label)
        run_loss += loss.data
        print("run loss: ")
        print(run_loss)
        print("-----------------------------------------------")
        
        loss.backward()
        optimizer.step()
        #run_correct += t.sum(pred == X_label.data)
        run_correct += (pred == X_label.data).sum()
        corr = (1.*run_correct).item()
        print("run_correct: ")
        print(corr)
        print("-----------------------------------------------")
        Loss_list.append(run_loss/total)
        Accurate_list.append(corr/total)
        
        print("total: ")
        print(total)
        print("-----------------------------------------------")
        
    num = math.ceil(total/batch_size) * epochs

    print('epoch %d, loss %.6f, currect %.4f ------' %(epoch+1, run_loss/total, corr/total))
    print("loss list: ")
    print(Loss_list)
    print("Accurate list: ")
    print(Accurate_list)

x1 = range(0, num)
x2 = range(0, num)
y1 = Accurate_list
y2 = Loss_list
plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'o-')
plt.title('Test accuracy vs. epoches')
plt.ylabel('Test accuracy')
plt.subplot(2, 1, 2)
plt.plot(x2, y2, '.-')
plt.xlabel('Test loss vs. epoches')
plt.ylabel('Test loss')
plt.show()

"""
