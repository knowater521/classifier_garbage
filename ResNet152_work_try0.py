%matplotlib inline
import torch as t
import os
import numpy as np
from PIL import Image
from torchvision import transforms 
from torch.utils.data import Dataset
import pandas as pd
import re
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import random
import math
import moxing as mox
mox.file.shift('os', 'mox')
posi1 = random.randint(1, 100)
posi2 = random.randint(1, 100)
size = 224

if posi1 % 20 == 0:
    size = 224
    
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
    #transforms.RandomHorizonotalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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
class ResBlock(nn.Module):
    #残差块
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut
        
    def forward(self, x):
        out = self.left(x)
        
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)
class ResNet(nn.Module):
    def __init__(self, num_classes=40):
        super(ResNet, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = self._make_layer(64, 256, 3)
        self.layer2 = self._make_layer(256, 512, 8, stride=2)
        self.layer3 = self._make_layer(512, 1024, 36, stride=2)
        self.layer4 = self._make_layer(1024, 2048, 3, stride=2)
        
        self.fc = nn.Linear(2048, num_classes)
        
    
    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        #构建包含多个残差块的layer
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        layers = []
        layers.append(ResBlock(inchannel, outchannel, stride, shortcut))
        
        for i in range(1, block_num):
            layers.append(ResBlock(outchannel, outchannel))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.pre(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        return self.fc(x)
ResNet()
batch_size = 64
dataset = DataClassify('s3://classifier-gar/train_data/', transforms=transform)
data_loader_train = t.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
net = ResNet()
cost = t.nn.CrossEntropyLoss()
optimizer = t.optim.Adam(list(net.parameters()), lr=0.0001,weight_decay=1e-8)
epochs = 2
Loss_list = []
Accurate_list = []

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

