from Classification.mobilenetv2 import arch
from CustomLoss.losses import CFocalLoss
from model import Model

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms.transforms import Compose

import tqdm
from time import sleep


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 10
lr = 1e-4
batch_size = 32
epochs = 100

# create dataset from directory "cinic10" for torchvision
train_cinic10 = datasets.ImageFolder(root='cinic10/cinic10/train',
                                     transform=Compose([
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomVerticalFlip(),
                                         transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                     ]))

train_loader = DataLoader(
    dataset=train_cinic10,
    batch_size=batch_size,
    shuffle=True,
)

valid_cinic10 = datasets.ImageFolder(root='cinic10/horse-test',
                                     transform=Compose([
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomVerticalFlip(),
                                         transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                     ]))

valid_loader = DataLoader(
    dataset=valid_cinic10,
    batch_size=10,
    shuffle=True,
)

net = arch.MobileNetV2(10).to(device)
optimizer = optim.Adam(net.parameters(),lr=lr)
metric = torchmetrics.Accuracy().to(device)
alpha = 0.84
gamma = 2.8
criterion = CFocalLoss(alpha,gamma).to(device)
model = Model(net,device)

model.train(train_loader,valid_loader,epochs,batch_size,lr,optimizer,criterion,"mobilenet")
