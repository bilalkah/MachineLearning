from torchvision.transforms.transforms import Compose
from mobilenetv2 import arch
from model import Model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm
from time import sleep
import torchmetrics
import torch.nn.functional as F
from losses import CFocalLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 10
lr = 1e-4
batch_size = 20
epochs = 100

# create dataset from directory "cinic10" for torchvision
train_cinic10 = datasets.ImageFolder(root='cinic10/train',
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
    batch_size=batch_size,
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
