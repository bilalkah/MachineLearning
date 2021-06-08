# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channel = 3
num_classes = 10
learning_rate = 0.001
batch_size = 1024
num_epochs = 5
load_model = True

import sys

class Identity(nn.Module):
    def __init__(self):
        super(Identity,self,).__init__()

    def forward(self,x):
        return x

# Load pretrain model and modify it
model = torchvision.models.vgg16(pretrained=True)

for param in model.parameters():
    param.requires_grad = False
    
model.avgpool = Identity()
#model.classifier = nn.Linear(512,10)
model.classifier = nn.Sequential(
    nn.Linear(512,100),
    nn.ReLU(),
    nn.Linear(100,10),
)
model.to(device=device)

"""
model.classifier = nn.Sequential(nn.Linear(512,100),nn.Dropout(p=0.5) ,nn.Linear(100,10))
"""

"""
for i in range(0,7):
    model.classifier[i] = Identity()
"""


# Load Data
train_dataset = datasets.CIFAR10(
    root='dataset/',
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
)
test_dataset = datasets.CIFAR10(
    root='dataset/',
    train=False,
    transform=transforms.ToTensor(),
    download=True,
)
test_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
)
# Initialize network


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)


# Train network
for epoch in range(num_epochs):
    losses = []

    print(f"Epoch: {epoch}")
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        scores = model(data)
        loss = criterion(scores,targets)
        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
    
    mean_loss = sum(losses) / len(losses)
    print(f"Loss at epoch {epoch} was {mean_loss:.5f}")
# Check accuracy on training and test to see how good our model

def check_accuracy(loader, model):

    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)

            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
    model.train()


check_accuracy(train_loader,model)
check_accuracy(test_loader,model)