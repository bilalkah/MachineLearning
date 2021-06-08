import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class LeNet(nn.Module):
    def __init__(self, in_channel=1, num_classes=10):
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=6,
            kernel_size=(5,5),
        )
        self.avgpool = nn.AvgPool2d(2,2)
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=(5,5),
        )
        self.avgpool1 = nn.AvgPool2d(2,2)
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=120,
            kernel_size=(5,5),
        )
        self.fc1 = nn.Linear(120,84)
        self.fc2 = nn.Linear(84,num_classes)
        

    def forward(self,x):
        x = torch.tanh(self.conv1(x))
        x = self.avgpool(x)
        x = torch.tanh(self.conv2(x))
        x = self.avgpool1(x)
        x = torch.tanh(self.conv3(x))
        x = x.reshape(x.shape[0],-1)
        x = torch.tanh(self.fc1(x))
        x = F.softmax(self.fc2(x),dim=0)

model = LeNet()
print(model)

x = torch.randn(64,1,32,32)
print(model(x).shape)

"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

in_channel = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

train_dataset = datasets.MNIST(
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
test_dataset = datasets.MNIST(
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
        
model = LeNet().to(device=device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range(num_epochs):

    for batch_idx, (data,targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        scores = model(data)
        loss = criterion(scores,targets)
        
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    print(f"epoch {epoch}")

"""