from tqdm import tqdm
from time import sleep
import torch
import torch.nn as nn
from PyTorch.implementations.efficientnet import EfficientNet
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)


def save_checkpoint(state, filename="effnet.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state,filename)


def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

phi_values = {
    # tuple of: (phi_value, resolution, drop_rate)
    "b0": (0, 224, 0.2), #alpha, beta, gamma, depth = alpha ** phi
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}

device = "cuda" if torch.cuda.is_available() else "cpu"

in_channel = 3
num_classes = 10
learning_rate = 0.001
batch_size = 16
num_epochs = 3
load_model = False

# Load Data
train_dataset = datasets.CIFAR10(
    root='dataset/',
    train=True,
    transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224))]),
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
    transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224))]),
    download=True,
)
test_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

version = "b1"
phi,res,drop_rate = phi_values[version]

model = EfficientNet(
    version=version,
    num_classes=num_classes,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

if load_model:
    load_checkpoint(torch.load("effnet.pth.tar"))

for epoch in range(num_epochs):
    with tqdm(train_loader, unit="batch") as tepoch:
        losses = []

        for data,targets in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            data = data.to(device)
            targets = targets.to(device)

            scores = model(data)
            loss = criterion(scores,targets)
            losses.append(loss.item())

            #backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent of adam step
            optimizer.step()

            tepoch.set_postfix(loss=loss.item())
            sleep(0.1)
        mean_loss = sum(losses) / len(losses)
        print(f"loss at epoch {epoch} was {mean_loss:.5f}")
    if epoch % 3 == 0 and epoch!=0:
        checkpoint = {'state_dict':model.state_dict(),'optimizer':optimizer.state_dict()}
        save_checkpoint(checkpoint)
# check accuracy on training and test to see how good our model

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

check_accuracy(test_loader,model)