import torch
import torch.nn as nn

DenseNet_arch = {
    "DenseNet-121": [6, 12, 24, 16],
    "DenseNet-169": [6, 12, 32, 32],
    "DenseNet-201": [6, 12, 48, 32],
    "DenseNet-161": [6, 12, 36, 24],
}

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        )
    def forward(self, x):
        return self.conv(x)
    
class TransitionBlock(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(TransitionBlock, self).__init__()
        self.transition_layer = nn.Sequential(
            ConvBlock(in_channels, out_channels, 1, 1, 0),
            nn.AvgPool2d(2, 2)
        )
    def forward(self, x):
        return self.transition_layer(x)

class DenseBlock(nn.Module):
    def __init__(self, growth_rate, num_layer):
        super(DenseBlock, self).__init__()
        self.dense_blocks = self.make_block(growth_rate, num_layer)
    
    def make_block(self, growth_rate, num_layer):
        blocks = []
        for i in range(1,num_layer+1):
            blocks.append(ConvBlock(growth_rate*i, growth_rate*4, 1, 1, 0))
            blocks.append(ConvBlock(growth_rate*4, growth_rate, 3, 1, 1))
        return nn.Sequential(*blocks)
    def forward(self,x):
        for i in range(0,len(self.dense_blocks),2):
            y = self.dense_blocks[i:i+2](x)
            torch.stack([x,y],dim=1)
            x = y
        return x


class DenseNet(nn.Module):
    def __init__(self, num_classes, model_arch="DenseNet-121", growth_rate=12):
        super(DenseNet, self).__init__()
        self.arch = DenseNet_arch[model_arch]
        self.model = nn.Sequential(
            nn.Conv2d(3,growth_rate,7,2,3),
            nn.MaxPool2d(3,2,1)
            DenseBlock(growth_rate, self.arch[0]),
            TransitionBlock(growth_rate*(self.arch[0]+1), growth_rate),
            DenseBlock(growth_rate, self.arch[1]),
            TransitionBlock(growth_rate*(self.arch[1]+1), growth_rate),
            DenseBlock(growth_rate, self.arch[2]),
            TransitionBlock(growth_rate*(self.arch[2]+1), growth_rate),
            DenseBlock(growth_rate, self.arch[3]),
            nn.AvgPool2d(7,1).view(-1,growth_rate*(self.arch[3]+1)),
            nn.Linear(growth_rate*(self.arch[3]+1), num_classes)
        )

if __name__ == '__main__':
    net = torch.max_pool2d(kernel_size=3, stride=2, padding=1)
    x = torch.randn(4,12,112,112)
    print(net(x).shape)