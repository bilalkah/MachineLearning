import torch
import torch.nn as nn


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
            ConvBlock(in_channels, out_channels, 1, 1, 1),
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
        for i in range(len(0,self.dense_blocks,2)):
            y = self.dense_blocks[i:i+2](x)
            torch.stack([x,y],dim=1)
            x = y
        return x


class DenseNet(nn.Module):
    pass

if __name__ == '__main__':
    net = torch.max_pool2d(kernel_size=3, stride=2, padding=1)
    x = torch.randn(4,12,112,112)
    print(net(x).shape)