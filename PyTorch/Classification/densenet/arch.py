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
    def __init__(self, in_channels):
        super(TransitionBlock, self).__init__()
        self.transition_layer = nn.Sequential(
            ConvBlock(in_channels, in_channels, 1, 1, 1),
            nn.AvgPool2d(2, 2)
        )
    def forward(self, x):
        return self.transition_layer(x)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layer):
        super(DenseBlock, self).__init__()
        self.dense_block = self.make_block(in_channels, growth_rate, num_layer)
    
    def make_block(self, in_channels, growth_rate, num_layer):
        block = []
        for i in range(num_layer):
            block.append(ConvBlock(in_channels*(i+1), 4*growth_rate,1,1,0))
            block.append(ConvBlock(4*growth_rate,in_channels,3,1,1))
        return nn.Sequential(*block)
    
    def forward(self,x):
        list = []
        for i in range(1,len(self.dense_block)+1):
            list.append(x)
            out = x
            
        list = []
        list.append(x)
        return self.dense_block[0:2](list[0:i*2])


class DenseNet(nn.Module):
    pass

if __name__ == '__main__':
    net = DenseBlock(12,12,6)
    x = torch.randn(4,12,56,56)
    print(net(x).shape)