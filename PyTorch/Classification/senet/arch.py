# Squeeze and Excitation block 
# implemented with in darknet19

import torch
import torch.nn as nn

net_cfgs = [
            # conv1s
            [(32, 3)],
            ['M', (64, 3)],
            ['M', (128, 3), (64, 1), (128, 3)],
            ['M', (256, 3), (128, 1), (256, 3)],
            ['M', (512, 3), (256, 1), (512, 3), (256, 1), (512, 3)],
            # conv2
            ['M', (1024, 3), (512, 1), (1024, 3), (512, 1), (1024, 3)],
            # ------------
            # conv3
            #[(1024, 3), (1024, 3)],
            # conv4
            #[(1024, 3)],
            [(1000,1)]
        ]

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.block = nn.Sequential(
            nn.Linear(in_channels,in_channels//reduction),
            nn.ReLU(),
            nn.Linear(in_channels//reduction,in_channels),
            nn.Sigmoid(),
        )
    def forward(self,x):
        y = self.avg_pool(x).view(x.size(0),-1)
        y = self.block(y)
        return x*y.view(y.size(0),y.size(1),1,1)


class Darknet19withSE(nn.Module):
    def __init__(self,num_classes=1000):
        super(Darknet19withSE,self).__init__()
        self.features = self.create_darknet19_features(num_classes)
        self.fc = nn.Linear(1000,num_classes)
    def create_darknet19_features(self,num_classes):
        features = []
        in_channels = 3
        for _,layers in enumerate(net_cfgs):
            for layer in layers:
                if layer == 'M':
                    features.append(nn.MaxPool2d(kernel_size=(2,2),stride=2))
                else:
                    out_channels,kernel_size = layer
                    if kernel_size == 1:
                        features.append(nn.Conv2d(in_channels,out_channels,kernel_size,1))
                        features.append(SEBlock(out_channels))
                    else:
                        features.append(nn.Conv2d(in_channels,out_channels,kernel_size,1,1))
                        features.append(SEBlock(out_channels))
                    features.append(nn.BatchNorm2d(out_channels))
                    features.append(nn.LeakyReLU(0.1))
                    in_channels = out_channels
        features.append(nn.AdaptiveAvgPool2d((1,1)))
        return nn.Sequential(*features)  
    
    def forward(self,x):
        x = self.features(x).view(x.size(0),-1)
        return self.fc(x)


        

if __name__ == '__main__':
    x = torch.randn(64,3,224,224)
    net = Darknet19withSE(10)
    print(net(x).shape)