# Class for covolutional layers and blocks

import torch
import torch.nn as nn

class Layers:
    class Conv2d(nn.Module):
        def __init__(self, *args,**kwargs):
            super(Layers.Conv2d, self).__init__()
            self.conv = nn.Conv2d(*args,**kwargs)
        
        def forward(self, x):
            return self.conv(x)
        
    class Flatten(nn.Module):
        def __init__(self):
            super(Layers.Flatten, self).__init__()
        
        def forward(self, x):
            return x.view(x.size(0), -1)
        
    class Linear(nn.Module):
        def __init__(self, *args,**kwargs):
            super(Layers.Linear, self).__init__()
            self.linear = nn.Linear(*args,**kwargs)
        
        def forward(self, x):
            return self.linear(x)
        
    class Dropout(nn.Module):
        def __init__(self, *args,**kwargs):
            super(Layers.Dropout, self).__init__()
            self.dropout = nn.Dropout(*args,**kwargs)
        
        def forward(self, x):
            return self.dropout(x)
    
    class BatchNorm2d(nn.Module):
        def __init__(self, *args,**kwargs):
            super(Layers.BatchNorm2d, self).__init__()
            self.batchnorm = nn.BatchNorm2d(*args,**kwargs)
        
        def forward(self, x):
            return self.batchnorm(x)
        
    class MaxPool2d(nn.Module):
        def __init__(self, *args,**kwargs):
            super(Layers.MaxPool2d, self).__init__()
            self.maxpool = nn.MaxPool2d(*args,**kwargs)
        
        def forward(self, x):
            return self.maxpool(x)
        
    class AvgPool2d(nn.Module):
        def __init__(self, *args,**kwargs):
            super(Layers.AvgPool2d, self).__init__()
            self.avgpool = nn.AvgPool2d(*args,**kwargs)
        
        def forward(self, x):
            return self.avgpool(x)
        
    class AdaptiveAvgPool2d(nn.Module):
        def __init__(self, *args,**kwargs):
            super(Layers.AdaptiveAvgPool2d, self).__init__()
            self.avgpool = nn.AdaptiveAvgPool2d(output_size = tuple(kwargs['output_size']))
        
        def forward(self, x):
            return self.avgpool(x)
        
    class ReLU(nn.Module):
        def __init__(self, *args,**kwargs):
            super(Layers.ReLU, self).__init__()
            self.relu = nn.ReLU(*args,**kwargs)
        
        def forward(self, x):
            return self.relu(x)
        
    class LeakyReLU(nn.Module):
        def __init__(self, *args, **kwargs):
            super(Layers.LeakyReLU, self).__init__()
            self.leakyrelu = nn.LeakyReLU(*args, **kwargs)
            
        def forward(self, x):
            return self.leakyrelu(x)
        
    def __getattribute__(self, name):
        if name == "conv2d":
            return Layers.Conv2d
        elif name == "flatten":
            return Layers.Flatten
        elif name == "linear":
            return Layers.Linear
        elif name == "dropout":
            return Layers.Dropout
        
        elif name == "bnorm2d":
            return Layers.BatchNorm2d
        
        elif name == "maxpool2d":
            return Layers.MaxPool2d
        elif name == "avgpool2d":
            return Layers.AvgPool2d
        elif name == 'adaptiveavgpool2d':
            return Layers.AdaptiveAvgPool2d
        
        elif name == "relu":
            return Layers.ReLU
        
        elif name == "leakyrelu":
            return Layers.LeakyReLU
        
        else:
            return super(Layers, self).__getattribute__(name)
            
    


