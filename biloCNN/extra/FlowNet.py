import torch
import torch.nn as nn

class FlowNetSimple(nn.Module):
    def __init__(self):
        super(FlowNetSimple,self).__init__()
        self.conv1 = nn.ModuleDict({
            "conv1" : nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size = 64,
                stride=2
            ),
            "batchnorm" : nn.BatchNorm2d(64)
        })
    
    def forward(self,x):
        pass
        
class FlowNetCorr(nn.Module):
    def __init__(self):
        super(FlowNetSimple,self).__init__()
        

