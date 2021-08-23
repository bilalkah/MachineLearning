import torch
import torch.nn as nn

MobileNetV1_arch = [
    [1, 32, 2],
    [32, 64, 1],
    [64, 128, 2],
    [128, 128, 1],
    [128, 256, 2],
    [256, 256, 1],
    [256, 512, 2],
    [512, 512, 1],
    [512, 512, 1],
    [512, 512, 1],
    [512, 512, 1],
    [512, 512, 1],
    [512, 1024, 2],
    [1024, 1024, 1],
]

class MobileNet(nn.Module):
    def __init__(self,num_classes=1000):
        super(MobileNet, self).__init__()

        self.features = self.make_layers(MobileNetV1_arch);
        self.fc = nn.Linear(1024, num_classes)
    
    def make_layers(self,arch):
        features = []
        for _, (in_channels, out_channels, stride) in enumerate(arch):
            if _ == 0:
                features.append(self.conv_bn(in_channels, out_channels, stride))
            else:
                features.append(self.conv_dw(in_channels, out_channels, stride))
        features.append(nn.AvgPool2d(7))
        return nn.Sequential(*features)
            
    def conv_bn(self,in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def conv_dw(self,in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.fc(self.features(x).view(-1, 1024))
    
if __name__ == "__main__":
    model = MobileNet(10)
    x = torch.randn(4,3,224,224)
    print(model(x).shape)