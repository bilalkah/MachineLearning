import torch
import torch.nn as nn

darknet_cfgs = [
    #in_channels, out_channels, kernel_size, stride, padding
    [3, 64, 7, 2, 3],
    ["MaxPool",0,2,2,0],
    
    [64, 192, 3, 1, 1],
    ["MaxPool",0,2,2,0],
    
    [192, 128, 1, 1, 0],
    [128, 256, 3, 1, 1],
    [256, 256, 1, 1, 0],
    [256, 512, 3, 1, 1],
    ["MaxPool",0,2,2,0],
    
    [512, 256, 1, 1, 0],
    [256, 512, 3, 1, 1],
    [512, 256, 1, 1, 0],
    [256, 512, 3, 1, 1],
    [512, 256, 1, 1, 0],
    [256, 512, 3, 1, 1],
    [512, 256, 1, 1, 0],
    [256, 512, 3, 1, 1],
    [512, 1024, 3, 1, 1],
    ["MaxPool",0,2,2,0],
    
    [1024, 512, 1, 1, 0],
    [512, 1024, 3, 1, 1],
    [1024, 512, 1, 1, 0],
    [512, 1024, 3, 1, 1],    
]

class DarkNet(nn.Module):
    def __init__(self, num_classes=10, include_top = True):
        super(DarkNet,self).__init__()
        
        self.num_classes = num_classes
        self.include_top = include_top
        
        self.conv_layers = self._make_conv_layers()
        self.classifier = self._make_classifier()
    
    def forward(self, x):
        if self.include_top:
            return self.classifier(self.conv_layers(x))
        else:
            return self.conv_layers(x)
    
    def _make_conv_layers(self):
        conv_layers = []
        for (in_channels, out_channels, kernel_size, stride, padding) in darknet_cfgs:
            if in_channels == "MaxPool":
                conv_layers.append(nn.MaxPool2d(kernel_size,stride))
            else:
                conv_layers.append(nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding))
                conv_layers.append(nn.BatchNorm2d(out_channels))
                conv_layers.append(nn.LeakyReLU(0.1,inplace=True))
        return nn.Sequential(*conv_layers)
    
    def _make_classifier(self):
        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1,1)),
            nn.Flatten(),
            nn.Linear(1024, self.num_classes)
        )
        return classifier



class Yolov1(nn.Module):
    def __init__(self, num_classes):
        super(Yolov1, self).__init__()
        
        self.c = num_classes
        self.b = 2
        self.s = 7
        
        self.backbone = DarkNet(num_classes=self.c, include_top=False)
        self.detection = self._make_detection_layers()
        self.prediction = self._make_prediction_layers()
    
    def _make_detection_layers(self):
        detection = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(negative_slope=0.1,inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(negative_slope=0.1,inplace=True),
            
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(negative_slope=0.1,inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(negative_slope=0.1,inplace=True),
        )
        return detection
        
    def _make_prediction_layers(self):
        prediction = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=7*7*1024, out_features=4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=self.s*self.s*(self.b*5+self.c)),
            nn.Sigmoid(),
        )
        return prediction

    def forward(self, x):
        return self.prediction(self.detection(self.backbone(x))).view(-1,self.s,self.s,5*self.b+self.c)

if __name__ == '__main__':
    net = Yolov1(num_classes=20).to('cuda')
    x = torch.randn(1,3,448,448).to('cuda')
    print(net(x).shape)