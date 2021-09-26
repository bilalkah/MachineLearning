from loss import YoloLoss
loss = YoloLoss()
import torch
pred = torch.randn(5,7,7,30)
target = torch.randn(5,7,7,30)
print(loss(pred,target))