# Test ../model/layers.py

import sys
sys.path.insert(0, "/home/bilal/repos/biloCNN/")
print(sys.path[0])

import torch
import torch.nn as nn
import torch.optim as optim # Optimizers

from builder.layers import Layers

test_layer = Layers()
args = list()
kwargs = {"in_channels":3, "out_channels":3, "kernel_size":3}
conv = getattr(test_layer, "Conv2d")(*args,**kwargs)
print(conv)
assert isinstance(conv,nn.Module)

from builder.optimizers import Optimizers

# create basic model
model = nn.Sequential(
    getattr(test_layer, "Conv2d")(*args,**kwargs),
)
test_optimizer = Optimizers()
args = list()
kwargs = {"lr":0.01,"params":model.parameters()}
sgd = getattr(test_optimizer, "SGD")(*args,**kwargs)
print(sgd)
assert isinstance(sgd,optim.SGD)

from builder.model import Model

test_model = Model("/home/bilal/repos/biloCNN/builder/cfg/darknet19.yaml", debug=True)
print(test_model.model)
x = torch.randn(1,3,224,224)
print(test_model(x).shape)
