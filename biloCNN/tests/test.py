# Test ../model/layers.py

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
print(sys.path[0])

import torch
import torch.nn as nn
import torch.optim as optim # Optimizers

from builder.layers import Layers
from builder.model import Model
from builder.optimizers import Optimizers

import unittest

layer_object = Layers()
optimizer_object = Optimizers()

class TestLayers(unittest.TestCase):
        
    def test_conv2d(self):
        args = list()
        kwargs = dict()
        kwargs['in_channels'] = 3
        kwargs['out_channels'] = 3
        kwargs['kernel_size'] = 3
        
        test_conv = getattr(layer_object, "conv2d")(*args, **kwargs)
        self.assertIsInstance(test_conv, nn.Module, "Conv2d is not an instance of nn.Module")
    
    def test_adaptiveavgpool2d(self):
        args = list()
        kwargs = dict()
        kwargs['output_size'] = (1,1)
        tensor = torch.rand(1, 3, 3, 3)
        test_adaptiveavgpool2d = getattr(layer_object, "adaptiveavgpool2d")(*args, **kwargs)
        self.assertEqual(test_adaptiveavgpool2d(tensor).shape, (1,3,1,1), "AdaptiveAvgPool2d is not working") 
        
    def test_flatten(self):
        args = list()
        kwargs = dict()
        tensor = torch.rand(1, 3, 3, 3)
        test_flatten = getattr(layer_object, "flatten")(*args, **kwargs)
        self.assertEqual(test_flatten(tensor).shape, (1,27), "Flatten is not working")
        
    def test_model(self):
        cfg = "tests/basic.yaml"
        model_object = Model(cfg=cfg)
        tensor = torch.rand(1, 3, 224, 224)
        print(model_object)
        self.assertIsInstance(model_object, nn.Module, "Model is not an instance of nn.Module")
        self.assertEqual(model_object(tensor).shape, (1,10), "Model is not working")
    
    def test_optimizer(self):
        cfg = "tests/basic.yaml"
        model_object = Model(cfg=cfg)
        args = list()
        kwargs = dict()
        kwargs["lr"] = 0.01
        kwargs["momentum"] = 0.9
        kwargs["params"] = model_object.parameters()
        sgd = getattr(optimizer_object, "SGD")(*args, **kwargs)
        print(sgd)
        self.assertIsInstance(sgd, optim.SGD, "SGD is not an instance of optim.SGD")
        

unittest.main()


