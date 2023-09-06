# builder class for building the model

import torch
import torch.nn as nn
import yaml
import os
import sys

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from layers import Layers
from optimizers import Optimizers


class Model(nn.Module):
    def __init__(self, cfg, debug = False):
        super(Model, self).__init__()
        self.cfg = cfg
        self.layers = Layers()
        self.optimizers = Optimizers()
        self.debug = debug
        
        self.build_model()

    def build_model(self):
        model = nn.Sequential()
        with open(self.cfg, 'r') as file:
            try:
                configs = yaml.load(file, Loader=yaml.FullLoader)
                self.model_name = configs['model_name']
                for layer in configs['layers']:
                    args = list()
                    kwargs = dict()
                    if layer['type'] == 'conv2d':
                        kwargs['kernel_size'] = layer['kernel_size']
                        kwargs['in_channels'] = layer['in_channels']
                        kwargs['out_channels'] = layer['out_channels']
                        if 'stride' in layer:
                            kwargs['stride'] = layer['stride']
                        if 'padding' in layer:
                            kwargs['padding'] = layer['padding']
                        if 'bias' in layer:
                            kwargs['bias'] = layer['bias']
                        model.append(getattr(self.layers, layer['type'])(*args, **kwargs))
                        if 'activation' in layer:
                            model.append(getattr(self.layers, layer['activation'])())
                        if 'normalize' in layer:
                            model.append(getattr(self.layers, layer['normalize'])(layer['out_channels']))
                        
                        if 'pooling' in layer:
                            model.append(getattr(self.layers, layer['pooling'][0])(**layer['pooling'][1]))
                    
                    if layer['type'] == 'flatten':
                        model.append(getattr(self.layers, layer['type'])())
                    
                    if layer['type'] == 'linear':
                        kwargs['in_features'] = layer['in_features']
                        kwargs['out_features'] = layer['out_features']
                        if 'bias' in layer:
                            kwargs['bias'] = layer['bias']
                        model.append(getattr(self.layers, layer['type'])(*args, **kwargs))
                        if 'activation' in layer:
                            model.append(getattr(self.layers, layer['activation'])())
                        if 'dropout' in layer:
                            model.append(getattr(self.layers, 'dropout')(layer['dropout']))
                    if self.debug:
                        print(layer['type'])
                        
            except yaml.YAMLError as exc:
                print(exc)
                assert False, "Error in loading yaml file"
                
        self.model = model
    
    def forward(self, x):
        return self.model(x)
    
    def save_weights(self, path):
        torch.save(self.state_dict(), path)
        
    def load_weights(self, path):
        self.load_state_dict(torch.load(path))