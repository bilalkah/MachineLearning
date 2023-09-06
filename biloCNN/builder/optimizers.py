# Class for optimizers

import torch
import torch.nn as nn
import torch.optim as optim # Optimizers


class Optimizers:
    class SGD(optim.SGD):
        def __init__(self, *args, **kwargs):
            super(Optimizers.SGD, self).__init__(*args, **kwargs)
            
        
            
    class Adam(optim.Adam):
        def __init__(self, *args, **kwargs):
            super(Optimizers.Adam, self).__init__(*args, **kwargs)
            
        
    def __getattribute__(self, name):
        if name == "SGD":
            return Optimizers.SGD
        elif name == "Adam":
            return Optimizers.Adam
        else:
            return super(Optimizers, self).__getattribute__(name)
    