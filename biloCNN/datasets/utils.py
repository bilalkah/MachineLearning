import cv2
import glob
import numpy as np
from open3d import *
import struct
import pykitti
import torch

def prepareIMG(current, previous):
    data = np.concatenate((current, previous), axis=2)
    data = cv2.resize(data, (224,224))
    data = np.transpose(data, (2,0,1))
    data = data.astype(np.float32)
    data = data/255.0
    return torch.from_numpy(data)

def prepareLabel(current, previous):
    rotation_prev = previous[:3,:3]
    translation_prev = previous[:3,3]
    
    rotation_current = current[:3,:3]
    translation_current = current[:3,3]
    
    rotation = np.dot(rotation_prev.T, rotation_current)
    translation = np.dot(rotation_prev.T, translation_current) - np.dot(rotation_prev.T, translation_prev)
    
    rotation_angles = cv2.Rodrigues(rotation)[0]
    rotation_angles = rotation_angles.reshape(3)
    
    odometry = np.concatenate((translation, rotation_angles), axis=0)
    return torch.from_numpy(odometry)