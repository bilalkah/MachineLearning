import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import glob
import numpy as np
from open3d import *
import struct
import pykitti
from utils import prepareIMG

class KittiIMGDataset(Dataset):
    def __init__(self, cfg="/home/bilal/repos/biloCNN/datasets/cfg/kitti_img.yaml"):
        super(KittiIMGDataset, self).__init__()
        self.cfg = cfg
        self.build_dataset()
        
        with open(cfg, "r") as config:
            try:
                dataset_config = yaml.load(config, Loader=yaml.FullLoader)
                self.path = dataset_config["path"]
                self.sequence = dataset_config["sequence"]
                self.data_augmentation = dataset_config["data_augmentation"]
                self.precision = dataset_config["precision"]
            except yaml.YAMLError as exc:
                assert False, "Error in loading config file"
                print(exc)
        
        self.dataset = []
        for seq in self.sequence:
            self.dataset.append(pykitti.odometry(self.path, seq))
        
        self.poses = []
                
        self.load_dataset()
        
    def load_dataset(self):
        for seq in range(len(self.sequence)):
            data = 0
            for pose in self.dataset[k].poses:
                homogenous_matrix = np.zeros((4, 4), dtype=np.float)
                
                for i in range(3):
                    for j in range(4):
                        homogenous_matrix[i][j] = pose[i][j]
                homogenous_matrix[3][3] = 1
                
                self.poses.append([homogenous_matrix,seq,data])
                data += 1
                
    def __len__(self):
        return len(self.poses)
    
    def __getitem__(self, index):
        if index >= len(self.poses) or index <= 0:
            raise IndexError("Index out of range")
        
        seq = self.poses[index][1]
        data = self.poses[index][2]
        
        if data == len(self.dataset[seq])-1:
            index -=1
            seq = self.poses[index][1]
            data = self.poses[index][2]
            
        current_data = cv2.imread(self.dataset[seq].cam2_files[data + 1])
        previous_data = cv2.imread(self.dataset[seq].cam2_files[data])
        
        data = prepareIMG(current_data, previous_data)
        
        current_label = self.poses[index + 1][0]
        previous_label = self.poses[index][0]
        
        data_label = prepareLabel(current_label, previous_label)
        
        return data, data_label
        