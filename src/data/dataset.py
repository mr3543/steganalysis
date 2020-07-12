import torch
import torch.nn as nn
from torchvision import transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch.nn.functional as F

import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
from sklearn import metrics
from efficientnet_pytorch import EfficientNet

import random
import glob 
import gc
import pdb

class IMDataset(torch.utils.data.Dataset):

    def __init__(self, data_frame,data_root,
            transform=None,binary=False):
        super().__init__()
        self.df = data_frame
        self.data_root = data_root
        self.transform = transform
        self.binary = binary

    def __getitem__(self, index: int):
        kind,image_name,label = self.df['kind'][index], \
                                self.df['image_name'][index], \
                                self.df['label'][index]
        
        if self.binary:
            label = int(label >= 1)
                
        image = cv2.imread(f'{self.data_root}/{kind}/{image_name}', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        if self.transform:
            sample = {'image': image}
            sample = self.transform(**sample)
            image = sample['image']
            
        
        image /= 255
        
        return image, label

    def __len__(self) -> int:
        return len(self.df)
