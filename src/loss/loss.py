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

class LabelSmoothing(nn.Module):
    def __init__(self, num_classes, smoothing = 0.05):
        super(LabelSmoothing, self).__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, x, target):
        
        target = F.one_hot(target,num_classes=self.num_classes)
        logits = F.log_softmax(x,dim=-1)
        nll_loss = -logits * target
        nll_loss = nll_loss.sum(-1)
        
        smooth_loss = -logits.mean(dim=-1)
        loss= (1-self.smoothing) * nll_loss + self.smoothing * smooth_loss
        
        return loss.mean()


