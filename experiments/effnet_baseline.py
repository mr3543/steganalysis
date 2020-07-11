import torch
import torch.nn as nn
from torchvision import transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

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
import sys
import os
import pprint

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.loss.loss import LabelSmoothing
from src.data.dataset import IMDataset
from src.utils.utils import *


# create metadata dataset
data_root = '/home/mmr/kaggle/data/steganalysis/data'
dataset = build_dataset(data_root)

# create train/test datasets
test_fold = 0
train_sample_frac = 0.001
test_sample_frace = 0.001


train_data_sample,test_data_sample = train_test_split(
        dataset,test_fold,train_sample_frac,test_sample_frace)

train_transform = get_train_transforms()
test_transform  = get_valid_transforms()

train_im_dataset = IMDataset(train_data_sample,data_root,train_transform)
test_im_dataset  = IMDataset(test_data_sample,data_root,test_transform)

# create model, loss, optimizer
num_classes = 4

model = get_net('efficientnet-b2',num_classes).cuda()
opt = torch.optim.Adam(model.parameters(),lr=3e-4)
loss = LabelSmoothing(num_classes)

batch_size = 8
num_loader_workers = 4

train_loader = torch.utils.data.DataLoader(train_im_dataset,
        batch_size=batch_size,num_workers=num_loader_workers)

test_loader  = torch.utils.data.DataLoader(test_im_dataset,
        batch_size=batch_size,num_workers=num_loader_workers)

# training loop
num_epochs = 25
sample_frac = 0.1
writer = SummaryWriter('runs/fashion_mnist_experiment_1')
ckpt_path = '/home/mmr/kaggle/steganalysis/ckpts'

for epoch in range(num_epochs):

    # train model 
    train(model,epoch,train_loader,loss,opt)
    
    # sample train_dataset for eval
    train_eval_loader = make_sampled_loader(
            train_data_sample,sample_frac,test_transform,
            data_root,batch_size,num_loader_workers)

    train_met = eval_model(model,train_eval_loader)
    test_met  = eval_model(model,test_loader)

    writer.add_scalar('Train/auc',train_met,epoch)
    writer.add_scalar('Test/auc',test_met,epoch)

    ckpt_file = ckpt_path + '/model_{}.pt'.format(epoch)
    torch.save({
        'epoch':epoch,
        'model_state_dict':model.state_dict(),
        'opt_state_dict':opt.state_dict()
        },ckpt_file)

