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

from time import gmtime, strftime
import random
import glob 
import gc
import pdb
import sys
import os
import pprint
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.loss.loss import LabelSmoothing
from src.data.dataset import IMDataset
from src.utils.utils import *

SEED = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)

# create metadata dataset
experiment_name = 'enet-b2-baseline'
data_root = '/home/mmr/kaggle/data/steganalysis/data'
dataset = build_dataset(data_root)

# create train/test datasets
test_fold = 0
train_sample_frac = 0.04
test_sample_frace = 0.04


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

model,opt = amp.initialize(
        model,opt,opt_level="02",
        keep_batchnorm_fp32=True,loss_scale='dynamic')

batch_size = 8
num_loader_workers = 4

train_loader = torch.utils.data.DataLoader(train_im_dataset,
        batch_size=batch_size,num_workers=num_loader_workers)

test_loader  = torch.utils.data.DataLoader(test_im_dataset,
        batch_size=batch_size,num_workers=num_loader_workers)

# training loop
num_epochs = 15
sample_frac = 0.1
writer = SummaryWriter('runs/steganalysis' + '/' + 
        experiment_name + '/' + strftime("%Y%m%d-%H%M%S"))

ckpt_path = '/home/mmr/kaggle/steganalysis/ckpts'

for epoch in range(num_epochs):

    t0 = time.time()
    # train model 
    train(model,epoch,train_loader,loss,opt,
            writer=writer,sched=None,mixed_pre=True)

    
    """
    # sample train_dataset for eval
    train_eval_loader = make_sampled_loader(
            train_data_sample,sample_frac,test_transform,
            data_root,batch_size,num_loader_workers)


    train_met = eval_model(model,train_eval_loader)
    """
    train_met = eval_model(model,train_loader)
    test_met  = eval_model(model,test_loader)
    print('train met: ',train_met)
    print('test met: ',test_met)

    writer.add_scalar('/train/auc',train_met,epoch)
    writer.add_scalar('/test/auc',test_met,epoch)

    ckpt_file = ckpt_path + '/model_{}.pt'.format(epoch)
    torch.save({
        'epoch':epoch,
        'model_state_dict':model.state_dict(),
        'opt_state_dict':opt.state_dict()
        },ckpt_file)

    t1 = time.time()
    print('epoch {}, time: {:0.2f} mins'.format(epoch,(t1-t0)/60))

