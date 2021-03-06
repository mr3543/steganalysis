import torch
import torch.nn as nn
from torchvision import transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from apex import amp

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
from src.config import Config 

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

C = Config()

# create metadata dataset
dataset = build_dataset(C.data_root)

# create train/test datasets

train_data_sample,test_data_sample = train_test_split(
        dataset,C.test_fold,C.train_sample_frac,C.test_sample_frac)

train_transform = get_train_transforms()
test_transform  = get_valid_transforms()

train_im_dataset = IMDataset(train_data_sample,C.data_root,train_transform)
test_im_dataset  = IMDataset(test_data_sample,C.data_root,test_transform)

# create model, loss, optimizer
model = get_net(C.enet,C.num_classes).cuda()
opt = torch.optim.Adam(model.parameters(),lr=C.lr)
loss = LabelSmoothing(C.num_classes)

model,opt = amp.initialize(
        model,opt,opt_level="O2",
        keep_batchnorm_fp32=True,loss_scale='dynamic')


train_loader = torch.utils.data.DataLoader(train_im_dataset,
        batch_size=C.batch_size,num_workers=C.num_workers)

test_loader  = torch.utils.data.DataLoader(test_im_dataset,
        batch_size=C.batch_size,num_workers=C.num_workers)

# training loop
writer = SummaryWriter(C.writer_dir)

for epoch in range(C.num_epochs):

    t0 = time.time()
    # train model 
    train(model,epoch,train_loader,loss,opt,
            writer=writer,sched=None,mixed_pre=C.mixed_pre)

    
    # sample train_dataset for eval
    train_eval_loader = make_sampled_loader(
            train_data_sample,C.train_eval_sample_frac,
            test_transform,
            C.data_root,C.batch_size,
            C.num_workers)


    train_met = eval_model(model,train_eval_loader)
    
    #train_met = eval_model(model,train_loader)
    test_met  = eval_model(model,test_loader)
    print('train met: ',train_met)
    print('test met: ',test_met)

    writer.add_scalar('/train/auc',train_met,epoch)
    writer.add_scalar('/test/auc',test_met,epoch)

    ckpt_file = C.ckpt_dir + '/model_{}.pt'.format(epoch)
    torch.save({
        'epoch':epoch,
        'model_state_dict':model.state_dict(),
        'opt_state_dict':opt.state_dict()
        },ckpt_file)

    t1 = time.time()
    print('epoch {}, time: {:0.2f} mins'.format(epoch,(t1-t0)/60))

