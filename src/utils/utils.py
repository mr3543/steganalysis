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
import sys

from src.data.dataset import IMDataset

def build_dataset(data_root):
    """
    data_root: path to dataset root

    returns: pandas data frame with image path, label and fold 
    """
    dataset = []

    for label, kind in enumerate(['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']):
        for path in glob.glob(data_root + '/Cover/*.jpg'):
            dataset.append({
                'kind': kind,
                'image_name': path.split('/')[-1],
                'label': label
            })

    random.shuffle(dataset)
    dataset = pd.DataFrame(dataset)

    gkf = GroupKFold(n_splits=5)

    dataset.loc[:, 'fold'] = 0
    for fold_number, (train_index, val_index) in \
        enumerate(gkf.split(X=dataset.index, y=dataset['label'], groups=dataset['image_name'])):
        dataset.loc[dataset.iloc[val_index].index, 'fold'] = fold_number
    
    return dataset


def make_sampled_loader(dataset,sample_frac,transforms,data_root,
                        batch_size,num_workers):
    """
    dataset: pandas dataframe 
    sample_frac: fraction to sample 
    transforms: torchvision.transform obj
    data_root: root of data 
    batch_size: batch size
    num_workers: num workers for DataLoader 

    samples from dataset according to the sample frac
    creates in IMDataset with the sampled dataframe
    creates and returns a DataLoader from the IMDataset
    
    """
    eval_dataset = dataset.sample(frac=sample_frac)\
                          .reset_index(drop=True)

    eval_im_dataset = IMDataset(eval_dataset,data_root,transforms)
    eval_im_loader  = torch.utils.data.DataLoader(
            eval_im_dataset,batch_size=batch_size,
            num_workers=num_workers)

    return eval_im_loader 


def train_test_split(dataset,test_fold,train_sample=1.0,test_sample=1.0):
    """
    splits the dataset into training and test samples. subsamples
    each split according to train_sample and test_sample
    """

    # create the train and test datasets 
    train_dataset = dataset[dataset['fold'] != test_fold].reset_index(drop=True)
    test_dataset  = dataset[dataset['fold'] == test_fold].reset_index(drop=True)

    # sample the dataset if necessary 
    if train_sample < 1:
        train_dataset = train_dataset.sample(frac=train_sample)\
                                     .reset_index(drop=True)
    if test_sample < 1:
        test_dataset = test_dataset.sample(frac=test_sample)\
                                   .reset_index(drop=True)

    print('selecting {} samples for training'.format(len(train_dataset)))
    print('selecting {} samples for testing'.format(len(test_dataset)))

    return train_dataset,test_dataset 


def get_train_transforms():
    return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

def get_valid_transforms():
    return A.Compose([
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)


def alaska_weighted_auc(y_true, y_valid):
    """
    y_true: ground truth labels, np array of 1's and 0's
    y_valid: array of scores, floats btw 0 and 1

    https://www.kaggle.com/anokas/weighted-auc-metric-updated
    """
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)

    # size of subsets
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])

    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.
    normalization = np.dot(areas, weights)

    competition_metric = 0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (tpr < y_max)
        if mask.sum() == 0:
            continue

        # pdb.set_trace()

        x_padding = np.linspace(fpr[mask][-1], 1, 100)

        x = np.concatenate([fpr[mask], x_padding])
        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
        y = y - y_min  # normalize such that curve starts at y=0
        score = metrics.auc(x, y)
        submetric = score * weight
        best_subscore = (y_max - y_min) * weight
        competition_metric += submetric

    return competition_metric / normalization


def collapse(preds,targets):
    """
    preds: model outputs from last layer, shape [B,4] 
    targets: ints btw 0 and 3 

    targets get transformed to 0 or 1, preds go thru
    softmax. we want the score to be the probability 
    the target is of class 1. take 1 - probability 
    target not class 1. 

    """
    # preds [B,4]
    # targets [B]
    
    det_targets = targets.clamp(min=0,max=1)
    det_preds   = 1 - F.softmax(preds,dim=-1)[:,0]

    return det_preds,det_targets

def eval_model(model,eval_loader):
    """
    evaluate a model and get the competition metric as output
    this call collapse so it only works for models that predict
    4 classes. 
    """
    model.eval()
    eval_scores = []
    eval_targets = []
    with torch.no_grad():
        for data,targets in eval_loader:
            data,targets = data.cuda(),targets.cuda()
            preds = model(data)
            scores,det_targets = collapse(preds,targets)
            eval_scores.append(scores)
            eval_targets.append(det_targets)
        
        scores_ten = torch.cat(eval_scores)
        targets_ten = torch.cat(eval_targets)
        
        met = alaska_weighted_auc(targets_ten.cpu().numpy()
            ,scores_ten.cpu().numpy())
    
    if met == None:
        print('return none from auc function')
        return .5
    return met

def train(model,epoch,train_loader,loss,opt,
        writer = None,sched=None,mixed_pre=False):
    """
    trains a model

    model: nn.module
    epoch: the current epoch, used for printing 
    train_loader: training dataloader 
    loss: nn.module 
    opt: optimizer 
    """
    model.train()
    for i,(data,targets) in enumerate(train_loader):
        data,targets = data.cuda(), targets.cuda()
        batch_size = data.size()[0]
        opt.zero_grad()
        output = model(data)
        batch_loss = loss(output,targets)
        if mixed_pre:
            with amp.scale_loss(batch_loss,opt) as scaled_loss:
                scaled_loss.backward()
        else:
            batch_loss.backward()

        opt.step()
        if sched: 
            sched.step()
        if i % 100 == 99:
            print('done: {}/{}'.format(i,len(train_loader)))
            if writer:
                print('logging to tb')
                writer.write(i,batch_loss.item())
    print('done epoch: {}'.format(epoch))

def get_net(net_type,pred_targets):

    """
    net_type: the name of the efficient net, i.e. effcientnet-b2
    pred_targets: the number of units in the last layer 

    gets a pretrained efficient net
    """
    if net_type == 'efficientnet-b2': out_layers=1408
    elif net_type == 'efficientnet-b7': out_layers = 2560
    net = EfficientNet.from_pretrained(net_type)
    net._fc = nn.Linear(in_features=out_layers, out_features=pred_targets, bias=True)
    return net
