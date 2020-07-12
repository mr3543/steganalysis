import os 
from time import gmtime, strftime

class Config:

    def __init__(self):

        # current time 
        time_now = strftime("%Y%m%d-%H%M%S")

        # data root and name of current experiment 
        self.data_root = '/home/mmr/kaggle/data/steganalysis/data'
        self.experiment_name = 'enet-b7-baseline'

        # portion of the data to sample for training and testing
        self.test_fold = 0
        self.train_sample_frac = 1.0
        self.test_sample_frac  = 1.0

        # portion of the data to sample for evaluation after training
        # epoch
        self.train_eval_sample_frac = 0.25
        self.test_eval_sample_frac  = 1.0

        # checkpoint directory 
        self.ckpt_dir = '/home/mmr/kaggle/steganalysis/ckpts/' + \
                self.experiment_name + time_now

        if not os.path.isdir(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)

        # tensorboard writer directory 
        self.writer_dir = '/home/mmr/kaggle/steganalysis/runs/' + \
                self.experiment_name + time_now
        

        # learning rate and momentum for Adam optimizer 
        self.lr = 3e-4
        self.momentum = .9

        # for one cycle schedule
        self.max_lr = 0.1

        # num classes 
        self.num_classes = 4

        # efficient net arch
        self.enet = 'efficientnet-b7'

        # batch size and training workers 
        self.batch_size = 4
        self.num_workers = 4

        # num epochs
        self.num_epochs = 30 

        # mixed precision training
        self.mixed_pre = True




