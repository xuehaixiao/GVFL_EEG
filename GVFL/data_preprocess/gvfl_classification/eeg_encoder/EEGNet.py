import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np
#from utils import *
from loss import ClipLoss
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
##############################################################
# EEG_Project
##############################################################
class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )
##############################################################
# EEGNet classifier
##############################################################
class EEGNet(nn.Module):
    def __init__(self, strategy,num_channels,temporal=512):
        super(EEGNet, self).__init__()
        #possible spatial [128, 96, 64, 32, 16, 8]
        #possible temporal [1024, 512, 440, 256, 200, 128, 100, 50]
        F1 = 8
        F2 = 16
        D = 2
        first_kernel = temporal//2
        first_padding = first_kernel//2
        self.network = nn.Sequential(
            nn.ZeroPad2d((first_padding, first_padding-1, 0, 0)),
            nn.Conv2d(in_channels = 1,
                      out_channels = F1,
                      kernel_size = (1, first_kernel)),
            nn.BatchNorm2d(F1),
            nn.Conv2d(in_channels = F1,
                      out_channels = F1,
                      kernel_size = (num_channels, 1),
                      groups = F1),
            nn.Conv2d(in_channels = F1,
                      out_channels = D*F1,
                      kernel_size = 1),
            nn.BatchNorm2d(D*F1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size = (1, 4)),
            nn.Dropout(),
            nn.ZeroPad2d((8, 7, 0, 0)),
            nn.Conv2d(in_channels = D*F1,
                      out_channels = D*F1,
                      kernel_size = (1, 16),
                      groups = F1),
            nn.Conv2d(in_channels = D*F1,
                      out_channels = F2,
                      kernel_size = 1),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size = (1, 8)),
            nn.Dropout())
        self.fc = nn.Linear(F2*(temporal//32), 40)
        self.proj_eeg = Proj_eeg(embedding_dim=F2*(temporal//32))
        self.classifier1 = nn.Linear(1024, 40)
        self.strategy = strategy

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, x):
        x = x.unsqueeze(0).permute(1, 0, 3, 2)
        x = self.network(x)
        x = x.view(x.size()[0], -1)
        eeg_embedding = self.proj_eeg(x)
        if self.strategy == 'retrieval' or self.strategy == 'pretraining':
            return eeg_embedding
        elif self.strategy == 'classify':
            class_logits = self.classifier1(eeg_embedding)
            return class_logits
        elif self.strategy == 'R+C':
            class_logits = self.classifier1(eeg_embedding)
            return eeg_embedding, class_logits
        else:
            print('Unknown strategy')
