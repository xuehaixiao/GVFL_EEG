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
#############################################################################
# EEG-ChannelNet classifier
#############################################################################

class EEGChannelnet(nn.Module):

    def __init__(self, strategy,num_channels,temporal=512):
        super(EEGChannelnet, self).__init__()
        self.temporal_layers = []
        self.temporal_layers.append(nn.Sequential(nn.Conv2d(in_channels = 1,
                                    out_channels = 10,
                                    kernel_size = (1, 33),
                                    stride = (1, 2),
                                    dilation = (1, 1),
                                    padding = (0, 16)),
                                    nn.BatchNorm2d(10),
                                    nn.ReLU()))
        self.temporal_layers.append(nn.Sequential(nn.Conv2d(in_channels = 1,
                                    out_channels = 10,
                                    kernel_size = (1, 33),
                                    stride = (1, 2),
                                    dilation = (1, 2),
                                    padding = (0, 32)),
                                    nn.BatchNorm2d(10),
                                    nn.ReLU()))
        self.temporal_layers.append(nn.Sequential(nn.Conv2d(in_channels = 1,
                                    out_channels = 10,
                                    kernel_size = (1, 33),
                                    stride = (1, 2),
                                    dilation = (1, 4),
                                    padding = (0, 64)),
                                    nn.BatchNorm2d(10),
                                    nn.ReLU()))
        self.temporal_layers.append(nn.Sequential(nn.Conv2d(in_channels = 1,
                                    out_channels = 10,
                                    kernel_size = (1, 33),
                                    stride = (1, 2),
                                    dilation = (1, 8),
                                    padding = (0, 128)),
                                    nn.BatchNorm2d(10),
                                    nn.ReLU()))
        self.temporal_layers.append(nn.Sequential(nn.Conv2d(in_channels = 1,
                                    out_channels = 10,
                                    kernel_size = (1, 33),
                                    stride = (1, 2),
                                    dilation = (1, 16),
                                    padding = (0, 256)),
                                    nn.BatchNorm2d(10),
                                    nn.ReLU()))
        self.spatial_layers = []
        self.spatial_layers.append(nn.Sequential(nn.Conv2d(in_channels = 50,
                                   out_channels = 50,
                                   kernel_size = (128, 1),
                                   stride = (2, 1),
                                   padding = (63, 0)),
                                   nn.BatchNorm2d(50),
                                   nn.ReLU()))
        self.spatial_layers.append(nn.Sequential(nn.Conv2d(in_channels = 50,
                                   out_channels = 50,
                                   kernel_size = (64, 1),
                                   stride = (2, 1),
                                   padding = (31, 0)),
                                   nn.BatchNorm2d(50),
                                   nn.ReLU()))
        self.spatial_layers.append(nn.Sequential(nn.Conv2d(in_channels = 50,
                                   out_channels = 50,
                                   kernel_size = (32, 1),
                                   stride = (2, 1),
                                   padding = (15, 0)),
                                   nn.BatchNorm2d(50),
                                   nn.ReLU()))
        self.spatial_layers.append(nn.Sequential(nn.Conv2d(in_channels = 50,
                                   out_channels = 50,
                                   kernel_size = (16, 1),
                                   stride = (2, 1),
                                   padding = (7, 0)),
                                   nn.BatchNorm2d(50),
                                   nn.ReLU()))
        self.residual_layers = []
        self.residual_layers.append(nn.Sequential(nn.Conv2d(in_channels = 200,
                                    out_channels = 200,
                                    kernel_size = 3,
                                    stride = 2,
                                    padding = 1),
                                    nn.BatchNorm2d(200),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 200,
                                    out_channels = 200,
                                    kernel_size = 3,
                                    stride = 1,
                                    padding = 1),
                                    nn.BatchNorm2d(200)))
        self.residual_layers.append(nn.Sequential(nn.Conv2d(in_channels = 200,
                                    out_channels = 200,
                                    kernel_size = 3,
                                    stride = 2,
                                    padding = 1),
                                    nn.BatchNorm2d(200),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 200,
                                    out_channels = 200,
                                    kernel_size = 3,
                                    stride = 1,
                                    padding = 1),
                                    nn.BatchNorm2d(200)))
        self.residual_layers.append(nn.Sequential(nn.Conv2d(in_channels = 200,
                                    out_channels = 200,
                                    kernel_size = 3,
                                    stride = 2,
                                    padding = 1),
                                    nn.BatchNorm2d(200),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 200,
                                    out_channels = 200,
                                    kernel_size = 3,
                                    stride = 1,
                                    padding = 1),
                                    nn.BatchNorm2d(200)))
        self.residual_layers.append(nn.Sequential(nn.Conv2d(in_channels = 200,
                                    out_channels = 200,
                                    kernel_size = 3,
                                    stride = 2,
                                    padding = 1),
                                    nn.BatchNorm2d(200),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 200,
                                    out_channels = 200,
                                    kernel_size = 3,
                                    stride = 1,
                                    padding = 1),
                                    nn.BatchNorm2d(200)))
        self.shortcuts = []
        self.shortcuts.append(nn.Sequential(nn.Conv2d(in_channels = 200,
                              out_channels = 200,
                              kernel_size = 1,
                              stride = 2),
                              nn.BatchNorm2d(200)))
        self.shortcuts.append(nn.Sequential(nn.Conv2d(in_channels = 200,
                              out_channels = 200,
                              kernel_size = 1,
                              stride = 2),
                              nn.BatchNorm2d(200)))
        self.shortcuts.append(nn.Sequential(nn.Conv2d(in_channels = 200,
                              out_channels = 200,
                              kernel_size = 1,
                              stride = 2),
                              nn.BatchNorm2d(200)))
        self.shortcuts.append(nn.Sequential(nn.Conv2d(in_channels = 200,
                              out_channels = 200,
                              kernel_size = 1,
                              stride = 2),
                              nn.BatchNorm2d(200)))
        spatial_kernel = 3
        temporal_kernel = 3

        self.final_conv = nn.Conv2d(in_channels = 200,
                                    out_channels = 50,
                                    kernel_size = (spatial_kernel,
                                                   temporal_kernel),
                                    stride = 1,
                                    dilation = 1,
                                    padding = 0)
        spatial_sizes = [128, 96, 64, 32, 16, 8]
        spatial_outs = [2, 1, 1, 1, 1, 1]
        temporal_sizes = [1024, 512, 440, 256, 200, 128, 100, 50]
        temporal_outs = [30, 14, 12, 6, 5, 2, 2, 1]
        inp_size = (50*
                    spatial_outs[spatial_sizes.index(num_channels)]*
                    temporal_outs[temporal_sizes.index(temporal)])
        self.fc1 = nn.Linear(inp_size, 1024)
        self.fc2 = nn.Linear(1024, 40)
        self.proj_eeg = Proj_eeg(embedding_dim=1024)
        self.classifier1 = nn.Linear(1024, 40)
        self.strategy = strategy
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, x):
        x = x.unsqueeze(0).permute(1, 0, 3, 2)
        #x = x.permute(0,2,1)
        y = []
        for i in range(5):
            y.append(self.temporal_layers[i](x))
        x = torch.cat(y, 1)

        y=[]
        for i in range(4):
            y.append(self. spatial_layers[i](x))
        x = torch.cat(y, 1)
        for i in range(4):
            x = F.relu(self.shortcuts[i](x)+self.residual_layers[i](x))
        x = self.final_conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
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
    def cuda(self, gpuIndex=0):
        for i in range(len(self.temporal_layers)):
            self.temporal_layers[i] = self.temporal_layers[i].cuda(gpuIndex)
        for i in range(len(self.spatial_layers)):
            self.spatial_layers[i] = self.spatial_layers[i].cuda(gpuIndex)
        for i in range(len(self.residual_layers)):
            self.residual_layers[i] = self.residual_layers[i].cuda(gpuIndex)
        for i in range(len(self.shortcuts)):
            self.shortcuts[i] = self.shortcuts[i].cuda(gpuIndex)
        self.final_conv = self.final_conv.cuda(gpuIndex)
        self.fc1 = self.fc1.cuda(gpuIndex)
        self.fc2 = self.fc2.cuda(gpuIndex)
        self.proj_eeg = self.proj_eeg.cuda(gpuIndex)
        return self