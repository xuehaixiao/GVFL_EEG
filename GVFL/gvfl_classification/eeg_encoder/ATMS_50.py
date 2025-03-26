import os

import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = 'offline'
from itertools import combinations

import clip
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce
#from loss import ClipLoss
from loss import ClipLoss
from torch import Tensor
import math
import torch.nn as nn


class YourModule(nn.Module):
    def __init__(self, dim,heads,depth):
        super(YourModule, self).__init__()
        self.layers = nn.ModuleList()

        for _ in range(depth):
            # 使用 nn.MultiheadAttention 作为自注意力机制
            multihead_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads,
                                                   batch_first=False)

            # 创建一个模块列表，包含自注意力、层归一化、前馈网络和层归一化
            self.layers.append(nn.ModuleList([
                multihead_attn,
                nn.LayerNorm(dim),
                FeedForward(dim),
                nn.LayerNorm(dim)
            ]))

    # 实现前向传播方法
    def forward(self, x):
        for layer_block in self.layers:
            attn, ln1, ff, ln2 = layer_block

            # 自注意力机制需要 Q, K, V 作为输入，通常 Q, K, V 是相同的（自注意力）
            query = key = value = x
            attn_output, _ = attn(query, key, value)

            # 残差连接和层归一化
            x = ln1(x + attn_output)

            # 前馈网络
            ff_output = ff(x)

            # 残差连接和层归一化
            x = ln2(x + ff_output)

        return x


# 请确保您有一个 FeedForward 类的实现
class FeedForward(nn.Module):
    def __init__(self, dim, mult=1, dropout=0.2):
        super(FeedForward, self).__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model + 1, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term[:d_model // 2 ])
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])

        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe[:x.size(0), :].unsqueeze(1).repeat(1, x.size(1), 1)
        x = x + pe
        return x


class EEGAttention(nn.Module):
    def __init__(self, channel, d_model, nhead):
        super(EEGAttention, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model=96)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        #self.attention=YourModule(d_model,heads=nhead,depth=1)
        self.channel = channel
        self.d_model = d_model

    def forward(self, src):
        src = src.permute(2, 0, 1)  # Change shape to [time_length, batch_size, channel]
        src = self.pos_encoder(src)
        #output = self.attention(src.permute(2, 1, 0))
        output = self.transformer_encoder(src.permute(2, 1, 0))
        return output.permute(1, 0, 2)  # Change shape back to [batch_size, channel, time_length]


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        # revised from shallownet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (96, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        x = x.unsqueeze(1)
        # print("x", x.shape)
        x = self.tsconv(x)
        # print("tsconv", x.shape)
        #x = self.projection(x)
        # print("projection", x.shape)
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )


class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=3520, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )


class Proj_img(nn.Sequential):
    def __init__(self, embedding_dim=768, proj_dim=768, drop_proj=0.3):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, x):
        return x


class ATMS_50(nn.Module):
    def __init__(self,strategy,num_channels):
        super(ATMS_50, self).__init__()
        self.attention_model = EEGAttention(num_channels, 512, nhead=1)
        # self.subject_wise_linear = nn.ModuleList(
        #     [nn.Linear(sequence_length, sequence_length) for _ in range(num_subjects)])
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()
        self.classifier1 = nn.Linear(1024, 40)
        self.strategy = strategy
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.attention_model(x)
        # print(f'After attention shape: {x.shape}')

        #x = self.subject_wise_linear[0](x)
        # print(f'After subject-specific linear transformation shape: {x.shape}')
        eeg_embedding = self.enc_eeg(x)

        eeg_embedding = self.proj_eeg(eeg_embedding)
        if self.strategy == 'pretraining':
            return eeg_embedding
        elif self.strategy == 'classify':
            class_logits = self.classifier1(eeg_embedding)
            return class_logits
        elif self.strategy == 'R+C':
            class_logits = self.classifier1(eeg_embedding)
            return eeg_embedding, class_logits
        else:
            print('Unknown strategy')