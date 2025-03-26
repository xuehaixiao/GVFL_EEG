import torch.nn as nn
from einops.layers.torch import Rearrange
from loss import ClipLoss
from torch import Tensor
#from utils import *
from iTransformer.iTransformer import iTransformer
from mamba_ssm import Mamba
import numpy as np
import torch
import math


class LearnablePositionalEmbeddings(nn.Module):

    def __init__(self, embedding_dim, max_seq_len):
        super(LearnablePositionalEmbeddings, self).__init__()

        self.positional_embeddings = nn.Embedding(max_seq_len, embedding_dim)

        nn.init.normal_(self.positional_embeddings.weight, mean=0.0, std=embedding_dim ** -0.5)

    def forward(self, x, seq_len=None):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)

        positional_embeddings = self.positional_embeddings(positions)



        x = x + positional_embeddings

        return x
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, c_out):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=c_out,
                                   kernel_size=3, padding=1, padding_mode='circular', bias=False)
    def forward(self, x):
        x = self.tokenConv(x)
        return x.permute(0, 2, 1)

class Mamba_Layer(nn.Module):
    def __init__(self, d_model):
        super(Mamba_Layer, self).__init__()
        self.mamba = Mamba(d_model, d_state=16, d_conv=4)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):

        x = x + self.mamba(x)
        x = self.norm(x)#.permute(0, 2, 1)

        return x
class EEGEmbedding(nn.Module):
    def __init__(self, num_channels,emb_size=40):
        super().__init__()
        # revised from shallownet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (num_channels, 1), (1, 1)),
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
class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
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
        #x = x.permute(2,0,1)
        pe = self.pe[:x.size(0), :].unsqueeze(1).repeat(1, x.size(1), 1)
        x = x + pe
        #x = x.permute(1, 2, 0)
        return x
# class EEGAttention(nn.Module):
#     def __init__(self, channel, d_model, nhead):
#         super(EEGAttention, self).__init__()
#         self.pos_encoder = PositionalEncoding(d_model)
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
#         self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
#         self.channel = channel
#         self.d_model = d_model
#
#     def forward(self, src):
#         src = src.permute(2, 0, 1)  # Change shape to [time_length, batch_size, channel]
#         src = self.pos_encoder(src)
#         output = self.transformer_encoder(src)
#         return output.permute(1, 2, 0)  # Change shape back to [batch_size, channel, time_length]
class Enc_eeg(nn.Sequential):
    def __init__(self, num_channels, **kwargs):
        super().__init__(
            #TokenEmbedding(c_in=num_channels, c_out=num_channels),
            # EEGAttention(num_channels, num_channels, nhead=1),
            #PositionalEncoding(num_channels),
            #LearnablePositionalEmbeddings(embedding_dim=96, max_seq_len=5000),
            Mamba_Layer(d_model=num_channels),
            iTransformer(
                num_variates = num_channels,
                lookback_len = 512,
                dim = 256,
                depth = 6,
                heads = 8,
                dim_head = 64,
                num_tokens_per_variate = 1,
                use_reversible_instance_norm = True),

            EEGEmbedding(num_channels),
            FlattenHead()
        )


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
        #return super().forward(x)
        return x


class Model(nn.Module):
    def __init__(self, strategy,num_channels):
        super(Model, self).__init__()

        self.enc_eeg = Enc_eeg(num_channels)

        self.proj_eeg = Proj_eeg(embedding_dim=1480)
        self.classifier1 = nn.Linear(768, 40)
        #self.classifier2 = nn.Linear(1480, 40)
        self.strategy = strategy
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, x):
        #x = x.permute(0, 2, 1)
        #print(x.size)
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









