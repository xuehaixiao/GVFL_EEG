import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from beartype import beartype
from beartype.typing import Optional, Union, Tuple

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from iTransformer.attend import Attend
from iTransformer.revin import RevIN
from mamba_ssm import Mamba

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t, *args, **kwargs):
    return t

def cast_tuple(t):
    return (t,) if not isinstance(t, tuple) else t

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        heads = 4,
        dropout = 0.,
        flash = True
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias = False),
            Rearrange('b n (qkv h d) -> qkv b h n d', qkv = 3, h = heads)
        )

        self.to_v_gates = nn.Sequential(
            nn.Linear(dim, heads, bias = False),
            nn.Sigmoid(),
            Rearrange('b n h -> b h n 1', h = heads)
        )

        self.attend = Attend(flash = flash, dropout = dropout)

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(dim_inner, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        q, k, v = self.to_qkv(x)

        out = self.attend(q, k, v)

        out = out * self.to_v_gates(x)
        return self.to_out(out)

# feedforward

class GEGLU(Module):
    def forward(self, x):
        x, gate = rearrange(x, '... (r d) -> r ... d', r = 2)
        return x * F.gelu(gate)

def FeedForward(dim, mult = 4, dropout = 0.):
    dim_inner = int(dim * mult * 2 / 3)
    return nn.Sequential(
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_inner, dim)
    )
class Mamba_Layer(nn.Module):
    def __init__(self, d_model):
        super(Mamba_Layer, self).__init__()
        self.mamba = Mamba(d_model, d_state=16, d_conv=4)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):

        x = x + self.mamba(x)
        x = self.norm(x).permute(0, 2, 1)

        return x

# main class

class iTransformer(Module):
    @beartype
    def __init__(
        self,
        *,
        num_variates: int,
        lookback_len: int,
        depth: int,
        dim: int,
        num_tokens_per_variate = 1,
        dim_head = 32,
        heads = 4,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        num_mem_tokens = 4,
        use_reversible_instance_norm=False,
        reversible_instance_norm_affine=False,
        flash_attn = True
    ):
        super().__init__()
        self.num_variates = num_variates
        self.lookback_len = lookback_len

        self.mem_tokens = nn.Parameter(torch.randn(num_mem_tokens, dim)) if num_mem_tokens > 0 else None

        self.num_tokens_per_variate = num_tokens_per_variate
        self.reversible_instance_norm = RevIN(num_variates,affine=reversible_instance_norm_affine) if use_reversible_instance_norm else None

        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, flash = flash_attn),
                nn.LayerNorm(dim),
                FeedForward(dim, mult = ff_mult, dropout = ff_dropout),
                nn.LayerNorm(dim)
            ]))
        self.mlp_in = nn.Sequential(
            nn.Linear(lookback_len, dim * num_tokens_per_variate),
            Rearrange('b v (n d) -> b (v n) d', n=num_tokens_per_variate),
            nn.LayerNorm(dim)
        )
        self.mamba_pre = Mamba_Layer(d_model=dim)

    @beartype
    def forward(
        self,
        x: Tensor,
        targets: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ):
        """
        einstein notation

        b - batch
        n - time
        v - variate
        t - num tokens per variate
        """
        t = self.num_tokens_per_variate
        has_mem = exists(self.mem_tokens)
        assert x.shape[1:] == (self.lookback_len, self.num_variates)

        # the crux of the paper is basically treating variates as the spatial dimension in attention
        # there is a lot of opportunity to improve on this, if the paper is successfully replicated

        x = rearrange(x, 'b n v -> b v n')

        # 在时间序列数据中，不同变量可能具有不同的量级和统计特性。
        # 实例归一化通过对每个样本（实例）进行归一化，能够消除这些差异，使模型更容易学习到有效的特征。
        # 然而，归一化也可能丢失一些有用的信息。
        # 因此，RevIN 提供了一种可逆的归一化方法，使得在必要时可以恢复原始数据。
        if exists(self.reversible_instance_norm):
            x, reverse_fn = self.reversible_instance_norm(x)
        x = self.mlp_in(x)
        # #x = self.mamba_pre(x).permute(0, 2, 1)
        # # memory tokens
        # # 扩展记忆token维度，将记忆token在第二个维度上合并x=(bs,len+num_mem,channel),上下文信息
        if has_mem:
            m = repeat(self.mem_tokens, 'm d -> b m d', b=x.shape[0])
            x, mem_ps = pack([m, x], 'b * d')

        # attention and feedforward layers

        for attn, attn_post_norm, ff, ff_post_norm in self.layers:
            x = attn(x) + x
            x = attn_post_norm(x)
            x = ff(x) + x
            x = ff_post_norm(x)

        # splice out memory tokens
        # mem_ps是记忆token的位置标记，unpack是解包函数，使x与token分离
        if has_mem:
            _, x = unpack(x, mem_ps, 'b * d')

        # reversible instance normalization, if needed

        if exists(self.reversible_instance_norm):
            x = rearrange(x, 'b (n t) d -> t b n d', t=t)
            x = reverse_fn(x)
            x = rearrange(x, 't b n d -> b (n t) d', t=t)

        return x
