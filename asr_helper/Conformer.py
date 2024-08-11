from typing import Tuple, List, Optional, Callable

import torch
from torch import nn

import numpy as np

from Attention import RelPartialLearnableMultiheadAttn, RelMultiheadAttention

import copy

def _lengths_to_padding_mask(lengths: torch.Tensor,
                             max_length: int) -> torch.Tensor:
    batch_size = lengths.shape[0]
    max_length = max_length
    padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
        batch_size, max_length
    ) >= lengths.unsqueeze(1)
    return padding_mask

def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class _MulHeadAttnModule(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.0,
                 relattn: bool = True,
                 bias: bool = True) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.layernorm = nn.LayerNorm(embed_dim)
        if relattn:
            self.multiheadattn = RelPartialLearnableMultiheadAttn(embed_dim,
                                                                  num_heads = num_heads,
                                                                  dropout = dropout,
                                                                  bias = bias)
        else:
            self.multiheadattn = nn.MultiheadAttention(embed_dim,
                                                       num_heads = num_heads,
                                                       dropout = dropout,
                                                       batch_first = True,
                                                       bias = bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.layernorm(x)
        out, _ = self.multiheadattn(x, x, x,
                                    key_padding_mask = key_padding_mask,
                                    attn_mask = attn_mask)
        out = self.dropout(out)
        return out

class _ConvolutionModule(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_channels: int,
                 kernel_size: int,
                 dropout: float = 0,
                 bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.num_channels = num_channels

        self.layernorm = nn.LayerNorm(in_channels)
        self.seq = nn.Sequential(
            nn.Conv1d(in_channels,
                      num_channels*2,
                      kernel_size = 1,
                      stride = 1,
                      padding = 0,
                      bias = bias),
            nn.GLU(dim = 1),
            nn.Conv1d(num_channels, # because of glu
                      num_channels,
                      kernel_size = kernel_size,
                      stride = 1,
                      padding = (kernel_size - 1)//2,
                      groups = num_channels,
                      bias = bias),
            nn.BatchNorm1d(num_channels),
            nn.SiLU(inplace = True),
            nn.Conv1d(num_channels,
                      in_channels,
                      kernel_size = 1,
                      stride = 1,
                      padding = 0,
                      bias = bias),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        x: (N, T, F)
        """
        x = self.layernorm(x).transpose(1, 2).contiguous()
        x = self.seq(x).transpose(1, 2).contiguous()
        return x

class _FeedForwardModule(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 dropout: float = 0.0,
                 ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim,
                      hidden_dim),
            nn.SiLU(inplace = True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,
                      input_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        x: (N, T, H)
        """
        x = self.seq(x)
        return x

class ConformerEncoderLayer(nn.Module):
    def __init__(self,
                 input_dim,
                 ffn_dim: int,
                 num_heads: int,
                 kernel_size: int,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.input_dim = input_dim

        self.ffn1 = _FeedForwardModule(input_dim,
                                       ffn_dim,
                                       dropout)
        self.selfattn = _MulHeadAttnModule(input_dim,
                                           num_heads,
                                          dropout)
        self.conv = _ConvolutionModule(input_dim,
                                       input_dim,
                                       kernel_size = kernel_size,
                                       dropout = dropout)
        self.ffn2 = _FeedForwardModule(input_dim,
                                       ffn_dim,
                                       dropout)
        self.layernorm = nn.LayerNorm(input_dim)

    def forward(self,
                x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (N, T, H)
        """
        x = x + 0.5*self.ffn1(x)
        x = x + 0.5*self.selfattn(x,
                                  key_padding_mask = key_padding_mask,
                                  attn_mask = attn_mask)
        x = x + self.conv(x)
        return self.layernorm(x + 0.5*self.ffn2(x))

class ConformerEncoder(nn.Module):
    def __init__(self,
                 encoder_layer: "ConformerEncoderLayer",
                 num_layers: int
                 ) -> None:
        super().__init__()
        self.num_layers = num_layers

        self.conformer_layers = _get_clones(encoder_layer, num_layers)

    def forward(self,
                input: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (N, T, H)
        """
        max_length = input.shape[1]
        key_padding_mask = _lengths_to_padding_mask(lengths, max_length) if lengths is not None else None
        x = input
        for layer in self.conformer_layers:
            x = layer(x, key_padding_mask = key_padding_mask)
        return x, lengths, key_padding_mask

if __name__ == '__main__':
    a = torch.rand(2, 1000, 80)
    conformerencoderlayer = ConformerEncoderLayer(80,
                                             256,
                                             num_heads = 4,
                                             kernel_size = 33,
                                             dropout = 0.1)
    conformerencoder = ConformerEncoder(conformerencoderlayer,
                                        16)
    memory, key_padding_mask = conformerencoder(a, lengths = torch.tensor([900, 800]))
    print(memory.shape, key_padding_mask.shape)


