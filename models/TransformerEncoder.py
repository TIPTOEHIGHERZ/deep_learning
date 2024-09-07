import torch
import torch.nn as nn
from .MLP import MLP
from .MultiHeadAttention import MultiHeadAttention


class TransformerEncoder(nn.Module):
    def __init__(self,
                 embedd_dim: int,
                 head_num: int,
                 mlp_dim: int,
                 device='cuda'):
        super().__init__()
        self.layer_norm_0 = nn.LayerNorm(embedd_dim)
        self.multi_head_attention = MultiHeadAttention(embedd_dim, head_num)
        self.layer_norm_1 = nn.LayerNorm(embedd_dim)
        self.mlp = MLP(embedd_dim, mlp_dim, embedd_dim)

        self.to(device)
        return

    def forward(self, x):
        x = self.layer_norm_0(x)
        x = self.multi_head_attention([x, x, x])
        x = self.layer_norm_1(x)
        x = self.mlp(x)

        return x
