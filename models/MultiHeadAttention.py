import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 embedd_dim: int,
                 head_num: int,
                 device='cuda'):
        super().__init__()
        # 必须能被整除
        assert embedd_dim % head_num == 0
        self.Wq = nn.Linear(embedd_dim, embedd_dim, bias=False)
        self.Wk = nn.Linear(embedd_dim, embedd_dim, bias=False)
        self.Wv = nn.Linear(embedd_dim, embedd_dim, bias=False)
        self.Wo = nn.Linear(embedd_dim, embedd_dim, bias=False)
        self.d_kq = embedd_dim // head_num

        self.to(device)
        return

    @staticmethod
    def sep_mat(mat: torch.Tensor, slice_size: int):
        return mat.transpose(-1, -2).unfold(-2, slice_size, slice_size)

    @staticmethod
    def merge_mat(mat: torch.Tensor):
        return mat.transpose(-1, -2).reshape(mat.shape[0], -1, mat.shape[-2]).transpose(-1, -2)

    def forward(self, x):
        q, k, v = x
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        q = self.sep_mat(q, self.d_kq)
        k = self.sep_mat(k, self.d_kq)
        v = self.sep_mat(v, self.d_kq)

        heads = self.attention(q, k, v)
        merged_heads = self.merge_mat(heads)
        result = self.Wo(merged_heads)

        return result

    @staticmethod
    def attention(q, k, v):
        sim_mat = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(k.shape[-1]), dim=1)
        result = torch.matmul(sim_mat, v)
        return result
