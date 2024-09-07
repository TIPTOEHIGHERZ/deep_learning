import torch
import torch.nn as nn
from typing import Union, Iterable, List
from .PositionalEncode import PositionalEncode


def patch_image(image: torch.Tensor, patch_size):
    result = image.unfold(-2, patch_size[0], patch_size[0]).unfold(-2, patch_size[1], patch_size[1])
    result = result.reshape(*result.shape[:2], -1, patch_size[0], patch_size[1])
    result = torch.transpose(result, 1, 2)
    return result


def patch_flatten(x, token_len, device='cuda'):
    x = x.transpose(-2, -3).transpose(-1, -2)
    x = x.reshape(*x.shape[:2], -1)
    # pad length to fixed length
    if token_len > x.shape[1]:
        x = torch.concatenate([x, torch.zeros(x.shape[0], token_len - x.shape[1], *x.shape[2:],
                                              device=device)], dim=1)
    return x


class PatchEmbedding(nn.Module):
    def __init__(self,
                 channels: int,
                 im_size: list,
                 patch_size: Union[int, list],
                 embedd_dim: int,
                 token_len: Union[int, None],
                 device='cuda'
                 ):
        super().__init__()

        self.im_size = im_size
        self.patch_size = [patch_size, patch_size] if isinstance(patch_size, int) else patch_size
        self.embedd_dim = embedd_dim
        self.device = device
        self.channels = channels

        self.grid_size = [self.im_size[0] // self.patch_size[0],
                          self.im_size[1] // self.patch_size[1]]
        self.token_len = token_len if token_len is not None else self.grid_size[0] * self.grid_size[1]
        self.patch_num = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Linear(self.patch_size[0] * self.patch_size[1] * self.channels, embedd_dim, bias=False)
        # because of the cls head, need to increase 1
        self.pos_encode = PositionalEncode([self.token_len + 1, embedd_dim])
        self.cls_token = nn.Parameter(torch.rand([1, 1, self.embedd_dim], device=self.device))

        self.to(device)
        return

    def forward(self, x):
        batch_size = x.shape[0]
        # patching image into size [batch_size, patch_num, channels, height, width]
        x = patch_image(x, self.patch_size)
        x = patch_flatten(x, self.token_len, device=self.device)
        x = self.proj(x)
        cls_token = self.cls_token.repeat(batch_size, *([1] * (len(x.shape) - 1)))
        x = torch.concatenate([cls_token, x], dim=1)
        x = self.pos_encode(x)

        return x


