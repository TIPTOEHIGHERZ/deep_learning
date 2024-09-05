import torch
import torch.nn as nn
from typing import Union, Iterable, List


def patch_image(image: torch.Tensor, patch_size):
    result = image.unfold(-2, patch_size[0], patch_size[0]).unfold(-2, patch_size[1], patch_size[1])
    result = result.reshape(*result.shape[:2], -1, patch_size[0], patch_size[1])
    result = torch.transpose(result, 1, 2)
    return result


def patch_flatten(x, embedd_dim):
    x = x.transpose(-2, -3).transpose(-1, -2)
    x = x.reshape(*x.shape[:2], -1)
    x = torch.concatenate([x, torch.zeros(x.shape[0], embedd_dim - x.shape[1], *x.shape[2:])], dim=1)
    return x


class PatchEmbedding(nn.Module):
    def __init__(self,
                 im_size: List[int, int],
                 batch_size: int,
                 patch_size: Union[int, Iterable[int, int]],
                 embedd_dim: int,
                 device='cuda'
                 ):
        super().__init__()

        self.im_size = im_size
        self.batch_size = batch_size
        self.patch_size = [patch_size, patch_size] if isinstance(patch_size, int) else patch_size
        self.embedd_dim = embedd_dim
        self.device = device

        self.patched_size = [self.im_size[0] // self.patch_size[0],
                             self.im_size[1] // self.patch_size[1]]
        self.patch_num = self.patched_size[0] * self.patched_size[1]

        self.proj_mat = nn.Linear(self.patched_size[0] * self.patched_size[1], embedd_dim, bias=False)

        self.to(device)
        return

    def forward(self, x):
        # patching image into size [batch_size, patch_num, channels, height, width]
        x = patch_image(x, self.patch_size)

