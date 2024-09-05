import torch
import torch.nn as nn
from typing import Union, Iterable, List
from .CNNFE import CNNFeatureExtract
from .PatchEmbedding import PatchEmbedding


class VisionTransformer(nn.Module):
    def __init__(self,
                 im_size: List[int, int],
                 batch_size: int,
                 patch_size: Union[int, Iterable[int, int]],
                 n_features,
                 embedd_dim: int
                 ):
        super().__init__()

        self.im_size = im_size
        self.batch_size = batch_size
        self.patch_size = [patch_size, patch_size] if isinstance(patch_size, int) else patch_size
        self.n_features = n_features
        self.embedd_dim = embedd_dim

        self.patched_size = [self.im_size[0] // self.patch_size[0],
                             self.im_size[1] // self.patch_size[1]]
        self.patch_num = self.patched_size[0] * self.patched_size[1]

        self.extractor = CNNFeatureExtract(self.patched_size, self.n_features)
        self.patch_embedding = PatchEmbedding

        return

    def forward(self, x):
        x = self.extractor(x)
        # patching image into size [batch_size, patch_num, channels, height, width]
        x = patch_image(x, self.patch_size)

