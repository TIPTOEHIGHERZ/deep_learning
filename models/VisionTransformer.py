import torch
import torch.nn as nn
from typing import Union, Iterable, List
from .CNNFE import CNNFeatureExtract
from .PatchEmbedding import PatchEmbedding
from .TransformerEncoder import TransformerEncoder


class VisionTransformer(nn.Module):
    def __init__(self,
                 label_num: int,
                 channels: int,
                 im_size: list,
                 patch_size: Union[int, list],
                 n_features: int,
                 token_len: int,
                 embedd_dim: int,
                 head_num: int,
                 mlp_dim: int,
                 n_encoders: int,
                 device='cuda'
                 ):
        super().__init__()

        self.im_size = im_size
        self.channels = channels
        self.patch_size = [patch_size, patch_size] if isinstance(patch_size, int) else patch_size
        self.n_features = n_features
        self.embedd_dim = embedd_dim
        self.token_len = token_len
        self.device = device

        self.patched_size = [self.im_size[0] // self.patch_size[0],
                             self.im_size[1] // self.patch_size[1]]
        self.patch_num = self.patched_size[0] * self.patched_size[1]

        self.extractor = CNNFeatureExtract(self.im_size, self.channels, self.n_features)
        self.patch_embedding = PatchEmbedding(self.n_features,
                                              self.im_size,
                                              self.patch_size,
                                              self.embedd_dim,
                                              None,
                                              device=device)
        self.encoder_blocks = nn.Sequential(*[TransformerEncoder(embedd_dim, head_num, mlp_dim, device=device)
                                              for _ in range(n_encoders)])
        self.mlp_head = nn.Linear(embedd_dim, label_num)

        self.to(device)
        return

    def forward(self, x):
        x = self.extractor(x)
        # patching image into size [batch_size, patch_num, channels, height, width]
        x = self.patch_embedding(x)
        x = self.encoder_blocks(x)
        x = self.mlp_head(x)

        return nn.functional.softmax(x, dim=-1)

