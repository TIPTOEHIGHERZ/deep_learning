import torch
import torch.nn as nn


class PositionalEncode(nn.Module):
    def __init__(self,
                 feature_shape,
                 device='cuda'):
        super().__init__()
        self.encoding = torch.zeros(feature_shape, device=device, requires_grad=False)
        pos = torch.arange(0, feature_shape[0], 1, device=device).unsqueeze(dim=1)
        i = torch.arange(0, feature_shape[1], 2, device=device).unsqueeze(dim=0)
        self.encoding[:, 0::2] = torch.sin(pos / torch.pow(10000, i / feature_shape[1]))
        self.encoding[:, 1::2] = torch.cos(pos / torch.pow(10000, i / feature_shape[1]))

        self.to(device)
        return

    def forward(self, x):
        return x + self.encoding

