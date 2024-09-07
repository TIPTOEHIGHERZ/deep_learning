import torch.nn as nn


class CNNFeatureExtract(nn.Module):
    def __init__(self,
                 input_shape,
                 in_channels,
                 out_channels,
                 device='cuda'):
        super().__init__()
        self.h, self.w = input_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.extractor = nn.Conv2d(self.in_channels, out_channels, kernel_size=3, padding=1)
        self.device = device
        self.to(self.device)

        return

    def forward(self, x):
        x = self.extractor(x)
        return x
