import torch.nn as nn


class CNNFeatureExtract(nn.Module):
    def __init__(self,
                 input_shape,
                 n_features):
        super().__init__()
        self.c, self.h, self.w = input_shape
        self.n_features = n_features
        self.extractor = nn.Conv2d(self.c, n_features, kernel_size=3)

        return

    def forward(self, x):
        x = self.extractor(x)
        return x
