import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features,
                 mid_features,
                 out_features,
                 drop_out: float = 0.1,
                 device='cuda'):
        super().__init__()

        self.linear_0 = nn.Linear(in_features, mid_features)
        self.gelu = nn.GELU()
        self.linear_1 = nn.Linear(mid_features, out_features)
        self.drop_out = nn.Dropout(drop_out)

        self.to(device)
        return

    def forward(self, x):
        x = self.linear_0(x)
        x = self.gelu(x)
        x = self.linear_1(x)
        x = self.drop_out(x)

        return x
