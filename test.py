import torch
import torch.nn as nn
from models import VisionTransformer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


vit = VisionTransformer(1000,
                        3,
                        [224, 224],
                        [32, 32],
                        64,
                        128,
                        768,
                        12,
                        3796,
                        12)

print(count_parameters(vit))


