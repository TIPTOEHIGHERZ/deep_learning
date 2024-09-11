import torch
from typing import Iterable, Callable
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from models import VisionTransformer
import os
from methods import train


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
batch_size = 128
datasets = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True)
# image, label = next(iter(datasets))
# print(label)

vit = VisionTransformer(10,
                        3,
                        [32, 32],
                        [8, 8],
                        64,
                        128,
                        768,
                        12,
                        3796,
                        12)
optimizer = torch.optim.Adam(vit.parameters(), lr=1e-5, betas=(0.9, 0.999))
idx = 23
epochs = 100
save_period = 10
print(f'load model_{idx}.pt')
vit.load_state_dict(torch.load(f'checkpoints/model_{idx}.pt', weights_only=False))
train(vit, datasets, 10, optimizer, nn.CrossEntropyLoss(),
      save_period=save_period,
      save_idx=idx + 1,
      save_dir='./checkpoints')
