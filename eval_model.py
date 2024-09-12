import torch
import torch.nn as nn
from models import VisionTransformer
import torchvision
import torchvision.transforms as transforms
from methods import evaluate


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
batch_size = 128
datasets = torch.utils.data.DataLoader(trainset, batch_size, shuffle=False)

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
index = 56
vit.load_state_dict(torch.load(f'checkpoints/model_{index}.pt'))
loss, acc = evaluate(vit, datasets, nn.CrossEntropyLoss(), verbose=True)
print(loss, acc)
