import torch
from typing import Iterable, Callable
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import tqdm
from models import VisionTransformer


def calc_acc(prob_pred: torch.Tensor, label: torch.Tensor) -> float:
    label_pred = torch.argmax(prob_pred, dim=-1)
    batch_size = prob_pred.shape[0]
    acc = torch.sum(torch.eq(label_pred, label)) / batch_size

    return acc.item()


def train(model: nn.Module,
          datasets: Iterable,
          epochs: int,
          optimizer: torch.optim.Optimizer,
          loss_fn: Callable,
          lr_scheduler=None,
          device='cuda'):
    model.train()

    for epoch in range(epochs):
        iteration = tqdm.tqdm(datasets)
        optimizer.zero_grad()

        for image, label in iteration:
            image = image.to(device)
            label = label.to(device)

            pred = model(image)
            loss = loss_fn(pred, label)

            loss.backward()
            optimizer.step()

            if lr_scheduler:
                lr_scheduler.step()

            iteration.set_postfix({'loss': loss.item(), 'acc': calc_acc(torch.softmax(pred, dim=-1), label)})
            iteration.set_description(f'Epochs: {epoch} / {epochs}')

    return


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
batch_size = 64
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
train(vit, datasets, 1, optimizer, nn.CrossEntropyLoss(), device='cuda')
