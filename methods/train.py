from typing import Iterable, Callable
import tqdm
import torch
import torch.nn as nn
from .evaluate import evaluate, calc_acc


def train(model: nn.Module,
          datasets: Iterable,
          epochs: int,
          optimizer: torch.optim.Optimizer,
          loss_fn: Callable,
          lr_scheduler=None,
          device='cuda'):
    model.train()

    acc_sum = 0.0
    loss_sum = 0.0
    total_data = 0
    
    for epoch in range(epochs):
        iteration = tqdm.tqdm(datasets)

        for image, label in iteration:
            image = image.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()

            pred = model(image)
            loss = loss_fn(pred, label)

            loss.backward()
            optimizer.step()

            if lr_scheduler:
                lr_scheduler.step()

            total_data += image.shape[0]
            acc_sum += calc_acc(torch.softmax(pred, dim=-1), label) * image.shape[0]
            loss_sum += loss.item() * image.shape[0]

            iteration.set_postfix({'loss': loss_sum / total_data, 'acc': acc_sum / total_data})
            iteration.set_description(f'Epochs: {epoch} / {epochs}')

        # evaluate model performance
        # loss, acc = evaluate(model, datasets, loss_fn, verbose=False, device=device)
        # iteration.set_postfix({'loss': loss, 'acc': acc})
        # iteration.set_description(f'Epochs: {epoch + 1} / {epochs}')

    return
