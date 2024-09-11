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
          save_period: int=None,
          save_idx: int=None,
          save_dir: str='./checkpoints/',
          device='cuda'):
    model.train()
    if save_dir[-1] != '/':
        save_dir += '/'

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

        if save_period and (epoch + 1) % save_period == 0:
            torch.save(model.state_dict(), save_dir + f'model_{save_idx}.pt')
            save_idx += 1

    if save_period and (epoch + 1) % save_period != 0:
        # 不可整除，当前模型尚未被保存，保存模型后再退出
        torch.save(model.state_dict(), save_dir + f'model_{save_idx}.pt')

    return
