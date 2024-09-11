from typing import Iterable, Callable
import torch.utils
import torch.utils.data
import tqdm
import torch
import torch.nn as nn


def calc_acc(prob_pred: torch.Tensor, label: torch.Tensor) -> float:
    label_pred = torch.argmax(prob_pred, dim=-1)
    batch_size = prob_pred.shape[0]
    acc = torch.sum(torch.eq(label_pred, label)) / batch_size

    return acc.item()


def evaluate(model: nn.Module,
             datasets: torch.utils.data.DataLoader,
             loss_fn: Callable,
             verbose=False,
             device='cuda'):
    model.eval()

    acc_sum = 0.0
    loss_sum = 0.0
    total_datasets = 0 if verbose else len(datasets.dataset)
    iteration = tqdm.tqdm(datasets) if verbose else datasets

    with torch.no_grad():
        for image, label in iteration:
            image = image.to(device)
            label = label.to(device)

            pred = model(image)
            loss = loss_fn(pred, label)
            acc = calc_acc(torch.softmax(pred, dim=-1), label)

            loss_sum += loss.item() * image.shape[0]
            acc_sum += acc * image.shape[0]

            if verbose:
                total_datasets += image.shape[0]
                avg_loss = loss_sum / total_datasets
                avg_acc = acc_sum / total_datasets
                iteration.set_postfix({'loss': avg_loss, 'acc': avg_acc}) 

    avg_loss = loss_sum / total_datasets
    avg_acc = acc_sum / total_datasets

    return avg_loss, avg_acc 
