import torch
import copy

device = torch.device('cuda:0')
# device = torch.device('cpu')
a = torch.tensor([1., 2.], device=device)
b = copy.deepcopy(a)
b[0] = 2.0
print(a, b)
