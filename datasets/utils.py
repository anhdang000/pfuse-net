import torch
from torch.utils.data.dataloader import default_collate


def collate_fn(batch):
    items = list(zip(*batch))
    items[0] = default_collate([i for i in items[0] if torch.is_tensor(i)])
    items[1] = default_collate([i for i in items[1] if torch.is_tensor(i)])
    items[2] = list([i for i in items[2] if i])
    items[3] = list([i for i in items[3] if i])
    items[4] = default_collate([i for i in items[4] if torch.is_tensor(i)])
    items[5] = default_collate([i for i in items[5] if torch.is_tensor(i)])
    return items