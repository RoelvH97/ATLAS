"""CIFAR-10 with the fixed 49,900/100/10,000 split."""

import numpy as np
import torch
from torchvision.datasets import CIFAR10


class CIFAR:
    def __init__(self, root, split, n_val=100, download=False, split_seed=0):
        self.dataset = CIFAR10(root=root, train=split != "test", download=download)
        indices = np.arange(len(self.dataset))
        if split != "test":
            order = torch.randperm(
                len(indices), generator=torch.Generator().manual_seed(split_seed)
            ).numpy()
            indices = order[:n_val] if split == "val" else order[n_val:]
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.dataset[int(self.indices[index])]
        image = np.asarray(image).astype(np.float32) / 255.0
        return (image - 0.5) / 0.5, np.int64(label)
