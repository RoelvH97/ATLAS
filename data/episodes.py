"""A common support/query interface for all three datasets."""

import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .cifar import CIFAR
from .coordinates import balanced_indices, grid, stable_seed
from .ombria import Ombria
from .polynomials import PolynomialFields


class Episodes(Dataset):
    def __init__(
        self, dataset, config, split, steps=3, disjoint=False, limit=0, coordinate_dtype="float32"
    ):
        self.coordinate_dtype = np.dtype(coordinate_dtype)
        self.dataset, self.config, self.split = dataset, config, split
        self.steps, self.disjoint, self.epoch = steps, disjoint, 0
        self.limit = min(limit, len(dataset)) if limit else len(dataset)
        self.coordinates = (
            np.asarray(grid(config["image_size"], dtype=self.coordinate_dtype, array_module=jnp))
            if isinstance(dataset, (CIFAR, PolynomialFields))
            else None
        )

    def __len__(self):
        return self.limit

    def rng(self, index, stream):
        epoch = self.epoch if self.split == "train" else 0
        seed = self.config["coord_seed"] if self.split == "train" else self.config["eval_seed"]
        if isinstance(self.dataset, Ombria):
            identity = self.dataset.ids[index]
            parts = (seed, self.split, epoch, identity, stream, "balanced_flood_v1")
        else:
            parts = (seed, self.split, epoch, index, stream)
        return np.random.default_rng(stable_seed(*parts))

    def __getitem__(self, index):
        values, labels = self.dataset[index]
        if (
            self.split == "train"
            and self.config["random_flip"]
            and self.rng(index, "flip").random() < 0.5
        ):
            values = np.flip(values, axis=1).copy()
            if np.ndim(labels) == 2:
                labels = np.flip(labels, axis=1).copy()
        height, width = values.shape[:2]
        if height != self.config["image_size"] or width != self.config["image_size"]:
            raise ValueError("data.image_size must match the input image dimensions")
        coordinates = (
            self.coordinates
            if self.coordinates is not None
            else grid(height, width, dtype=self.coordinate_dtype)
        )
        values = values.reshape(-1, values.shape[-1])
        count = int(self.config["inner_coords"] or len(values))
        count *= self.steps if self.disjoint else 1
        if count > len(values):
            raise ValueError(f"Support count {count} exceeds {len(values)} pixels")
        support = (
            np.arange(len(values))
            if count == len(values)
            else self.rng(index, "support").choice(len(values), count, replace=False)
        )
        if isinstance(self.dataset, PolynomialFields) and self.split != "train":
            namespace = 0 if self.split == "val" else 1
            rng = np.random.default_rng(np.random.SeedSequence([20260813, namespace, index]))
            support = np.sort(rng.choice(len(values), count, replace=False))
        query = support
        if self.split != "train" or self.config["outer_full_grid"]:
            query = np.arange(len(values))
        elif np.ndim(labels) == 2:
            flat_mask = labels.ravel()
            count = int(self.config["outer_coords"])
            if count < 2 or count % 2:
                raise ValueError("Balanced outer_coords must be positive and even")
            rng = self.rng(index, "outer")
            query = np.concatenate(
                (
                    balanced_indices(rng, np.flatnonzero(flat_mask > 0.5), count // 2),
                    balanced_indices(rng, np.flatnonzero(flat_mask <= 0.5), count // 2),
                )
            )
            rng.shuffle(query)
        elif self.config["outer_coords"] and not self.config["shared_queries"]:
            query = self.rng(index, "outer").choice(
                len(values), self.config["outer_coords"], replace=False
            )
        if isinstance(self.dataset, PolynomialFields) and self.split != "train":
            query = np.setdiff1d(np.arange(len(values)), support)
        return {
            "support_coords": coordinates[support],
            "support_values": values[support],
            "query_coords": coordinates[query],
            "query_values": values[query],
            "labels": labels.ravel()[query] if np.ndim(labels) == 2 else labels,
            "shape": np.array([height, width]),
            "index": np.int64(index),
        }


class DataModule:
    def __init__(self, config, adaptation, seed=42, coordinate_dtype="float32"):
        self.coordinate_dtype = coordinate_dtype
        self.config, self.adaptation, self.seed = config, adaptation, seed

    def dataset(self, split):
        cfg = self.config
        if cfg["type"] == "cifar":
            dataset = CIFAR(cfg["root"], split, cfg["n_val"], cfg["download"], cfg["split_seed"])
        elif cfg["type"] == "ombria":
            dataset = Ombria(cfg["root"], split, cfg["n_val"], cfg["split_seed"])
        elif cfg["type"] == "polynomials":
            dataset = PolynomialFields(split, size=cfg["image_size"], **cfg["polynomials"])
        else:
            raise ValueError(f"Unknown dataset: {cfg['type']}")
        return Episodes(
            dataset,
            cfg,
            split,
            self.adaptation.steps,
            self.adaptation.disjoint,
            cfg["limit_train"] if split == "train" else cfg["limit_eval"],
            self.coordinate_dtype,
        )

    def loader(self, dataset):
        training = dataset.split == "train"
        batch_size = self.config["batch_size"] if training else 1
        if len(dataset) < batch_size:
            raise ValueError("Dataset must contain at least one complete batch")
        generator = torch.Generator().manual_seed(self.seed + dataset.epoch)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=training,
            drop_last=training,
            num_workers=self.config["num_workers"],
            generator=generator,
            multiprocessing_context="spawn" if self.config["num_workers"] else None,
        )
