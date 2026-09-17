"""Degree-six polynomial fields and their exact coefficient representation."""

import numpy as np

from .coordinates import grid


class PolynomialFields:
    def __init__(
        self,
        split,
        size=32,
        degree=6,
        n_train=10000,
        n_val=1024,
        n_test=1024,
        seed=0,
        degree_decay=0.85,
        offset_scale=0.5,
        target_bound=0.9,
        min_peak_fraction=0.65,
    ):
        self.size = size
        self.exponents = np.asarray(
            [(d - q, q) for d in range(degree + 1) for q in range(d + 1)], dtype=np.int32
        )
        count = {"train": n_train, "val": n_val, "test": n_test}[split]
        rng = np.random.default_rng(seed + {"train": 0, "val": 1, "test": 2}[split])
        scale = degree_decay ** self.exponents.sum(axis=1)
        scale[0] *= offset_scale
        coefficients = rng.uniform(-1, 1, (count, len(scale))) * scale
        self.coords = grid(size, dtype=np.float64)
        self.design = self.basis(self.coords)
        peak = np.max(np.abs(coefficients @ self.design.T), axis=1)
        desired = target_bound * rng.uniform(min_peak_fraction, 1, count)
        self.coefficients = (coefficients * (desired / peak)[:, None]).astype(np.float32)

    def basis(self, coords):
        coords = np.asarray(coords)
        return (
            coords[:, :1] ** self.exponents[None, :, 0]
            * coords[:, 1:2] ** self.exponents[None, :, 1]
        )

    def __len__(self):
        return len(self.coefficients)

    def __getitem__(self, index):
        values = self.coefficients[index] @ self.design.T
        return values.astype(np.float32).reshape(self.size, self.size, 1), np.int64(0)
