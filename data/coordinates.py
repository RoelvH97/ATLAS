"""Pixel-center coordinates and reproducible coordinate sampling."""

import hashlib

import numpy as np


def stable_seed(*parts):
    encoded = "\0".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")


def grid(height, width=None, dtype=np.float32, array_module=np):
    width = height if width is None else width
    x = array_module.linspace(-1 + 1 / width, 1 - 1 / width, width, dtype=dtype)
    y = array_module.linspace(-1 + 1 / height, 1 - 1 / height, height, dtype=dtype)
    xx, yy = array_module.meshgrid(x, y, indexing="xy")
    return array_module.stack((xx.ravel(), yy.ravel()), axis=-1)


def balanced_indices(rng, candidates, count):
    if len(candidates) == 0:
        raise ValueError("Balanced sampling requires both segmentation classes")
    quotient, remainder = divmod(count, len(candidates))
    blocks = [rng.permutation(candidates) for _ in range(quotient)]
    if remainder:
        blocks.append(rng.permutation(candidates)[:remainder])
    indices = np.concatenate(blocks)
    rng.shuffle(indices)
    return indices
