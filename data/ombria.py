"""Co-registered Sentinel-1/2 imagery and flood masks."""

import re
from pathlib import Path

import numpy as np
from PIL import Image

from .coordinates import stable_seed


class Ombria:
    sources = (
        ("OmbriaS1", "BEFORE", "S1_before"),
        ("OmbriaS1", "AFTER", "S1_after"),
        ("OmbriaS2", "BEFORE", "S2_before"),
        ("OmbriaS2", "AFTER", "S2_after"),
        ("OmbriaS1", "MASK", "S1_mask"),
        ("OmbriaS2", "MASK", "S2_mask"),
    )

    def __init__(self, root, split, n_val=10, split_seed=0):
        root = Path(root).expanduser()
        source_split = "test" if split == "test" else "train"
        inventories = []
        for modality, phase, prefix in self.sources:
            directory = root / modality / source_split / phase
            if not directory.is_dir():
                raise FileNotFoundError(directory)
            pattern = re.compile(rf"{prefix}_(\d+)\.png", re.IGNORECASE)
            files = {}
            for path in directory.iterdir():
                if path.name.startswith("."):
                    continue
                match = pattern.fullmatch(path.name)
                if not match or int(match[1]) in files:
                    raise ValueError(f"Invalid or duplicate Ombria sample: {path}")
                files[int(match[1])] = path
            inventories.append(files)
        ids = sorted(inventories[0])
        if not ids or any(set(inventory) != set(ids) for inventory in inventories):
            raise ValueError("Ombria modalities must contain identical, nonempty sample IDs")
        if split != "test":
            if not 0 < n_val < len(ids):
                raise ValueError("n_val must leave nonempty training and validation sets")
            rng = np.random.default_rng(stable_seed("ombria-val", int(split_seed)))
            selected = set(rng.permutation(len(ids))[:n_val].tolist())
            ids = [
                sample
                for index, sample in enumerate(ids)
                if (index in selected) == (split == "val")
            ]
        self.ids = ids
        self.paths = [tuple(inventory[sample] for inventory in inventories) for sample in ids]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        arrays = []
        for position, path in enumerate(self.paths[index]):
            with Image.open(path) as image:
                array = np.asarray(image.convert("RGB" if position in (2, 3) else "L"))
            arrays.append(array)
        if len({array.shape[:2] for array in arrays}) != 1:
            raise ValueError(f"Ombria sample {self.ids[index]} is not co-registered")
        if any(not np.isin(mask, [0, 1, 255]).all() for mask in arrays[4:]):
            raise ValueError("Ombria masks must be binary")
        if not np.array_equal(arrays[4] > 0, arrays[5] > 0):
            raise ValueError("Sentinel-1 and Sentinel-2 masks disagree")
        values = np.concatenate(
            (arrays[0][..., None], arrays[1][..., None], arrays[2], arrays[3]), axis=-1
        )
        return values.astype(np.float32) / 127.5 - 1, (arrays[4] > 0).astype(np.float32)
