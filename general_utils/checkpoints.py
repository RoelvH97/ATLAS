"""Array-only checkpoints with support for trusted legacy NumPy states."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict
from omegaconf import OmegaConf


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Checkpoint:
    params: dict
    config: dict
    metadata: dict

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            name: np.asarray(value) for name, value in flatten_dict(self.params, sep="/").items()
        }
        arrays["__metadata__"] = np.array(
            json.dumps({"format_version": 1, "config": self.config, **self.metadata})
        )
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)

    @classmethod
    def load(cls, path, *, legacy_config=None, trusted=False):
        path = Path(path)
        if path.suffix == ".npy":
            if not trusted:
                raise ValueError("Legacy .npy checkpoints require trusted=True")
            config_path = (
                Path(legacy_config) if legacy_config else path.parent / ".hydra/config.yaml"
            )
            config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)
            params = np.load(path, allow_pickle=True).item()
            metadata = {"legacy": True}
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["__metadata__"]))
                if metadata.pop("format_version") != 1:
                    raise ValueError("Unsupported checkpoint version")
                config = metadata.pop("config")
                params = unflatten_dict(
                    {key: archive[key].copy() for key in archive.files if key != "__metadata__"},
                    sep="/",
                )
        else:
            raise ValueError("Checkpoint must be .npz or .npy")
        if not {"model", "z_init"} <= params.keys():
            raise ValueError("Checkpoint lacks model or z_init")
        for value in jax.tree.leaves(params):
            if (
                not np.issubdtype(np.asarray(value).dtype, np.number)
                or not np.isfinite(value).all()
            ):
                raise ValueError("Checkpoint contains non-numeric or non-finite parameters")
        return cls(jax.tree.map(jnp.asarray, params), config, metadata)

    def validate(self, expected):
        actual = flatten_dict(self.params, sep="/")
        wanted = flatten_dict(expected, sep="/")
        if actual.keys() != wanted.keys():
            raise ValueError(f"Checkpoint keys differ: {sorted(actual.keys() ^ wanted.keys())}")
        for key in actual:
            if actual[key].shape != wanted[key].shape:
                raise ValueError(
                    f"Checkpoint shape differs for {key}: {actual[key].shape} vs {wanted[key].shape}"
                )
