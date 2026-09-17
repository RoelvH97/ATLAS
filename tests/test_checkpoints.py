import jax.numpy as jnp
import numpy as np
import pytest

from general_utils.checkpoints import Checkpoint


def test_array_only_roundtrip_and_shape_validation(tmp_path):
    params = {
        "model": {"kernel": jnp.arange(6).reshape(2, 3)},
        "z_init": {"ctx": jnp.zeros(3), "pose": jnp.ones(2)},
    }
    path = tmp_path / "model.npz"
    Checkpoint(params, {"task": "reconstruct"}, {"epoch": 4}).save(path)
    with np.load(path, allow_pickle=False) as archive:
        assert all(not archive[key].dtype.hasobject for key in archive.files)
    loaded = Checkpoint.load(path)
    loaded.validate(params)
    assert loaded.metadata["epoch"] == 4
    np.testing.assert_array_equal(loaded.params["model"]["kernel"], params["model"]["kernel"])
    with pytest.raises(ValueError, match="shape"):
        loaded.validate({**params, "model": {"kernel": jnp.zeros((3, 2))}})


def test_legacy_pickle_requires_explicit_trust(tmp_path):
    with pytest.raises(ValueError, match="trusted=True"):
        Checkpoint.load(tmp_path / "legacy.npy")
