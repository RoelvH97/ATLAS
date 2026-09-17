import jax
import jax.numpy as jnp
import numpy as np
import pytest

from meta_learning.adaptation import Adaptation
from meta_learning.model import ModelFactory
from models import ATLAS, ENF, Functa, SpatialFuncta


@pytest.mark.parametrize(
    "model",
    [
        ATLAS(
            n_latents=4,
            latent_channels=4,
            model_dim=16,
            n_heads=2,
            n_self_attn=1,
            nearest_k=2,
            sa_nearest_k=2,
        ),
        ENF(n_latents=4, latent_channels=4, num_hidden=16, att_dim=8, nearest_k=2),
        Functa(latent_dim=8, hidden_dim=16, num_layers=4),
        SpatialFuncta(latent_dim=8, spatial_size=2, latent_channels=2, hidden_dim=16, num_layers=4),
    ],
)
def test_model_adapts_with_finite_meta_gradients(model):
    coords = jnp.array([[-0.4, -0.3], [0.7, -0.2], [-0.2, 0.6], [0.3, 0.4]])
    target = jnp.broadcast_to(coords[:, :1], (4, 3))
    adaptation = Adaptation()
    params, _ = ModelFactory.initialize(model, jax.random.PRNGKey(0), coords, adaptation)

    def loss(state):
        latent = adaptation.fit(state, model, coords, target)
        return jnp.mean((model.apply({"params": state["model"]}, coords, latent) - target) ** 2)

    value, gradient = jax.jit(jax.value_and_grad(loss))(params)
    assert np.isfinite(value)
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(gradient))
    assert any(np.count_nonzero(leaf) for leaf in jax.tree.leaves(gradient["meta_lr"]))


def test_atlas_translation_equivariance():
    model = ATLAS(
        n_latents=4,
        latent_channels=4,
        model_dim=16,
        n_heads=2,
        n_self_attn=2,
        nearest_k=2,
        sa_nearest_k=2,
        bounded_pose=False,
    )
    coords = jnp.array([[-0.4, -0.3], [0.7, -0.2], [-0.2, 0.6]])
    params, _ = ModelFactory.initialize(model, jax.random.PRNGKey(0), coords, Adaptation())
    latent = params["z_init"]
    shift = jnp.array([0.15, -0.27])
    translated = {**latent, "pose": (latent["pose"].reshape(-1, 2) + shift).ravel()}
    original = model.apply({"params": params["model"]}, coords, latent)
    moved = model.apply({"params": params["model"]}, coords + shift, translated)
    np.testing.assert_allclose(original, moved, atol=1e-6, rtol=1e-5)
