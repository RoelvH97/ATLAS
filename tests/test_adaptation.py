import jax
import jax.numpy as jnp
import numpy as np
import pytest

from meta_learning.adaptation import Adaptation


class LinearField:
    def apply(self, variables, coords, latent):
        return variables["params"]["scale"] * latent * jnp.ones_like(coords[:, :1])


def test_maml_matches_closed_form_and_gradient():
    adaptation = Adaptation(steps=3, inner_lr=0.1, meta_sgd=False)
    coords, target = jnp.zeros((5, 2)), jnp.ones((5, 1))

    def adapted(scale):
        state = {"model": {"scale": scale}, "z_init": jnp.array([0.2])}
        return adaptation.fit(state, LinearField(), coords, target)[0]

    def exact(scale):
        return 1 / scale + (0.2 - 1 / scale) * (1 - 0.2 * scale**2) ** 3

    np.testing.assert_allclose(adapted(1.3), exact(1.3), rtol=1e-6)
    np.testing.assert_allclose(jax.grad(adapted)(1.3), jax.grad(exact)(1.3), rtol=1e-6, atol=1e-7)


def test_disjoint_steps_and_learned_rates():
    adaptation = Adaptation(steps=2, disjoint=True)
    coords, target = jnp.zeros((4, 2)), jnp.array([[1.0], [1.0], [3.0], [3.0]])

    def fit(rates):
        state = {"model": {"scale": 1.0}, "z_init": jnp.zeros(1), "meta_lr": rates}
        return adaptation.fit(state, LinearField(), coords, target)[0]

    rates = jnp.array([[0.1], [0.2]])
    np.testing.assert_allclose(fit(rates), 1.32, rtol=1e-6)
    np.testing.assert_allclose(jax.grad(fit)(rates), [[1.2], [5.6]], rtol=1e-6)


def test_stopped_code_has_no_latent_path_gradient():
    adaptation = Adaptation(meta_sgd=False, gradient_rule="stopped_code")
    np.testing.assert_array_equal(
        jax.grad(lambda z: adaptation.outer_latent(z).sum())(jnp.ones(4)), 0
    )
    with pytest.raises(ValueError, match="cannot learn"):
        Adaptation(gradient_rule="stopped_code")
