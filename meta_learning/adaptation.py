"""Differentiable latent adaptation shared by all tasks."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Adaptation:
    steps: int = 3
    inner_lr: float = 0.01
    pose_lr: float = 0.1
    content_lr: float = 0.2
    meta_sgd: bool = True
    per_step: bool = True
    disjoint: bool = False
    gradient_rule: str = "maml"

    @classmethod
    def from_config(cls, config):
        return cls(**{name: config[name] for name in cls.__dataclass_fields__ if name in config})

    def __post_init__(self):
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.gradient_rule not in ("maml", "stopped_code"):
            raise ValueError("gradient_rule must be maml or stopped_code")
        if self.meta_sgd and self.gradient_rule == "stopped_code":
            raise ValueError("Stopped-code training cannot learn inner step sizes")

    def rates(self, latent):
        if isinstance(latent, dict):
            return {"pose": self.pose_lr, "ctx": self.content_lr}
        return self.inner_lr

    def fit(self, state, model, coords, values):
        latent = state["z_init"]
        rates = state.get("meta_lr", self.rates(latent))
        if self.disjoint and coords.shape[0] % self.steps:
            raise ValueError("Disjoint support must divide evenly across steps")
        count = coords.shape[0] // self.steps
        per_step = all(
            jnp.shape(rate) == (self.steps, *value.shape)
            for rate, value in zip(jax.tree.leaves(rates), jax.tree.leaves(latent), strict=True)
        )

        def loss(code, x, y):
            prediction = model.apply({"params": state["model"]}, x, code)
            return jnp.mean((prediction[..., : y.shape[-1]] - y) ** 2)

        for step in range(self.steps):
            x = coords[step * count : (step + 1) * count] if self.disjoint else coords
            y = values[step * count : (step + 1) * count] if self.disjoint else values
            gradient = jax.grad(loss)(latent, x, y)
            rate = (
                jax.tree.map(lambda value, index=step: value[index], rates) if per_step else rates
            )
            if not isinstance(rate, dict) and isinstance(latent, dict):
                rate = jax.tree.map(lambda _, value=rate: value, latent)
            latent = jax.tree.map(lambda z, g, lr: z - lr * g, latent, gradient, rate)
        return latent

    def outer_latent(self, latent):
        if self.gradient_rule == "stopped_code":
            return jax.tree.map(jax.lax.stop_gradient, latent)
        return latent
