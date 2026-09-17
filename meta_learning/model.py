"""Model construction and the shared initial latent state."""

import math
from dataclasses import fields
from typing import ClassVar

import jax
import jax.numpy as jnp

from models import ATLAS, ENF, Functa, SpatialFuncta


class ModelFactory:
    models: ClassVar[dict[str, type]] = {
        "atlas": ATLAS,
        "enf": ENF,
        "functa": Functa,
        "spatial_functa": SpatialFuncta,
    }

    @classmethod
    def build(cls, config, output_dim, image_size=32):
        options = dict(config)
        kind = options.pop("type")
        if kind not in cls.models:
            raise ValueError(f"Unknown model {kind}; choose from {tuple(cls.models)}")
        model_class = cls.models[kind]
        options["output_dim"] = output_dim
        if kind == "spatial_functa":
            options["image_size"] = image_size
        if options.get("latent_grid") is not None:
            options["latent_grid"] = tuple(options["latent_grid"])
        allowed = {field.name for field in fields(model_class)}
        unknown = options.keys() - allowed
        if unknown:
            raise ValueError(f"Unknown {kind} settings: {sorted(unknown)}")
        return model_class(**options)

    @staticmethod
    def latent(model):
        if isinstance(model, (Functa, SpatialFuncta)):
            return jnp.zeros(model.latent_dim)
        size = round(model.n_latents ** (1 / model.coord_dim))
        grid = model.latent_grid or (size,) * model.coord_dim
        if math.prod(grid) != model.n_latents:
            raise ValueError("latent_grid must contain exactly n_latents points")
        axes = [jnp.linspace(-1 + 1 / n, 1 - 1 / n, n) for n in grid]
        mesh = jnp.meshgrid(*axes, indexing="xy")
        position = jnp.stack([axis.ravel() for axis in mesh], axis=-1) + 1e-3
        if model.bounded_pose:
            position = jnp.arctanh(jnp.clip(position, -1 + 1e-6, 1 - 1e-6))
        if model.bi_invariant == "roto_translation":
            position = jnp.concatenate((position, jnp.zeros((model.n_latents, 1))), axis=-1)
        elif model.bi_invariant == "roto_translation_3d":
            frame = jnp.broadcast_to(
                jnp.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), (model.n_latents, 6)
            )
            position = jnp.concatenate((position, frame), axis=-1)
        return {
            "pose": position.ravel(),
            "ctx": (
                jnp.ones((model.n_latents, model.latent_channels)) / model.latent_channels
            ).ravel(),
        }

    @classmethod
    def initialize(cls, model, key, coords, adaptation):
        latent = cls.latent(model)
        dummy = jax.tree.map(jnp.zeros_like, latent)
        key, initialization = jax.random.split(key)
        state = {"model": model.init(initialization, coords, dummy)["params"], "z_init": latent}
        rates = adaptation.rates(latent)
        if adaptation.meta_sgd:
            state["meta_lr"] = jax.tree.map(
                lambda z, rate: jnp.full(
                    ((adaptation.steps,) if adaptation.per_step else ()) + z.shape, rate
                ),
                latent,
                rates,
            )
        return state, key
