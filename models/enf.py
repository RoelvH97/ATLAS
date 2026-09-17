"""Equivariant neural field with posed content latents."""

import flax.linen as nn
import jax
import jax.numpy as jnp

from .attention import FourierEmbedding
from .geometry import bound_pose, pose_dimension, relative_coordinates


class ENF(nn.Module):
    output_dim: int = 3
    n_latents: int = 25
    latent_channels: int = 64
    num_hidden: int = 128
    att_dim: int = 64
    num_heads: int = 3
    nearest_k: int | None = 4
    gaussian_window: bool = True
    emb_freq_q: float = 1.0
    emb_freq_v: float = 3.0
    bi_invariant: str = "translation"
    coord_dim: int = 2
    bounded_pose: bool = False
    latent_grid: tuple | None = None

    @property
    def pose_dim(self):
        return pose_dimension(self.bi_invariant, self.coord_dim)

    def setup(self):
        self.pos_emb_q = nn.Sequential(
            [
                FourierEmbedding(self.num_hidden, self.emb_freq_q),
                nn.Dense(self.num_heads * self.att_dim),
            ]
        )
        self.pos_emb_v = nn.Sequential(
            [FourierEmbedding(self.num_hidden, self.emb_freq_v), nn.Dense(2 * self.num_hidden)]
        )
        self.W_k = nn.Dense(self.num_heads * self.att_dim)
        self.W_v = nn.Dense(self.num_hidden)
        self.v_cond = nn.Sequential(
            [nn.Dense(self.num_hidden), nn.gelu, nn.Dense(self.num_heads * self.num_hidden)]
        )
        self.W_out = nn.Dense(self.output_dim)

    def token_features(self, latent):
        return latent["ctx"].reshape(self.n_latents, self.latent_channels)

    def __call__(self, coords, latent):
        pose = latent["pose"].reshape(self.n_latents, self.pose_dim)
        if self.bounded_pose:
            pose = bound_pose(pose, self.bi_invariant)
        content = self.token_features(latent)
        relative = relative_coordinates(coords, pose, self.bi_invariant)
        distance = (relative[..., : self.coord_dim] ** 2).sum(-1)
        count = self.n_latents if self.nearest_k is None else min(self.nearest_k, self.n_latents)
        if self.nearest_k is None:
            keys = self.W_k(content)[None].repeat(coords.shape[0], axis=0)
            values = self.W_v(content)[None].repeat(coords.shape[0], axis=0)
        else:
            indices = jnp.argsort(distance, axis=-1)[:, :count]
            rows = jnp.arange(coords.shape[0])[:, None]
            relative, distance = relative[rows, indices], distance[rows, indices]
            keys, values = self.W_k(content)[indices], self.W_v(content)[indices]
        query = self.pos_emb_q(relative)
        scale, shift = jnp.split(self.pos_emb_v(relative), 2, axis=-1)
        values = self.v_cond(values * (1 + scale) + shift)
        query = query.reshape(coords.shape[0], count, self.num_heads, self.att_dim)
        keys = keys.reshape(query.shape)
        values = values.reshape(coords.shape[0], count, self.num_heads, self.num_hidden)
        logits = (query * keys).sum(-1, keepdims=True)
        if self.gaussian_window:
            if self.latent_grid is None:
                window = self.coord_dim / self.n_latents ** (1 / self.coord_dim)
                penalty = (1 / window**2) * distance
            else:
                inverse = jnp.asarray([(size / self.coord_dim) ** 2 for size in self.latent_grid])
                penalty = (relative[..., : self.coord_dim] ** 2 * inverse).sum(-1)
            logits = logits - penalty[..., None, None]
        weights = jax.nn.softmax(logits, axis=1)
        output = (weights * values).sum(axis=1).reshape(coords.shape[0], -1)
        return self.W_out(output)
