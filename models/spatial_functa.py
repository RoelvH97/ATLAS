"""Spatial Functa with interpolated SIREN modulations."""

import math

import flax.linen as nn
import jax.numpy as jnp

from .siren import sine_network


class SpatialFuncta(nn.Module):
    output_dim: int = 3
    latent_dim: int = 1024
    spatial_size: int = 8
    latent_channels: int = 16
    hidden_dim: int = 256
    num_layers: int = 6
    image_size: int = 32
    w0: float = 10.0
    interp_mode: str = "bilinear"
    coord_dim: int = 2

    def token_features(self, latent):
        return latent.reshape(self.spatial_size**2, self.latent_channels)

    def sample(self, grid, coords):
        unit = (coords + 1) * 0.5
        size = self.spatial_size
        if self.interp_mode == "nearest":
            index = jnp.floor(unit * size).astype(jnp.int32).clip(0, size - 1)
            return grid[index[:, 1], index[:, 0]]
        offset = unit * size - 0.5
        lower = jnp.floor(offset).astype(jnp.int32).clip(0, size - 1)
        upper = (lower + 1).clip(0, size - 1)
        weight = jnp.clip(offset - lower.astype(jnp.float32), 0, 1)
        x, y = weight[:, 0, None], weight[:, 1, None]
        return (
            (1 - y) * (1 - x) * grid[lower[:, 1], lower[:, 0]]
            + (1 - y) * x * grid[lower[:, 1], upper[:, 0]]
            + y * (1 - x) * grid[upper[:, 1], lower[:, 0]]
            + y * x * grid[upper[:, 1], upper[:, 0]]
        )

    def coordinates(self, coords):
        if self.spatial_size == 1:
            return coords
        unit = (coords + 1) * 0.5
        if self.interp_mode == "nearest":
            return (self.spatial_size * unit) % 1
        bits = max(1, math.ceil(math.log2(self.image_size)))
        x = jnp.floor(unit[:, 0] * self.image_size).astype(jnp.uint16)
        y = jnp.floor(unit[:, 1] * self.image_size).astype(jnp.uint16)
        x_bits = ((x[..., None] >> jnp.arange(bits)) & 1).astype(jnp.float32)
        y_bits = ((y[..., None] >> jnp.arange(bits)) & 1).astype(jnp.float32)
        return jnp.concatenate((x_bits, y_bits), axis=-1)

    @nn.compact
    def __call__(self, coords, latent):
        if self.coord_dim != 2 or self.interp_mode not in ("bilinear", "nearest"):
            raise ValueError("SpatialFuncta supports 2D bilinear or nearest interpolation")
        if self.latent_dim != self.spatial_size**2 * self.latent_channels:
            raise ValueError("latent_dim must equal spatial_size**2 * latent_channels")
        grid = latent.reshape(self.spatial_size, self.spatial_size, self.latent_channels)
        shifts = nn.Dense((self.num_layers - 1) * self.hidden_dim, name="lat_to_mod")(grid)
        shifts = self.sample(shifts, coords).reshape(
            coords.shape[0], self.num_layers - 1, self.hidden_dim
        )
        return sine_network(
            self.coordinates(coords),
            shifts,
            self.hidden_dim,
            self.num_layers,
            self.output_dim,
            self.w0,
            True,
        )
