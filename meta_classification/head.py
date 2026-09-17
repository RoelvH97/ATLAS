"""Mean-pooled linear classification of adapted features."""

import flax.linen as nn
import jax.numpy as jnp


class PooledClassifier(nn.Module):
    num_classes: int = 10
    hidden_dim: int = 0
    layer_norm: bool = True

    @nn.compact
    def __call__(self, features, token_mask=None):
        if token_mask is None:
            pooled = features.mean(axis=1)
        else:
            keep = token_mask[..., None].astype(features.dtype)
            pooled = (features * keep).sum(axis=1) / jnp.maximum(keep.sum(axis=1), 1)
        if self.layer_norm:
            pooled = nn.LayerNorm(name="pool_norm")(pooled)
        if self.hidden_dim:
            pooled = nn.gelu(nn.Dense(self.hidden_dim, name="hidden")(pooled))
        return nn.Dense(self.num_classes, name="output")(pooled)
