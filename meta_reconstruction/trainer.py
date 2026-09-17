"""Meta-learned reconstruction through a shared latent initialization."""

import jax.numpy as jnp

from meta_learning.trainer import Trainer


class ReconstructionTrainer(Trainer):
    def loss(self, params, batch, key):
        latent = self.adapted(params, batch)
        prediction = self.predictions(params, batch["query_coords"], latent)
        loss = jnp.mean((prediction - batch["query_values"]) ** 2)
        return loss, {"reconstruction": loss}
