"""Functa with a global, shift-modulated SIREN latent."""

import flax.linen as nn

from .siren import sine_network


class Functa(nn.Module):
    output_dim: int = 3
    latent_dim: int = 512
    hidden_dim: int = 512
    num_layers: int = 15
    w0: float = 30.0

    def token_features(self, latent):
        return latent.reshape(1, self.latent_dim)

    @nn.compact
    def __call__(self, coords, latent):
        shifts = nn.Dense((self.num_layers - 2) * self.hidden_dim, name="modulation")(latent)
        shifts = shifts.reshape(self.num_layers - 2, self.hidden_dim)
        return sine_network(
            coords, shifts, self.hidden_dim, self.num_layers, self.output_dim, self.w0, False
        )
