"""Joint imagery reconstruction and binary flood segmentation."""

import jax
import jax.numpy as jnp
import optax


def joint_loss(prediction, values, labels, weight=1.0, normalize=True):
    channels = values.shape[-1]
    reconstruction = jnp.mean((prediction[..., :channels] - values) ** 2)
    segmentation = optax.sigmoid_binary_cross_entropy(prediction[..., channels], labels).mean()
    if normalize:
        stop = jax.lax.stop_gradient
        loss = reconstruction / (stop(reconstruction) + 1e-8) + weight * segmentation / (
            stop(segmentation) + 1e-8
        )
    else:
        loss = reconstruction + weight * segmentation
    return loss, {"reconstruction": reconstruction, "segmentation": segmentation}
