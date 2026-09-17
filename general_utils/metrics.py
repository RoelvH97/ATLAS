"""Reconstruction and foreground-segmentation metrics."""

import jax.numpy as jnp
import numpy as np


def psnr(prediction, target):
    error = jnp.mean(((jnp.clip(prediction, -1, 1) - target) / 2) ** 2)
    return -10 * jnp.log10(error + 1e-8)


class MetricAccumulator:
    def __init__(self):
        self.values = {}
        self.intersection = 0
        self.union = 0

    def add(self, metrics):
        for name, value in metrics.items():
            self.values.setdefault(name, []).append(float(value))

    def add_mask(self, logits, target):
        predicted, target = np.asarray(logits) >= 0, np.asarray(target) >= 0.5
        intersection = int(np.count_nonzero(predicted & target))
        union = int(np.count_nonzero(predicted | target))
        self.intersection += intersection
        self.union += union
        self.add({"image_iou": intersection / union if union else 1.0})

    def summary(self):
        result = {name: float(np.mean(values)) for name, values in self.values.items()}
        if "image_iou" in result:
            result.update(
                pooled_iou=self.intersection / self.union if self.union else 1.0,
                image_iou_sd=float(np.std(self.values["image_iou"])),
                intersection=self.intersection,
                union=self.union,
            )
        return result
