"""Small visualizations of fitted fields."""

import matplotlib
import numpy as np

matplotlib.use("Agg")


def _panels(images, titles, path):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(images), figsize=(3 * len(images), 3))
    for axis, image, title in zip(axes, images, titles, strict=True):
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def reconstruction(target, prediction, shape, path):
    height, width = map(int, shape)
    target = np.asarray(target).reshape(height, width, -1)
    prediction = np.asarray(prediction).reshape(height, width, -1)
    channels = slice(2, 5) if target.shape[-1] == 8 else slice(None)
    images = [
        np.clip((image[..., channels] + 1) / 2, 0, 1).squeeze() for image in (target, prediction)
    ]
    _panels(images, ("Target", "Reconstruction"), path)


def segmentation(values, labels, logits, shape, path):
    height, width = map(int, shape)
    inputs = np.asarray(values).reshape(height, width, -1)
    images = (
        np.clip((inputs[..., 2:5] + 1) / 2, 0, 1),
        np.clip((inputs[..., 5:8] + 1) / 2, 0, 1),
        np.asarray(labels).reshape(height, width),
        np.asarray(logits).reshape(height, width) >= 0,
    )
    _panels(images, ("Before", "After", "Flood mask", "Prediction"), path)
