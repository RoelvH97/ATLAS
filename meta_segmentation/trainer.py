"""Reconstruction-only adaptation with an outer segmentation objective."""

from meta_learning.trainer import Trainer

from .losses import joint_loss


class SegmentationTrainer(Trainer):
    def loss(self, params, batch, key):
        latent = self.adapted(params, batch)
        prediction = self.predictions(params, batch["query_coords"], latent)
        cfg = self.config["segmentation"]
        return joint_loss(
            prediction,
            batch["query_values"],
            batch["labels"],
            cfg["weight"],
            cfg["normalize_losses"],
        )

    @property
    def selection_metric(self):
        return "pooled_iou"

    def evaluate_sample(self, latent, prediction, batch, meter):
        meter.add_mask(prediction[..., -1], batch["labels"])

    def visualize_sample(self, prediction, batch, shape, path):
        from general_utils.plotting import segmentation

        segmentation(batch["query_values"], batch["labels"], prediction[..., -1], shape, path)
