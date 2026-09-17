"""Joint reconstruction and classification through latent adaptation."""

import jax
import jax.numpy as jnp
import optax

from meta_learning.trainer import Trainer

from .head import PooledClassifier


class ClassificationTrainer(Trainer):
    def initialize_task(self):
        cfg = self.config["classification"]
        self.head = PooledClassifier(cfg["num_classes"], cfg["hidden_dim"], cfg["layer_norm"])
        features = self.features(self.params, self.params["z_init"])[None]
        self.key, key = jax.random.split(self.key)
        self.params["head"] = self.head.init(key, features)["params"]
        self._classify = jax.jit(
            lambda params, z: self.head.apply(
                {"params": params["head"]}, self.features(params, z)[None]
            )[0]
        )

    def features(self, params, latent):
        return self.model.apply(
            {"params": params["model"]}, latent, method=self.model.token_features
        )

    def loss(self, params, batch, key):
        latent = self.adapted(params, batch)
        prediction = self.predictions(params, batch["query_coords"], latent)
        reconstruction = jnp.mean((prediction - batch["query_values"]) ** 2)
        features = jax.vmap(lambda z: self.features(params, z))(latent)
        batch_size, tokens = features.shape[:2]
        keep = max(1, round(tokens * (1 - self.config["classification"]["token_dropout"])))
        order = jnp.argsort(jax.random.uniform(key, (batch_size, tokens)), axis=-1)[:, :keep]
        mask = (
            jnp.zeros((batch_size, tokens), dtype=bool)
            .at[jnp.arange(batch_size)[:, None], order]
            .set(True)
        )
        logits = self.head.apply({"params": params["head"]}, features, mask)
        cross_entropy = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch["labels"]
        ).mean()
        cfg = self.config["classification"]
        stop = jax.lax.stop_gradient
        if cfg["normalize_losses"]:
            loss = reconstruction / stop(reconstruction) + cfg["weight"] * cross_entropy / stop(
                cross_entropy
            )
        else:
            loss = reconstruction + cfg["weight"] * cross_entropy
        return loss, {
            "reconstruction": reconstruction,
            "cross_entropy": cross_entropy,
            "accuracy": jnp.mean(jnp.argmax(logits, axis=-1) == batch["labels"]),
        }

    @property
    def selection_metric(self):
        return "accuracy"

    def evaluate_sample(self, latent, prediction, batch, meter):
        logits = self._classify(self.params, latent)
        meter.add({"accuracy": jnp.argmax(logits) == batch["labels"]})
