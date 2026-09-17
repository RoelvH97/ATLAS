"""Training, checkpointing, and evaluation shared by the three tasks."""

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import OmegaConf

from data import DataModule
from general_utils.checkpoints import Checkpoint
from general_utils.metrics import MetricAccumulator, psnr

from .adaptation import Adaptation
from .model import ModelFactory


class Trainer(ABC):
    def __init__(self, config):
        self.config = OmegaConf.to_container(config, resolve=True)
        if HydraConfig.initialized() and HydraConfig.get().mode == RunMode.MULTIRUN:
            self.config["output_dir"] = HydraConfig.get().runtime.output_dir
        loaded = None
        if self.config["checkpoint"]:
            loaded = Checkpoint.load(
                self.config["checkpoint"],
                legacy_config=self.config.get("legacy_config"),
                trusted=self.config["trusted_checkpoint"],
            )
            if loaded.metadata.get("legacy"):
                from general_utils.legacy import convert_config

                loaded.config = convert_config(loaded.config, self.config["task"])
            if loaded.config["task"] != self.config["task"]:
                raise ValueError("Checkpoint task does not match this entry point")
            runtime = {
                key: self.config[key]
                for key in (
                    "mode",
                    "checkpoint",
                    "trusted_checkpoint",
                    "output_dir",
                    "evaluation",
                    "polynomial_evaluation",
                )
            }
            runtime["data"] = {
                key: self.config["data"][key]
                for key in ("root", "download", "limit_train", "limit_eval", "num_workers")
            }
            self.config = OmegaConf.to_container(
                OmegaConf.merge(loaded.config, runtime), resolve=True
            )
        dtype = (
            self.config["evaluation"].get("dtype", "float32")
            if self.config["mode"] == "evaluate"
            else "float32"
        )
        if dtype not in ("float32", "float64"):
            raise ValueError("evaluation.dtype must be float32 or float64")
        jax.config.update("jax_enable_x64", dtype == "float64")
        self.output = Path(self.config["output_dir"]).expanduser().resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.training = self.config["training"]
        self.adaptation = Adaptation.from_config(self.training)
        self.model = ModelFactory.build(
            self.config["model"], self.output_channels, self.config["data"]["image_size"]
        )
        self.data = DataModule(self.config["data"], self.adaptation, self.training["seed"], dtype)
        self.key = jax.random.PRNGKey(self.training["seed"])
        self.params, self.key = ModelFactory.initialize(
            self.model, self.key, jnp.zeros((1, 2)), self.adaptation
        )
        self.initialize_task()
        if loaded:
            loaded.validate(self.params)
            self.params = loaded.params
        self._fit = jax.jit(lambda params, x, y: self.adaptation.fit(params, self.model, x, y))
        self._decode = jax.jit(
            lambda params, x, z: self.model.apply({"params": params["model"]}, x, z)
        )

    @property
    def output_channels(self):
        return self.config["data"]["channels"] + int(self.config["task"] == "segment")

    def initialize_task(self):
        return None

    def adapted(self, params, batch):
        def fit(x, y):
            return self.adaptation.fit(params, self.model, x, y)

        latent = jax.vmap(fit)(batch["support_coords"], batch["support_values"])
        return self.adaptation.outer_latent(latent)

    def predictions(self, params, coords, latent):
        return jax.vmap(lambda x, z: self.model.apply({"params": params["model"]}, x, z))(
            coords, latent
        )

    @abstractmethod
    def loss(self, params, batch, key):
        pass

    def optimizer(self, batches):
        cfg = self.training
        main = (
            optax.adamw(cfg["outer_lr"], weight_decay=cfg["weight_decay"])
            if cfg["weight_decay"]
            else optax.adam(cfg["outer_lr"])
        )
        transforms = {
            "main": (
                optax.chain(optax.clip_by_global_norm(cfg["grad_clip"]), main)
                if cfg["grad_clip"] is not None
                else main
            ),
            "meta": optax.adam(cfg["meta_lr"]),
        }
        labels = {key: "meta" if key == "meta_lr" else "main" for key in self.params}
        if "head" in self.params:
            head = self.config["classification"]
            if head["schedule"] == "cosine":
                warmup = int(head["warmup_epochs"] * batches)
                total = cfg["epochs"] * batches
                if not 0 <= warmup < total:
                    raise ValueError("Classifier warmup must be shorter than training")
                rate = (
                    optax.warmup_cosine_decay_schedule(0, head["learning_rate"], warmup, total)
                    if warmup
                    else optax.cosine_decay_schedule(head["learning_rate"], total)
                )
            else:
                rate = head["learning_rate"]
            head_optimizer = optax.adamw(rate, weight_decay=head["weight_decay"])
            transforms["head"] = (
                optax.chain(optax.clip_by_global_norm(cfg["grad_clip"]), head_optimizer)
                if cfg["grad_clip"] is not None
                else head_optimizer
            )
            labels["head"] = "head"
        if cfg["freeze_init"]:
            transforms["frozen"] = optax.set_to_zero()
            labels["z_init"] = "frozen"
        return optax.multi_transform(transforms, labels)

    @staticmethod
    def device_batch(batch):
        return {
            key: jnp.asarray(np.asarray(value))
            for key, value in batch.items()
            if key not in ("shape", "index")
        }

    def train(self):
        dataset = self.data.dataset("train")
        batches = len(dataset) // self.config["data"]["batch_size"]
        if self.training["max_batches"]:
            batches = min(batches, self.training["max_batches"])
        if batches < 1:
            raise ValueError("Training requires at least one batch")
        optimizer = self.optimizer(batches)
        optimizer_state = optimizer.init(self.params)

        @jax.jit
        def step(params, state, batch, key):
            (loss, metrics), gradient = jax.value_and_grad(self.loss, has_aux=True)(
                params, batch, key
            )
            updates, state = optimizer.update(gradient, state, params)
            params = optax.apply_updates(params, updates)
            if "meta_lr" in params:
                low, high = self.training["meta_clip"]
                params["meta_lr"] = jax.tree.map(
                    lambda value: jnp.clip(value, low, high), params["meta_lr"]
                )
            finite = jnp.isfinite(loss) & jnp.all(
                jnp.stack(
                    [jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves((params, state))]
                )
            )
            return params, state, metrics, finite

        OmegaConf.save(OmegaConf.create(self.config), self.output / "config.yaml")
        policy = self.training["checkpoint_policy"]
        if policy not in ("best", "last"):
            raise ValueError("checkpoint_policy must be best or last")
        best = -np.inf
        for epoch in range(self.training["epochs"]):
            dataset.epoch = epoch
            meter = MetricAccumulator()
            start = time.monotonic()
            for index, batch in enumerate(self.data.loader(dataset)):
                if index >= batches:
                    break
                self.key, key = jax.random.split(self.key)
                self.params, optimizer_state, metrics, finite = step(
                    self.params, optimizer_state, self.device_batch(batch), key
                )
                if not bool(finite):
                    raise FloatingPointError(
                        f"Non-finite update at epoch {epoch + 1}, batch {index + 1}"
                    )
                meter.add(metrics)
            metrics = {
                "epoch": epoch + 1,
                "train": meter.summary(),
                "seconds": time.monotonic() - start,
            }
            validate = (epoch + 1) % self.training["eval_every"] == 0 or epoch + 1 == self.training[
                "epochs"
            ]
            if validate and policy == "best":
                metrics["validation"] = self.evaluate("val", write=False)
                score = metrics["validation"][self.selection_metric]
                if score > best:
                    best = score
                    self.save("best.npz", epoch + 1, metrics)
            self.save("last.npz", epoch + 1, metrics)
            if self.training["save_every"] and (epoch + 1) % self.training["save_every"] == 0:
                self.save(f"epoch_{epoch + 1:03d}.npz", epoch + 1, metrics)
            with (self.output / "history.jsonl").open("a") as handle:
                handle.write(json.dumps(metrics) + "\n")
            print(json.dumps(metrics), flush=True)
        return metrics

    def save(self, name, epoch, metrics):
        Checkpoint(self.params, self.config, {"epoch": epoch, "metrics": metrics}).save(
            self.output / name
        )

    @property
    def selection_metric(self):
        return "psnr"

    def evaluate_sample(self, latent, prediction, batch, meter):
        return None

    def evaluate(self, split=None, write=True):
        split = split or self.config["evaluation"]["split"]
        if split not in ("val", "test"):
            raise ValueError("Evaluation split must be val or test")
        dataset = self.data.dataset(split)
        meter = MetricAccumulator()
        for index, raw in enumerate(self.data.loader(dataset)):
            batch = self.device_batch(raw)
            batch = jax.tree.map(lambda value: value[0], batch)
            latent = self._fit(self.params, batch["support_coords"], batch["support_values"])
            count = len(batch["query_coords"])
            chunk = self.config["evaluation"]["chunk_size"]
            prediction = jnp.concatenate(
                [
                    self._decode(self.params, batch["query_coords"][i : i + chunk], latent)
                    for i in range(0, count, chunk)
                ]
            )
            if not np.isfinite(prediction).all():
                raise FloatingPointError(f"Non-finite evaluation for sample {index}")
            channels = batch["query_values"].shape[-1]
            meter.add({"psnr": psnr(prediction[..., :channels], batch["query_values"])})
            self.evaluate_sample(latent, prediction, batch, meter)
            if (
                write
                and index < self.config["evaluation"]["visuals"]
                and count == int(np.prod(np.asarray(raw["shape"])[0]))
            ):
                self.visualize_sample(
                    prediction,
                    batch,
                    np.asarray(raw["shape"])[0],
                    self.output / f"sample_{index:03d}.png",
                )
        result = {"split": split, "n_samples": len(dataset), **meter.summary()}
        if write:
            (self.output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
        return result

    def visualize_sample(self, prediction, batch, shape, path):
        from general_utils.plotting import reconstruction

        values = batch["query_values"]
        reconstruction(values, prediction[..., : values.shape[-1]], shape, path)

    def run(self):
        if self.config["mode"] == "train":
            return self.train()
        if self.config["mode"] == "evaluate":
            if not self.config["checkpoint"]:
                raise ValueError("Evaluation requires checkpoint=PATH")
            return self.evaluate()
        raise ValueError("mode must be train or evaluate")
