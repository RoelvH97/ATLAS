"""Translate supported research checkpoint configurations into release settings."""

from dataclasses import fields
from pathlib import Path

from omegaconf import OmegaConf

from meta_learning.model import ModelFactory


def convert_config(source, task):
    root = Path(__file__).resolve().parents[1] / "configs"
    config = OmegaConf.load(root / "base.yaml")
    del config["defaults"]
    config.pop("hydra", None)
    config.task = task
    config.output_dir = f"runs/{task}"
    model = dict(source["model"])
    kind = {"spatial_token": "atlas", "modulated_siren": "functa"}.get(model["type"], model["type"])
    inactive = {
        "use_mask_token": False,
        "n_global_tokens": 0,
        "nonmixing_self_attn": False,
        "ca_knn_gate": "none",
    }
    for key, default in inactive.items():
        if model.get(key, default) != default:
            raise ValueError(f"Unsupported checkpoint feature: {key}={model[key]}")
    if not model.get("spatial_self_attn", True):
        raise ValueError("Checkpoint requires non-spatial self-attention")
    if model.get("spatial_grid") is not None:
        raise ValueError("Anisotropic SpatialFuncta checkpoints are outside this release")
    if kind == "atlas":
        model["content_query"] = model.get("sa_content_query", False)
        model["sa_film_base"] = model.get("sa_film_base", "content")
        model["decoder_ff"] = model.get("decoder_ff", False)
        model["sa_gaussian_window"] = model.get("sa_gaussian_window")
        if model["sa_gaussian_window"] is None:
            model["sa_gaussian_window"] = model.get("gaussian_window", True)
    if kind in ("atlas", "enf"):
        model["bounded_pose"] = model.get("bounded_pose", False)
    allowed = {field.name for field in fields(ModelFactory.models[kind])}
    config.model = {"type": kind, **{key: value for key, value in model.items() if key in allowed}}
    data = source["data"]
    data_kind = {"cifar10": "cifar", "polynomial_fields": "polynomials"}.get(
        data["type"], data["type"]
    )
    config.data = OmegaConf.load(root / "data" / f"{data_kind}.yaml")
    if task == "reconstruct" and data_kind == "cifar" and kind == "atlas":
        config.evaluation.dtype = "float64"
    for old, new in {
        "img_size": "image_size",
        "img_channels": "channels",
        "batch_size": "batch_size",
        "n_val": "n_val",
        "split_seed": "split_seed",
        "coord_seed": "coord_seed",
        "val_coord_seed": "eval_seed",
    }.items():
        if data.get(old) is not None:
            config.data[new] = data[old]
    training = source["training"]
    config.training = OmegaConf.load(root / "training/maml.yaml")
    mapping = {
        "num_inner_steps": "steps",
        "inner_lr": "inner_lr",
        "outer_lr": "outer_lr",
        "num_epochs": "epochs",
        "seed": "seed",
        "meta_sgd": "meta_sgd",
        "meta_sgd_lr": "meta_lr",
        "meta_sgd_clip": "meta_clip",
        "meta_per_step": "per_step",
        "weight_decay": "weight_decay",
    }
    for old, new in mapping.items():
        if old in training:
            config.training[new] = training[old]
    config.training.meta_sgd = bool(training.get("meta_sgd", False))
    config.training.per_step = bool(training.get("meta_per_step", False))
    config.training.gradient_rule = (
        "stopped_code" if training.get("first_order") is True else "maml"
    )
    config.training.pose_lr = model.get("inner_lr_pose", config.training.inner_lr)
    config.training.content_lr = model.get("inner_lr_ctx", config.training.inner_lr)
    config.training.freeze_init = training.get("freeze_z_init") is True
    clipping = training.get("grad_clip_norm", 1.0)
    config.training.grad_clip = 1.0 if clipping == "auto" else clipping
    config.training.disjoint = training.get("inner_disjoint") is True or task == "segment"
    count = data.get("inner_coords_per_step", data.get("num_coords"))
    if count is not None:
        config.data.inner_coords = int(count)
    config.data.outer_coords = int(data.get("outer_coords", config.data.inner_coords))
    if data_kind == "polynomials":
        p = config.data.polynomials
        for old, new in {
            "max_degree": "degree",
            "n_train": "n_train",
            "n_test": "n_val",
            "n_report": "n_test",
            "gen_seed": "seed",
            "degree_decay": "degree_decay",
            "offset_scale": "offset_scale",
            "target_bound": "target_bound",
            "min_peak_fraction": "min_peak_fraction",
        }.items():
            if old in data:
                p[new] = data[old]
    if task == "classify":
        head = source["classifier"]
        if kind != "atlas" or head["type"] != "pooled_linear":
            raise ValueError("Only ATLAS pooled-head classification checkpoints are included")
        cfg = source["meta_classify"]
        if cfg.get("n_keep", 0) or cfg.get("mask_mode", "bottleneck") != "bottleneck":
            raise ValueError("Masked-decoding classifiers are outside this release")
        for old, new in {
            "num_classes": "num_classes",
            "hidden_dim": "hidden_dim",
            "layer_norm": "layer_norm",
        }.items():
            if old in head:
                config.classification[new] = head[old]
        for old, new in {
            "aug_mask_fraction": "token_dropout",
            "ce_lambda": "weight",
            "ce_normalise": "normalize_losses",
            "head_lr": "learning_rate",
            "head_weight_decay": "weight_decay",
            "head_schedule": "schedule",
            "head_warmup_epochs": "warmup_epochs",
        }.items():
            if old in cfg:
                config.classification[new] = cfg[old]
        config.data.random_flip = bool(cfg.get("random_flip", False))
        config.training.checkpoint_policy = cfg.get("checkpoint_policy", "best")
    if task == "segment":
        cfg = source["meta_segment"]
        config.segmentation.weight = cfg.get("segmentation_lambda", 1.0)
        config.segmentation.normalize_losses = cfg.get("normalise_losses", True)
        config.training.weight_decay = training.get("weight_decay", 0.0001)
    return OmegaConf.to_container(config, resolve=True)
