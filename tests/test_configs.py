from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from meta_learning.adaptation import Adaptation
from meta_learning.model import ModelFactory

ROOT = Path(__file__).resolve().parents[1]
RECIPES = [
    (task, f"{prefix}_{model}")
    for task, prefix in [
        ("reconstruct", "cifar_reconstruct"),
        ("classify", "cifar_classify"),
        ("segment", "ombria"),
        ("polynomials", "polynomial"),
    ]
    for model in ("atlas", "enf", "functa", "spatial_functa")
    if not (task == "polynomials" and model == "spatial_functa")
]


@pytest.mark.parametrize("entry,recipe", RECIPES)
def test_recipe_composes_and_constructs(entry, recipe):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        config = OmegaConf.to_container(
            compose(config_name=entry, overrides=[f"experiment={recipe}"]), resolve=True
        )
    Adaptation.from_config(config["training"])
    model = ModelFactory.build(
        config["model"],
        config["data"]["channels"] + (entry == "segment"),
        config["data"]["image_size"],
    )
    assert model.output_dim > 0
