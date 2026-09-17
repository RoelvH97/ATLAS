import numpy as np
import pytest
from PIL import Image

from data.coordinates import balanced_indices, grid
from data.ombria import Ombria
from data.polynomials import PolynomialFields


def test_polynomial_values_and_independent_pools():
    fields = PolynomialFields("train", n_train=7)
    same = PolynomialFields("train", n_train=7)
    validation = PolynomialFields("val", n_val=7)
    np.testing.assert_array_equal(fields.coefficients, same.coefficients)
    assert not np.array_equal(fields.coefficients, validation.coefficients)
    assert fields.basis(fields.coords).shape == (1024, 28)
    for index in range(len(fields)):
        image, _ = fields[index]
        expected = fields.basis(fields.coords) @ fields.coefficients[index]
        np.testing.assert_allclose(image.ravel(), expected, atol=1e-7)
        assert 0.585 - 1e-6 <= np.abs(image).max() <= 0.9 + 1e-6


def test_coordinates_and_balanced_coverage():
    np.testing.assert_array_equal(grid(2), [[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]])
    samples = balanced_indices(np.random.default_rng(0), np.array([2, 4, 6]), 8)
    _, counts = np.unique(samples, return_counts=True)
    assert max(counts) - min(counts) == 1
    with pytest.raises(ValueError):
        balanced_indices(np.random.default_rng(0), np.array([]), 8)


def test_ombria_channel_order_and_mask_agreement(tmp_path):
    for position, (modality, phase, prefix) in enumerate(Ombria.sources):
        directory = tmp_path / modality / "test" / phase
        directory.mkdir(parents=True, exist_ok=True)
        shape = (4, 4, 3) if position in (2, 3) else (4, 4)
        values = np.full(shape, position * 20 if position < 4 else 255, dtype=np.uint8)
        Image.fromarray(values).save(directory / f"{prefix}_1.png")
    dataset = Ombria(tmp_path, "test")
    inputs, labels = dataset[0]
    np.testing.assert_allclose(
        inputs[0, 0], np.array([0, 20, 40, 40, 40, 60, 60, 60]) / 127.5 - 1, atol=1e-7
    )
    assert np.all(labels == 1)
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(
        tmp_path / "OmbriaS2/test/MASK/S2_mask_1.png"
    )
    with pytest.raises(ValueError, match="disagree"):
        dataset[0]
