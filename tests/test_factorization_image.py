import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pytorch_grad_cam.utils.image import show_factorization_on_image


@pytest.mark.parametrize("with_labels", [False, True])
def test_factorization_default_colormap(with_labels):
    image = np.full((64, 80, 3), 0.5, dtype=np.float32)
    explanations = np.zeros((2, 64, 80), dtype=np.float32)
    explanations[0, :, :40] = 0.75
    explanations[1, :, 40:] = 0.5
    labels = ["left", "right"] if with_labels else None
    colors = [np.asarray(plt.get_cmap("gist_rainbow")(value)) for value in (0.0, 0.5)]
    figures = plt.get_fignums()
    expected = show_factorization_on_image(
        image, explanations.copy(), colors=colors, concept_labels=labels
    )
    actual = show_factorization_on_image(
        image, explanations.copy(), concept_labels=labels
    )
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.uint8
    assert actual.shape == (64, 160 if with_labels else 80, 3)
    assert plt.get_fignums() == figures


def test_factorization_explicit_colors():
    image = np.full((4, 6, 3), 0.5, dtype=np.float32)
    explanations = np.ones((1, 4, 6), dtype=np.float32)
    actual = show_factorization_on_image(
        image, explanations, colors=[np.array([1.0, 0.0, 0.0, 1.0])]
    )
    expected = np.broadcast_to(np.array([191, 63, 63], dtype=np.uint8), image.shape)
    np.testing.assert_array_equal(actual, expected)


def test_factorization_model_output_visualization():
    import torch
    from pytorch_grad_cam.feature_factorization.deep_feature_factorization import DeepFeatureFactorization

    model = torch.nn.Sequential(torch.nn.Conv2d(2, 2, 1, bias=False), torch.nn.ReLU()).eval()
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].weight[0, 0, 0, 0] = 1
        model[0].weight[1, 1, 0, 0] = 1
    horizontal = torch.linspace(0, 1, 80).repeat(64, 1)
    vertical = torch.linspace(0, 1, 64)[:, None].repeat(1, 80)
    inputs = torch.stack([horizontal, vertical]).unsqueeze(0)
    factorizer = DeepFeatureFactorization(model, model[1])
    try:
        _, explanations = factorizer(inputs, n_components=2)
        actual = show_factorization_on_image(
            np.full((64, 80, 3), 0.5, dtype=np.float32), explanations[0]
        )
        assert actual.shape == (64, 80, 3)
        assert actual.dtype == np.uint8
    finally:
        factorizer.activations_and_grads.release()
