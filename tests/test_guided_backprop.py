import numpy as np
import torch
import torchvision

from pytorch_grad_cam import GuidedBackpropReLUModel


def _gb_model():
    torch.manual_seed(0)
    model = torchvision.models.resnet18(weights=None)
    return GuidedBackpropReLUModel(model=model, device="cpu")


def test_guided_backprop_single_image_shape():
    gb_model = _gb_model()
    input_tensor = torch.randn(1, 3, 64, 64)
    gb = gb_model(input_tensor, target_category=None)
    assert gb.shape == (64, 64, 3)


def test_guided_backprop_repeated_calls_do_not_accumulate():
    gb_model = _gb_model()
    input_tensor = torch.randn(1, 3, 64, 64)

    first = gb_model(input_tensor, target_category=3)
    first_copy = first.copy()
    second = gb_model(input_tensor, target_category=3)

    assert np.allclose(first, first_copy)
    assert np.allclose(first, second)
    assert input_tensor.grad is None
    assert not input_tensor.requires_grad


def test_guided_backprop_non_leaf_input():
    gb_model = _gb_model()
    input_tensor = torch.randn(1, 3, 64, 64)
    non_leaf = input_tensor * torch.ones(1, requires_grad=True)

    gb = gb_model(non_leaf, target_category=3)
    expected = gb_model(input_tensor, target_category=3)
    assert np.allclose(gb, expected)


def test_guided_backprop_batch():
    gb_model = _gb_model()
    input_tensor = torch.randn(2, 3, 64, 64)

    gb = gb_model(input_tensor, target_category=[3, 5])
    assert gb.shape == (2, 64, 64, 3)
    for i, category in enumerate([3, 5]):
        expected = gb_model(input_tensor[i:i + 1], target_category=category)
        assert np.allclose(gb[i], expected, atol=1e-6)

    gb = gb_model(input_tensor, target_category=None)
    assert gb.shape == (2, 64, 64, 3)
