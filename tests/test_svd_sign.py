import numpy as np
import pytest
import torch
import torch.nn as nn

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection

_real_svd = np.linalg.svd


def _svd_with_flipped_first_component(a, *args, **kwargs):
    """The other valid SVD: negating the first left and right singular vectors
    gives the same decomposition, and LAPACK builds may return either one."""
    u, s, vt = _real_svd(a, *args, **kwargs)
    u, vt = u.copy(), vt.copy()
    u[:, 0] *= -1
    vt[0, :] *= -1
    return u, s, vt


def _blob_activations(seed, channels=16, size=14):
    """Non-negative activations with one bright region, like a ReLU layer on an object."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    cy, cx = rng.integers(3, size - 3, 2)
    blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / 8.0))
    noise = rng.random((channels, size, size)) * 0.2
    return (noise + blob * rng.random((channels, 1, 1))).astype(np.float32)


class _TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
        self.conv2 = nn.Conv2d(6, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 5)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return self.fc(self.pool(x).flatten(1))


def test_projection_does_not_depend_on_the_sign_the_svd_returns(monkeypatch):
    batch = np.stack([_blob_activations(seed) for seed in range(8)])
    expected = get_2d_projection(batch)

    monkeypatch.setattr(np.linalg, "svd", _svd_with_flipped_first_component)
    actual = get_2d_projection(batch)

    np.testing.assert_allclose(actual, expected, atol=1e-5)


@pytest.mark.parametrize("seed", range(30))
def test_projection_agrees_with_the_channel_sum_map(seed):
    activations = _blob_activations(seed)
    projection = get_2d_projection(activations[None])[0]
    channel_sum = activations.sum(axis=0)

    correlation = np.corrcoef(projection.ravel(), channel_sum.ravel())[0, 1]
    assert correlation > 0


def test_projection_agrees_with_the_channel_sum_map_for_3d_activations():
    rng = np.random.default_rng(0)
    zz, yy, xx = np.mgrid[:6, :8, :8]
    blob = np.exp(-(((zz - 2) ** 2 + (yy - 3) ** 2 + (xx - 5) ** 2) / 6.0))
    activations = (rng.random((10, 6, 8, 8)) * 0.2 + blob * rng.random((10, 1, 1, 1))).astype(np.float32)

    projection = get_2d_projection(activations[None])[0]

    assert projection.shape == (6, 8, 8)
    assert np.corrcoef(projection.ravel(), activations.sum(axis=0).ravel())[0, 1] > 0


def test_eigen_smooth_cam_does_not_depend_on_the_sign_the_svd_returns(monkeypatch):
    torch.manual_seed(0)
    model = _TinyCNN().eval()
    images = torch.randn(3, 3, 16, 16)
    targets = [ClassifierOutputTarget(category) for category in (1, 3, 0)]
    cam = GradCAM(model=model, target_layers=[model.conv2])

    expected = cam(input_tensor=images, targets=targets, eigen_smooth=True)
    monkeypatch.setattr(np.linalg, "svd", _svd_with_flipped_first_component)
    actual = cam(input_tensor=images, targets=targets, eigen_smooth=True)

    np.testing.assert_allclose(actual, expected, atol=1e-5)
