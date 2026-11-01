import pytest
import torch

from pytorch_grad_cam.metrics.road import NoisyLinearImputer


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("weighting,expected_center", [
    (None, 34.0 / 6),
    ([((0, -1), 0.5), ((0, 1), 0.5)], 2.0),
    ([((-1, 0), 0.5), ((1, 0), 0.5)], 15.0),
    ([((0, -1), 0.25), ((0, 1), 0.75)], 2.5),
])
def test_imputer_uses_requested_neighbor_weights(channels, weighting, expected_center):
    image = torch.tensor([[[0.0, 10.0, 0.0], [1.0, 99.0, 3.0], [0.0, 20.0, 0.0]]]).repeat(channels, 1, 1)
    scales = torch.arange(1, channels + 1, dtype=image.dtype)
    image = image * scales[:, None, None]
    mask = torch.ones(3, 3)
    mask[1, 1] = 0
    imputer = NoisyLinearImputer(noise=0) if weighting is None else NoisyLinearImputer(noise=0, weighting=weighting)
    actual = imputer(image, mask)
    expected = image.clone()
    expected[:, 1, 1] = scales * expected_center
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(image[:, 1, 1], scales * 99)


@pytest.mark.parametrize("channels", [1, 2])
def test_imputer_custom_weights_coupled_missing_pixels(channels):
    image = torch.zeros(channels, 3, 4)
    scales = torch.arange(1, channels + 1, dtype=image.dtype)
    image[:, 1, 3] = scales * 9
    mask = torch.ones(3, 4)
    mask[1, 1:3] = 0
    actual = NoisyLinearImputer(noise=0, weighting=[((0, -1), 0.5), ((0, 1), 0.5)])(image, mask)
    expected = image.clone()
    expected[:, 1, 1] = scales * 3
    expected[:, 1, 2] = scales * 6
    torch.testing.assert_close(actual, expected)
