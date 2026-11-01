import numpy as np
import pytest
import torch

from pytorch_grad_cam.metrics.cam_mult_image import multiply_tensor_with_cam
from pytorch_grad_cam.metrics.perturbation_confidence import RemoveLeastRelevantFirst, RemoveMostRelevantFirst


@pytest.mark.parametrize("method", [RemoveMostRelevantFirst, RemoveLeastRelevantFirst])
@pytest.mark.parametrize("percentile", ["auto", 50])
@pytest.mark.parametrize("channels", [1, 3])
def test_removal_mask_direction_and_scale(method, percentile, channels):
    image = torch.tensor([[[1.0, 2.0], [10.0, 20.0]]]).repeat(channels, 1, 1)
    cam = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    expected_mask = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    if method is RemoveLeastRelevantFirst:
        expected_mask = 1 - expected_mask
    actual = method(percentile, multiply_tensor_with_cam)(image, cam)
    torch.testing.assert_close(actual, image * expected_mask)
    assert actual.dtype == image.dtype


@pytest.mark.parametrize("least_relevant", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_road_auto_matches_explicit_binary_imputation(least_relevant, batch_size):
    from pytorch_grad_cam.metrics.road import NoisyLinearImputer, ROADLeastRelevantFirst, ROADMostRelevantFirst
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    inputs = torch.tensor([[[[1.0, 2.0], [10.0, 20.0]]]]).repeat(batch_size, 1, 1, 1)
    cams = np.broadcast_to(np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32), (batch_size, 2, 2)).copy()
    model = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1, bias=False), torch.nn.Flatten(1), torch.nn.Linear(4, 1, bias=False))
    with torch.no_grad():
        model[0].weight.fill_(1)
        model[2].weight.fill_(1)
    mask = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    metric_class = ROADLeastRelevantFirst if least_relevant else ROADMostRelevantFirst
    if least_relevant:
        mask = 1 - mask
    imputer = NoisyLinearImputer(noise=0)
    explicit_inputs = torch.stack([imputer(sample, mask) for sample in inputs])
    with torch.no_grad():
        expected = (model(explicit_inputs) - model(inputs)).numpy().reshape(-1)
    metric = metric_class(percentile="auto")
    metric.perturbation.imputer.noise = 0
    actual = metric(inputs, cams, [ClassifierOutputTarget(0)] * batch_size, model)
    np.testing.assert_allclose(actual, expected)
