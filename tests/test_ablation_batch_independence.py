import numpy as np
import pytest
import torch

from pytorch_grad_cam import AblationCAM
from pytorch_grad_cam.ablation_layer import AblationLayer, AblationLayerVit
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


@pytest.mark.parametrize("layer_class", [AblationLayer, AblationLayerVit])
@pytest.mark.parametrize("minimum, baseline", [(-4, -10000004), (0, 0), (4, -9999996)])
@pytest.mark.parametrize("batch_size", [2, 3])
def test_ablation_is_independent_of_batch_size(layer_class, minimum, baseline, batch_size):
    activations = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2) + minimum
    if layer_class is AblationLayerVit:
        activations = activations.flatten(2).transpose(1, 2)
    indices = [2, 0, 1]
    layer = layer_class()
    results = []

    for size in [1, batch_size]:
        batches = []
        for start in range(0, len(indices), size):
            layer.indices = indices[start:start + size]
            layer.set_next_batch(0, activations, len(layer.indices))
            batches.append(layer(activations))
        results.append(torch.cat(batches))

    expected = activations.repeat(3, *([1] * (activations.ndim - 1)))
    for row, channel in enumerate(indices):
        if layer_class is AblationLayerVit:
            expected[row, :, channel] = baseline
        else:
            expected[row, channel, :] = baseline
    assert torch.equal(results[0], expected)
    assert torch.equal(results[1], results[0])


@pytest.mark.parametrize("batch_size", [2, 3])
def test_ablation_cam_weights_are_independent_of_batch_size(batch_size):
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 3, 1),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(3, 1),
    ).eval()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([1., 2., 3.]).reshape(3, 1, 1, 1))
        model[0].bias.copy_(torch.tensor([-2., 1., 3.]))
        model[3].weight.copy_(torch.tensor([[1., 2., 3.]]))
        model[3].bias.fill_(100.)
    input_tensor = torch.tensor([[[[-1., 2.], [3., 4.]]]])
    targets = [ClassifierOutputTarget(0)]
    with torch.no_grad():
        activations = model[0](input_tensor).numpy()
    weights = []
    maps = []

    for size in [1, batch_size]:
        with AblationCAM(model, [model[0]], ablation_layer=AblationLayer(), batch_size=size) as cam:
            weights.append(cam.get_cam_weights(input_tensor, model[0], targets, activations, None))
            maps.append(cam(input_tensor, targets))

    np.testing.assert_allclose(weights[1], weights[0])
    np.testing.assert_allclose(maps[1], maps[0])
