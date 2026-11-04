import numpy as np
import pytest
import torch

from pytorch_grad_cam import GradCAM, SESS
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class SingleOutputTarget:
    """A scalar target following BaseCAM's per-sample output contract."""
    def __init__(self, category):
        self.category = category

    def __call__(self, output):
        assert output.ndim == 1
        return output[self.category]


@pytest.fixture(params=[False, True], ids=["negative_scores", "positive_scores"])
def model_and_input(request):
    torch.manual_seed(42)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(4, 3),
    ).eval()
    if request.param:
        with torch.no_grad():
            model[-1].bias[1] += 1
    return model, torch.rand(2, 3, 8, 12)


def test_gradcam_accepts_single_output_target(model_and_input):
    model, inputs = model_and_input
    with GradCAM(model, [model[0]]) as cam:
        actual = cam(inputs, [SingleOutputTarget(1)] * len(inputs))
        expected = cam(inputs, [ClassifierOutputTarget(1)] * len(inputs))
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize('batch_size', [1, 3, 32])
@pytest.mark.parametrize('use_softmax_scores', [False, True])
def test_sess_scores_single_output_targets(model_and_input, batch_size, use_softmax_scores):
    model, inputs = model_and_input
    with SESS(model, [model[0]], scales=[8, 12], batch_size=batch_size,
              use_softmax_scores=use_softmax_scores) as cam:
        patches, _ = cam.collect_patches(inputs[:1], 8, 8)
        with torch.no_grad():
            outputs = model(torch.cat(patches))
            if use_softmax_scores:
                outputs = torch.softmax(outputs, dim=-1)
            expected = outputs[:, 1].numpy().clip(min=0)
        actual = cam.get_scores(patches, SingleOutputTarget(1))
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        assert actual.shape == (len(patches),)


@pytest.mark.parametrize('batch_size', [1, 3, 32])
@pytest.mark.parametrize('use_softmax_scores', [False, True])
def test_sess_single_output_targets(model_and_input, batch_size, use_softmax_scores):
    model, inputs = model_and_input
    with SESS(model, [model[0]], scales=[8, 12], batch_size=batch_size,
              use_softmax_scores=use_softmax_scores, smooth=False) as cam:
        expected = cam(inputs, [ClassifierOutputTarget(1)] * len(inputs))
        actual = cam(inputs, [SingleOutputTarget(1)] * len(inputs))
    assert actual.shape == (2, 8, 12)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=1e-6)
