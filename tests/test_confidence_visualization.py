import numpy as np
import pytest
import torch

from pytorch_grad_cam.metrics.cam_mult_image import (
    DropInConfidence,
    IncreaseInConfidence,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


@pytest.mark.parametrize("metric_cls,return_diff,expected", [
    (DropInConfidence, True, [0.75, 0.0]),
    (DropInConfidence, False, [0.0, 0.25]),
    (IncreaseInConfidence, True, [0.0, 1.0]),
    (IncreaseInConfidence, False, [1.0, 0.0]),
])
@pytest.mark.parametrize("return_visualization", [False, True])
@pytest.mark.parametrize("positional_options", [False, True])
def test_confidence_metric_preserves_visualization(
        metric_cls, return_diff, expected, return_visualization,
        positional_options):
    input_tensor = torch.tensor([1.0, -1.0]).reshape(2, 1, 1, 1)
    cams = np.full((2, 1, 1), 0.25, dtype=np.float32)
    targets = [ClassifierOutputTarget(0), ClassifierOutputTarget(0)]
    model = torch.nn.Flatten(start_dim=1)
    metric = metric_cls()

    if positional_options:
        result = metric(input_tensor, cams, targets, model,
                        return_visualization, return_diff)
    else:
        result = metric(input_tensor, cams, targets, model,
                        return_visualization=return_visualization,
                        return_diff=return_diff)

    if return_visualization:
        scores, visualization = result
        torch.testing.assert_close(
            visualization, torch.tensor([0.25, -0.25]).reshape(2, 1, 1, 1))
    else:
        scores = result
    np.testing.assert_array_equal(scores, np.array(expected, dtype=np.float32))
