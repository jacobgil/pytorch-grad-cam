import torch
import numpy as np
from typing import List, Callable
from pytorch_grad_cam.metrics.perturbation_confidence import PerturbationConfidenceMetric


def multiply_tensor_with_cam(input_tensor: torch.Tensor,
                             cam: torch.Tensor):
    """ Multiply an input tensor (after normalization)
        with a pixel attribution map
    """
    return input_tensor * cam


class CamMultImageConfidenceChange(PerturbationConfidenceMetric):
    def __init__(self):
        super(CamMultImageConfidenceChange,
              self).__init__(multiply_tensor_with_cam)


class DropInConfidence(CamMultImageConfidenceChange):
    def __init__(self):
        super(DropInConfidence, self).__init__()

    def __call__(self, *args, **kwargs):
        scores = super(DropInConfidence, self).__call__(*args, **kwargs)
        if isinstance(scores, tuple):
            scores, visualization = scores
            return np.maximum(-scores, 0), visualization
        scores = -scores
        return np.maximum(scores, 0)


class IncreaseInConfidence(CamMultImageConfidenceChange):
    def __init__(self):
        super(IncreaseInConfidence, self).__init__()

    def __call__(self, *args, **kwargs):
        scores = super(IncreaseInConfidence, self).__call__(*args, **kwargs)
        if isinstance(scores, tuple):
            scores, visualization = scores
            return np.float32(scores > 0), visualization
        return np.float32(scores > 0)
