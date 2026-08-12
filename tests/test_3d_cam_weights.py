import numpy as np
import pytest
import torch
import torch.nn as nn

from pytorch_grad_cam.grad_cam_plusplus import GradCAMPlusPlus
from pytorch_grad_cam.xgrad_cam import XGradCAM


def test_grad_cam_plusplus_accepts_3d_activations():
    activations = np.ones((1, 2, 3, 4, 5), dtype=np.float32)
    grads = np.ones_like(activations)

    weights = GradCAMPlusPlus.get_cam_weights(
        None, None, None, None, activations, grads)

    assert weights.shape == (1, 2)


def test_xgrad_cam_accepts_3d_activations():
    activations = np.ones((1, 2, 3, 4, 5), dtype=np.float32)
    grads = np.ones_like(activations)

    weights = XGradCAM.get_cam_weights(
        None, None, None, None, activations, grads)

    assert weights.shape == (1, 2)


class Tiny3DConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 2, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(2, 2)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


@pytest.mark.parametrize("cam_cls", [GradCAMPlusPlus, XGradCAM])
def test_gradient_cam_methods_support_3d_conv_layers(cam_cls):
    model = Tiny3DConvNet().eval()
    input_tensor = torch.randn(1, 1, 3, 4, 5)

    with cam_cls(model, [model.conv]) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)

    assert grayscale_cam.shape == (1, 3, 4, 5)
