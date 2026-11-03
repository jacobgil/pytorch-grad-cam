import numpy as np
import pytest
import torch
import torch.nn as nn

from pytorch_grad_cam import (
    FEM,
    EigenCAM,
    EigenGradCAM,
    GradCAM,
    GradCAMElementWise,
    GradCAMPlusPlus,
    HiResCAM,
    KPCA_CAM,
    LayerCAM,
    ShapleyCAM,
    XGradCAM,
)

torch.manual_seed(0)


class Tiny1DConvNet(nn.Module):
    def __init__(self, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv1d(1, 4, kernel_size=kernel_size, stride=stride, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


@pytest.mark.parametrize("cam_cls", [
    GradCAM, HiResCAM, GradCAMElementWise, GradCAMPlusPlus, XGradCAM,
    LayerCAM, EigenCAM, EigenGradCAM, KPCA_CAM, FEM, ShapleyCAM])
@pytest.mark.parametrize("stride", [1, 2])
def test_cam_methods_support_1d_conv_layers(cam_cls, stride):
    model = Tiny1DConvNet(stride=stride).eval()
    input_tensor = torch.randn(2, 1, 50)

    with cam_cls(model, [model.conv]) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)

    assert grayscale_cam.shape == (2, 50)
    assert np.all(np.isfinite(grayscale_cam))


def test_grad_cam_1d_highlights_the_part_of_the_signal_the_class_uses():
    # The class 0 logit is the mean of relu(x), so only the bump contributes.
    model = Tiny1DConvNet().eval()
    model.conv = nn.Conv1d(1, 1, kernel_size=1, bias=False)
    model.fc = nn.Linear(1, 2, bias=False)
    with torch.no_grad():
        model.conv.weight.fill_(1.0)
        model.fc.weight.copy_(torch.tensor([[1.0], [-1.0]]))

    input_tensor = torch.zeros(1, 1, 50)
    input_tensor[..., 30:35] = 1.0
    expected = np.zeros(50, dtype=np.float32)
    expected[30:35] = 1.0

    with GradCAM(model, [model.conv]) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)

    np.testing.assert_allclose(grayscale_cam[0], expected, atol=1e-3)
