import pytest
import torchvision
import torch
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

torch.manual_seed(0)


@pytest.fixture
def numpy_image():
    return cv2.imread("examples/both.png")


@pytest.mark.parametrize("cam_method",
                         [GradCAM])
def test_memory_usage_in_loop(numpy_image, cam_method):
    model = torchvision.models.resnet18(pretrained=False)
    model.conv1 = torch.nn.Conv2d(
        1, 64, kernel_size=(
            7, 7), stride=(
            2, 2), padding=(
                3, 3), bias=False)
    target_layers = [model.layer4]
    gray_img = numpy_image[:, :, 0]
    input_tensor = torch.from_numpy(
        np.float32(gray_img)).unsqueeze(0).unsqueeze(0)
    input_tensor = input_tensor.repeat(16, 1, 1, 1)
    targets = None
    with cam_method(model=model,
                    target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets)
        print(grayscale_cam.shape)
