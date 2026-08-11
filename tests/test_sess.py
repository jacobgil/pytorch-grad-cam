import cv2
import numpy as np
import pytest
import torch
import torchvision

from pytorch_grad_cam import (
    EigenCAM,
    GradCAM,
    GradCAMPlusPlus,
    LayerCAM,
    ScoreCAM,
    SESS,
    XGradCAM,
)
from pytorch_grad_cam.sess import sliding_window
from pytorch_grad_cam.utils.image import preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


@pytest.fixture
def numpy_image():
    return cv2.imread("examples/both.png")


def make_input_tensor(numpy_image, batch_size, width, height):
    img = cv2.resize(numpy_image, (width, height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = preprocess_image(img.astype(np.float32) / 255.0)
    return input_tensor.repeat(batch_size, 1, 1, 1)


@pytest.mark.parametrize("batch_size,width,height", [
    (2, 112, 112),
    (1, 224, 160)
])
@pytest.mark.parametrize("base_method", [
    GradCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    XGradCAM,
    EigenCAM,
    LayerCAM
])
def test_sess(numpy_image, batch_size, width, height, base_method):
    input_tensor = make_input_tensor(numpy_image, batch_size, width, height)
    model = torchvision.models.resnet18(weights="DEFAULT")
    model.eval()

    targets = [ClassifierOutputTarget(243) for _ in range(batch_size)]

    with SESS(model=model,
              target_layers=[model.layer4[-1]],
              base_method=base_method,
              num_scales=2) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    assert grayscale_cam.shape[0] == input_tensor.shape[0]
    assert grayscale_cam.shape[1:] == input_tensor.shape[2:]
    assert not np.isnan(grayscale_cam).any()
    assert np.all(grayscale_cam >= 0) and np.all(grayscale_cam <= 1)


@pytest.mark.parametrize("kwargs", [
    {"pool": "max"},
    {"pool": "mean", "theta": 0.5},
    {"smooth": False},
    {"pre_filter_ratio": 0.5},
    {"use_softmax_scores": False},
    {"scales": [224, 256], "window_size": 224, "step_size": 112},
])
def test_sess_options(numpy_image, kwargs):
    input_tensor = make_input_tensor(numpy_image, 1, 224, 224)
    model = torchvision.models.resnet18(weights="DEFAULT")
    model.eval()

    with SESS(model=model,
              target_layers=[model.layer4[-1]],
              num_scales=2,
              **kwargs) as cam:
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=[ClassifierOutputTarget(243)])

    assert grayscale_cam.shape == (1, 224, 224)
    assert not np.isnan(grayscale_cam).any()


def test_sess_without_targets(numpy_image):
    """Without targets the highest scoring category of every image is explained."""
    input_tensor = make_input_tensor(numpy_image, 1, 224, 224)
    model = torchvision.models.resnet18(weights="DEFAULT")
    model.eval()

    with torch.no_grad():
        category = model(input_tensor).argmax(dim=-1).item()

    with SESS(model=model, target_layers=[model.layer4[-1]], num_scales=2) as cam:
        without_targets = cam(input_tensor=input_tensor)
        with_targets = cam(input_tensor=input_tensor,
                           targets=[ClassifierOutputTarget(category)])

    np.testing.assert_allclose(without_targets, with_targets, atol=1e-6)


def test_sess_invalid_arguments():
    model = torchvision.models.resnet18()
    with pytest.raises(ValueError):
        SESS(model=model, target_layers=[model.layer4[-1]], pool="median")
    with pytest.raises(ValueError):
        SESS(model=model, target_layers=[model.layer4[-1]], pre_filter_ratio=1.0)
    with pytest.raises(ValueError):
        SESS(model=model, target_layers=[model.layer4[-1]], num_scales=0)

    input_tensor = torch.zeros(1, 3, 224, 224)
    cam = SESS(model=model, target_layers=[model.layer4[-1]], scales=[128])
    with pytest.raises(ValueError):
        # A scale that is smaller than the window can't be cropped.
        cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(0)])
    cam = SESS(model=model, target_layers=[model.layer4[-1]], num_scales=1)
    with pytest.raises(ValueError):
        # One target per image is expected.
        cam(input_tensor=input_tensor.repeat(2, 1, 1, 1),
            targets=[ClassifierOutputTarget(0)])


@pytest.mark.parametrize("height,width,step_size,window_size,expected", [
    # A single window when the image is exactly the window size.
    (224, 224, 224, 224, [(0, 0)]),
    # Windows crossing the border are shifted back into the image.
    (288, 288, 224, 224, [(0, 0), (64, 0), (0, 64), (64, 64)]),
    (224, 448, 224, 224, [(0, 0), (224, 0)]),
    # Overlapping windows.
    (224, 448, 112, 224, [(0, 0), (112, 0), (224, 0)]),
])
def test_sliding_window(height, width, step_size, window_size, expected):
    assert sliding_window(height, width, step_size, window_size) == expected
