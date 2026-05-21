import pytest
import torchvision
import cv2
import numpy as np

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM
from pytorch_grad_cam.refine_cam import RefineCAM
from pytorch_grad_cam.metrics.arcc import ARCC
from pytorch_grad_cam.utils.image import preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


@pytest.fixture
def numpy_image():
    return cv2.imread("examples/both.png")


@pytest.mark.parametrize("cnn_model,target_layer_names", [
    (torchvision.models.resnet18, ["layer4[-1]", "layer4[-2]"])
])
@pytest.mark.parametrize("batch_size,width,height", [
    (2, 112, 112),
    (1, 224, 224)
])
@pytest.mark.parametrize("base_method", [
    GradCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    XGradCAM,
    EigenCAM,
    EigenGradCAM,
    LayerCAM
])
def test_refine_cam(numpy_image, batch_size, width, height,
                    cnn_model, target_layer_names, base_method):
    img = cv2.resize(numpy_image, (width, height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float = img.astype(np.float32) / 255.0
    input_tensor = preprocess_image(img_float)
    input_tensor = input_tensor.repeat(batch_size, 1, 1, 1)

    model = cnn_model(weights="DEFAULT")
    model.eval()
    
    target_layers = []
    for layer in target_layer_names:
        target_layers.append(eval(f"model.{layer}"))

    targets = [ClassifierOutputTarget(243) for _ in range(batch_size)]

    cam = RefineCAM(model=model,
                    target_layers=target_layers,
                    base_method=base_method)

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    assert grayscale_cam is not None
    assert grayscale_cam.shape[0] == input_tensor.shape[0]
    assert grayscale_cam.shape[1:] == input_tensor.shape[2:]

    metric = ARCC(base_method=cam)
    arcc_scores = metric(input_tensor=input_tensor,
                         grayscale_cams=grayscale_cam,
                         targets=targets,
                         model=model)
    
    assert len(arcc_scores) == batch_size
    assert np.all(arcc_scores >= 0)
    assert np.all(arcc_scores <= 1)
