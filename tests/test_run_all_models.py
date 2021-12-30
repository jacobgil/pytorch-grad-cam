import pytest
import torchvision
import torch
import cv2
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


@pytest.fixture
def numpy_image():
    return cv2.imread("examples/both.png")


@pytest.mark.parametrize("cnn_model,target_layer_names", [
    (torchvision.models.resnet18, ["layer4[-1]", "layer4[-2]"]),
    (torchvision.models.vgg11, ["features[-1]"])
])
@pytest.mark.parametrize("batch_size,width,height", [
    (2, 32, 32),
    (1, 32, 40)
])
@pytest.mark.parametrize("target_category", [
    None,
    100
])
@pytest.mark.parametrize("aug_smooth", [
    False
])
@pytest.mark.parametrize("eigen_smooth", [
    True,
    False
])
@pytest.mark.parametrize("cam_method",
                         [ScoreCAM,
                          AblationCAM,
                          GradCAM,
                          ScoreCAM,
                          GradCAMPlusPlus,
                          XGradCAM,
                          EigenCAM,
                          EigenGradCAM,
                          LayerCAM,
                          FullGrad])
def test_all_cam_models_can_run(numpy_image, batch_size, width, height,
                                cnn_model, target_layer_names, cam_method,
                                target_category, aug_smooth, eigen_smooth):
    img = cv2.resize(numpy_image, (width, height))
    input_tensor = preprocess_image(img)
    input_tensor = input_tensor.repeat(batch_size, 1, 1, 1)

    model = cnn_model(pretrained=True)
    target_layers = []
    for layer in target_layer_names:
        target_layers.append(eval(f"model.{layer}"))

    cam = cam_method(model=model,
                     target_layers=target_layers,
                     use_cuda=False)
    cam.batch_size = 4
    if target_category is None:
      targets = None
    else:
      targets = [ClassifierOutputTarget(target_category) for _ in range(batch_size)]

    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets,
                        aug_smooth=aug_smooth,
                        eigen_smooth=eigen_smooth)
    assert(grayscale_cam.shape[0] == input_tensor.shape[0])
    assert(grayscale_cam.shape[1:] == input_tensor.shape[2:])
