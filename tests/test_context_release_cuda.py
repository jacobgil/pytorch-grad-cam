import pytest
import torchvision
import torch
import cv2
import psutil
import pytorch_grad_cam
print("pytorch_grad_cam loaded from:", pytorch_grad_cam.__file__)
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad, \
    ShapleyCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

torch.manual_seed(0)


@pytest.fixture
def numpy_image():
    return cv2.imread("examples/both.png")


@pytest.mark.parametrize("cnn_model,target_layer_names", [
    (torchvision.models.resnet18, ["layer4[-1]"])
])
@pytest.mark.parametrize("batch_size,width,height", [
    (1, 224, 224)
])
@pytest.mark.parametrize("target_category", [
    100
])
@pytest.mark.parametrize("aug_smooth", [
    False
])
@pytest.mark.parametrize("eigen_smooth", [
    False
])
@pytest.mark.parametrize("cam_method",
                         [GradCAM, ShapleyCAM])
def test_memory_usage_in_loop(numpy_image, batch_size, width, height,
                              cnn_model, target_layer_names, cam_method,
                              target_category, aug_smooth, eigen_smooth):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        print("CUDA not available")
        return
    img = cv2.resize(numpy_image, (width, height))
    input_tensor = preprocess_image(img)
    input_tensor = input_tensor.repeat(batch_size, 1, 1, 1).to(device)
    model = cnn_model(pretrained=True).to(device)
    target_layers = []
    for layer in target_layer_names:
        target_layers.append(eval(f"model.{layer}"))
    targets = [ClassifierOutputTarget(target_category)
               for _ in range(batch_size)]
    initial_memory = 0
    for i in range(100):
        with cam_method(model=model,
                        target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=targets,
                                aug_smooth=aug_smooth,
                                eigen_smooth=eigen_smooth)

            if i == 0:
                initial_memory = torch.cuda.memory_allocated()
        assert(torch.cuda.memory_allocated() <= initial_memory * 1.5)
