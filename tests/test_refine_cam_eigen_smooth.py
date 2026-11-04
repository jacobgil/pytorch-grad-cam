import torch

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.refine_cam import RefineCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class RecordingGradCAM(GradCAM):
    calls = []

    def __call__(self, input_tensor, targets, aug_smooth=False, eigen_smooth=False):
        RecordingGradCAM.calls.append((aug_smooth, eigen_smooth))
        return super().__call__(input_tensor, targets, aug_smooth, eigen_smooth)


def test_refine_cam_forwards_eigen_smooth():
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(4, 4, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(4, 2),
    )
    input_tensor = torch.rand(1, 3, 16, 16)
    targets = [ClassifierOutputTarget(0)]

    RecordingGradCAM.calls = []
    cam = RefineCAM(model=model,
                    target_layers=[model[0], model[2]],
                    base_method=RecordingGradCAM)
    cam(input_tensor=input_tensor, targets=targets, eigen_smooth=True)

    assert RecordingGradCAM.calls == [(False, True), (False, True)]
