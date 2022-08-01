import numpy as np
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection


class HiResCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(
            HiResCAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform)

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        elementwise_activations = grads * activations

        if eigen_smooth:
            print(
                "Warning: HiResCAM's faithfulness guarantees do not hold if smoothing is applied")
            cam = get_2d_projection(elementwise_activations)
        else:
            cam = elementwise_activations.sum(axis=1)
        return cam
