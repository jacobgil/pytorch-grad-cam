import numpy as np
from pytorch_grad_cam.base_cam import BaseCAM


class XGradCAM(BaseCAM):
    def __init__(
            self,
            model,
            target_layers,
            reshape_transform=None):
        super(
            XGradCAM,
            self).__init__(
            model,
            target_layers,
            reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        spatial_axes = tuple(range(2, activations.ndim))
        sum_activations = np.sum(activations, axis=spatial_axes)
        sum_activations = sum_activations.reshape(
            sum_activations.shape + (1,) * len(spatial_axes))
        eps = 1e-7
        weights = grads * activations / \
            (sum_activations + eps)
        weights = weights.sum(axis=spatial_axes)
        return weights
