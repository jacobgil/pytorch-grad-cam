from typing import Callable, List, Optional

import numpy as np
import torch

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import scale_cam_image


class RefineCAM:
    """
    Meta-method for aggregating CAMs from multiple layers.
    
    This methods works very well for denoising early layers attribution maps.
    
    Paper: https://arxiv.org/abs/2605.14641
    """
    def __init__(self, model: torch.nn.Module, target_layers: List[torch.nn.Module], reshape_transform: Callable = None, base_method=GradCAMPlusPlus, **kwargs):
        """
        Any additional kwargs are passed to the `base_method` during initialization.
        This allows you to customize the underlying CAM method (e.g., passing
        `compute_input_gradient=True` or `uses_gradients=False` to the `base_method`).
        """
        self.base_cam_per_layer = [base_method(model, [target_layer], reshape_transform, **kwargs) for target_layer in target_layers]

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.forward(*args, **kwargs)

    def forward(
        self,
        input_tensor: torch.Tensor,
        targets: Optional[List[torch.nn.Module]] = None,
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        # Compute the CAM for each layer
        cam_per_layer = np.array(
            [base_cam(input_tensor, targets, eigen_smooth) for base_cam in self.base_cam_per_layer]
        )
        
        # Make sure that cams are in the range [0, 1]
        for cam in cam_per_layer:
            cam = np.maximum(cam, 0)
            cam = cam / (np.max(cam) + 1e-8)

        # Compute the product of the CAMs across layers
        final_cam = np.prod(cam_per_layer, axis=0)
        return scale_cam_image(final_cam)