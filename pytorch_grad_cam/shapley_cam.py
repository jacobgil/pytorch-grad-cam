from typing import Callable, List, Optional, Tuple
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch
import numpy as np

"""
Weights the activation maps using the gradient and Hessian-Vector product.
This method (https://arxiv.org/abs/2501.06261) reinterpret CAM methods (include GradCAM, HiResCAM and the original CAM) from a Shapley value perspective.
"""
class ShapleyCAM(BaseCAM):
    def __init__(self, model, target_layers,
                 reshape_transform=None):
        super(
            ShapleyCAM,
            self).__init__(
            model = model,
            target_layers = target_layers,
            reshape_transform = reshape_transform,
            compute_input_gradient = True,
            uses_gradients = True,
            detach = False)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        
        hvp = torch.autograd.grad(
            outputs=grads,
            inputs=activations,
            grad_outputs=activations,
            retain_graph=False,
            allow_unused=True
        )[0]
        # print(torch.max(hvp[0]).item())  # check if hvp is not all zeros
        if hvp is None:
            hvp = torch.tensor(0).to(self.device)
        else:
            if self.activations_and_grads.reshape_transform is not None:
                hvp = self.activations_and_grads.reshape_transform(hvp)

        if self.activations_and_grads.reshape_transform is not None:
            activations = self.activations_and_grads.reshape_transform(activations)
            grads = self.activations_and_grads.reshape_transform(grads)

        weight = (grads  - 0.5 * hvp).detach().cpu().numpy()
        # 2D image
        if len(activations.shape) == 4:
            weight = np.mean(weight, axis=(2, 3))
            return weight
        # 3D image
        elif len(activations.shape) == 5:
            weight = np.mean(weight, axis=(2, 3, 4))
            return weight
        else:
            raise ValueError("Invalid grads shape."
                             "Shape of grads should be 4 (2D image) or 5 (3D image).")
