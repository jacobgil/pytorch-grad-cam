import numpy as np
import torch
from typing import List, Callable
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import FinerWeightedTarget

class FinerCAM:
    def __init__(self, model: torch.nn.Module, target_layers: List[torch.nn.Module], reshape_transform: Callable = None, base_method=GradCAM):
        self.base_cam = base_method(model, target_layers, reshape_transform)
        self.compute_input_gradient = self.base_cam.compute_input_gradient
        self.uses_gradients = self.base_cam.uses_gradients

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, input_tensor: torch.Tensor, targets: List[torch.nn.Module] = None, eigen_smooth: bool = False,
                alpha: float = 1, comparison_categories: List[int] = [1, 2, 3], target_idx: int = None
                ) -> np.ndarray:
        input_tensor = input_tensor.to(self.base_cam.device)

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        outputs = self.base_cam.activations_and_grads(input_tensor)

        if targets is None:
            output_data = outputs.detach().cpu().numpy()
            target_logits = np.max(output_data, axis=-1) if target_idx is None else output_data[:, target_idx]
            # Sort class indices for each sample based on the absolute difference 
            # between the class scores and the target logit, in ascending order.
            # The most similar classes (smallest difference) appear first.
            sorted_indices = np.argsort(np.abs(output_data - target_logits[:, None]), axis=-1)
            targets = [FinerWeightedTarget(int(sorted_indices[i, 0]), 
                                           [int(sorted_indices[i, idx]) for idx in comparison_categories], 
                                           alpha) 
                       for i in range(output_data.shape[0])]

        if self.uses_gradients:
            self.base_cam.model.zero_grad()
            loss = sum([target(output) for target, output in zip(targets, outputs)])
            if self.base_cam.detach:
                loss.backward(retain_graph=True)
            else:
                # keep the computational graph, create_graph = True is needed for hvp
                torch.autograd.grad(loss, input_tensor, retain_graph = True, create_graph = True)
                # When using the following loss.backward() method, a warning is raised: "UserWarning: Using backward() with create_graph=True will create a reference cycle"
                # loss.backward(retain_graph=True, create_graph=True)
            if 'hpu' in str(self.base_cam.device):
                self.base_cam.__htcore.mark_step()

        cam_per_layer = self.base_cam.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
        return self.base_cam.aggregate_multi_layers(cam_per_layer)