from typing import Callable, List, Optional, Tuple
from pytorch_grad_cam.base_cam import BaseCAM
import torch
import numpy as np

"""
Weighting the activation maps using Gradient and Hessian-Vector Product.
This method (https://arxiv.org/abs/2501.06261) reinterpret CAM methods (include GradCAM, HiResCAM and the original CAM) from a Shapley value perspective.
"""
class ShapleyCAM(BaseCAM):
    def __init__(self, model, target_layers,
                 reshape_transform=None, detach=False):
        super(
            ShapleyCAM,
            self).__init__(
            model,
            target_layers,
            reshape_transform,
            detach = detach)

    def forward(
        self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool = False
    ) -> np.ndarray:
        input_tensor = input_tensor.to(self.device)

        input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        self.outputs = outputs = self.activations_and_grads(input_tensor)

        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output) for target, output in zip(targets, outputs)])
            # keep the graph
            torch.autograd.grad(loss, input_tensor,  retain_graph = True, create_graph = True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)


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
