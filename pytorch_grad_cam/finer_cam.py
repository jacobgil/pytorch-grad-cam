import numpy as np
import torch
from typing import Callable, List, Optional
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import FinerWeightedTarget


DEFAULT_COMPARISON_CATEGORIES = (1, 2, 3)


class FinerCAM:
    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: List[torch.nn.Module],
        reshape_transform: Callable = None,
        base_method=GradCAM,
    ):
        self.base_cam = base_method(model, target_layers, reshape_transform)
        self.compute_input_gradient = self.base_cam.compute_input_gradient
        self.uses_gradients = self.base_cam.uses_gradients

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        # Delegate teardown to the underlying CAM so forward hooks are released
        # when users wrap FinerCAM in a `with` statement, mirroring BaseCAM.
        self.base_cam.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            print(
                f"An exception occurred in CAM with block: {exc_type}. "
                f"Message: {exc_value}"
            )
            return True

    def __del__(self):
        try:
            self.base_cam.activations_and_grads.release()
        except Exception:
            pass

    def forward(
        self,
        input_tensor: torch.Tensor,
        targets: List[torch.nn.Module] = None,
        eigen_smooth: bool = False,
        alpha: float = 1,
        comparison_categories: Optional[List[int]] = None,
        target_idx: int = None,
    ) -> np.ndarray:
        input_tensor = input_tensor.to(self.base_cam.device)

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        outputs = self.base_cam.activations_and_grads(input_tensor)

        if targets is None:
            output_data = outputs.detach().cpu().numpy()
            num_classes = output_data.shape[-1]

            # Default the comparison slice to [1, 2, 3] but clamp to the number of
            # available sorted-by-similarity classes so binary / ternary models
            # don't IndexError. Using a tuple constant avoids the mutable-default
            # argument anti-pattern.
            if comparison_categories is None:
                max_comparisons = max(num_classes - 1, 0)
                comparison_categories = list(DEFAULT_COMPARISON_CATEGORIES[:max_comparisons])
            else:
                comparison_categories = [c for c in comparison_categories if c < num_classes]

            target_logits = (
                np.max(output_data, axis=-1)
                if target_idx is None
                else output_data[:, target_idx]
            )
            # Sort class indices for each sample based on the absolute difference
            # between the class scores and the target logit, in ascending order.
            # The most similar classes (smallest difference) appear first.
            sorted_indices = np.argsort(
                np.abs(output_data - target_logits[:, None]), axis=-1
            )
            targets = [
                FinerWeightedTarget(
                    int(sorted_indices[i, 0]),
                    [int(sorted_indices[i, idx]) for idx in comparison_categories],
                    alpha,
                )
                for i in range(output_data.shape[0])
            ]

        if self.uses_gradients:
            self.base_cam.model.zero_grad()
            loss = sum(target(output) for target, output in zip(targets, outputs))
            if self.base_cam.detach:
                loss.backward(retain_graph=True)
            else:
                # keep the computational graph, create_graph=True is needed for hvp
                torch.autograd.grad(
                    loss, input_tensor, retain_graph=True, create_graph=True
                )
            if "hpu" in str(self.base_cam.device):
                # Access the underscore-prefixed attribute directly; the previous
                # double-underscore access triggered Python name mangling to
                # `_FinerCAM__htcore` and would AttributeError on HPU.
                self.base_cam._htcore.mark_step()

        cam_per_layer = self.base_cam.compute_cam_per_layer(
            input_tensor, targets, eigen_smooth
        )
        return self.base_cam.aggregate_multi_layers(cam_per_layer)
