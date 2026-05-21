import torch
import numpy as np
from typing import List, Callable

from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.metrics.road import ROADCombined

def batch_pearson_coherency(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Computes Pearson correlation for a batch of matrices.
    """
    a = A.reshape(A.shape[0], -1)
    b = B.reshape(B.shape[0], -1)

    a_centered = a - a.mean(axis=1, keepdims=True)
    b_centered = b - b.mean(axis=1, keepdims=True)

    cov = (a_centered * b_centered).sum(axis=1) / (a.shape[1] - 1)
    std_a = np.sqrt((a_centered**2).sum(axis=1) / (a.shape[1] - 1))
    std_b = np.sqrt((b_centered**2).sum(axis=1) / (b.shape[1] - 1))

    eps = 1e-8
    rho = cov / (std_a * std_b + eps)
    return rho

class Complexity:
    """
    Defined as the L1 norm of the attribution map.
    It is a number between 0 and 1, where 0 means that the attribution map is all zeros, and 1 means that the attribution map is all ones.
    
    Paper: https://arxiv.org/abs/2104.10252
    """
    def __call__(self,
                 grayscale_cams: np.ndarray) -> np.ndarray:
        
        # Flatten out the attribution maps (consider first the batch dimension, then the spatial dimensions)
        flattened_cams = grayscale_cams.reshape(grayscale_cams.shape[0], -1)
        
        # Compute the L1 norm of the attribution maps
        l1_norm = np.sum(np.abs(flattened_cams), axis=1)
        
        # Normalize the L1 norm by the number of pixels in the attribution map
        num_pixels = flattened_cams.shape[1]
        complexity = l1_norm / num_pixels
        
        return complexity


class Coherency:
    """
    Defined as the pearson correlation coefficient between:
    - The attribution map,
    - The attribution map computed after masking the input with the previous attribution map.

    Paper: https://arxiv.org/abs/2104.10252
    """
    # def __init__(self, base_method: BaseCAM = None):
    #     if base_method is None:
    #         self.base_method = GradCAMPlusPlus()
    #     self.base_method = base_method
        

    def __call__(self,
                 input_tensor: torch.Tensor,
                 grayscale_cams: np.ndarray,
                 targets: List[Callable],
                 model: torch.nn.Module,
                 base_method: BaseCAM) -> np.ndarray:
        # Mask the input with the previous attribution map
        tensor_cams = torch.from_numpy(grayscale_cams).unsqueeze(1).to(input_tensor.device)
        mixed_images = input_tensor * tensor_cams

        # Compute the attribution map on the mixed images
        mixed_cams = base_method(input_tensor=mixed_images, targets=targets)

        # Compute the pearson correlation coefficient
        pearson = (batch_pearson_coherency(mixed_cams, grayscale_cams) + 1) / 2
        return pearson


class ARCC:
    """
    Defined as the geometric mean of Complexity, Coherency and ROAD.
    
    Paper: https://arxiv.org/abs/2605.14641
    """
    def __init__(self, base_method: BaseCAM, road_percentiles=[20,40,60,80]):
        self.complexity = Complexity()
        self.coherency = Coherency()
        self.road = ROADCombined(percentiles=road_percentiles)
        self.base_method = base_method

    def __call__(self,
                 input_tensor: torch.Tensor,
                 grayscale_cams: np.ndarray,
                 targets: List[Callable],
                 model: torch.nn.Module) -> np.ndarray:
        
        complexity = self.complexity(grayscale_cams)
        coherency = self.coherency(input_tensor, grayscale_cams, targets, model, self.base_method)
        road = self.road(input_tensor, grayscale_cams, targets, model)

        # ROAD is between -0.5 and 0.5, but we need it to be between 0 and 1, so we scale it and clip it.
        road = (road * 2).clip(0, 1)

        arcc = 3 * ((1/(coherency + 1e-8)) + (1/((1 - complexity) + 1e-8)) + (1/(road + 1e-8)))**(-1)
        return arcc