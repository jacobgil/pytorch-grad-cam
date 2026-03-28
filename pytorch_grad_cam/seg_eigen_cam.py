import numpy as np
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection_with_sign_correction

# Based on this paper:
# https://doi.org/10.3390/app15137562
# Chung, C.-T.; Ying, J.J.-C.
# Seg-Eigen-CAM: Eigen-Value-Based Visual Explanations for Semantic Segmentation Models.
# Applied Sciences, 2025, 15(13), 7562.


class SegEigenCAM(BaseCAM):
    """
    Seg-Eigen-CAM: Eigen-Value-Based Visual Explanations for Semantic
    Segmentation Models.

    Extends Eigen-CAM with two contributions tailored for segmentation:

    1. **Gradient Weighting** (Section 3.2.1): Element-wise product between
       absolute gradients and activations (Eq. 10), providing local pixel-wise
       spatial information instead of a global average.

    2. **Sign Correction** (Section 3.2.2): Dynamically corrects the sign
       ambiguity from SVD by comparing |max| vs |min| of the projection
       (Eq. 13), ensuring salient regions are always positive.

    Reference:
        Chung, C.-T.; Ying, J.J.-C. Seg-Eigen-CAM: Eigen-Value-Based Visual
        Explanations for Semantic Segmentation Models. Appl. Sci. 2025,
        15(13), 7562. https://doi.org/10.3390/app15137562

    Args:
        model: The neural network model to explain.
        target_layers: List of convolutional layers to extract activations from.
        reshape_transform: Optional callable for non-standard activation shapes.
    """

    def __init__(self, model, target_layers, reshape_transform=None):
        super(SegEigenCAM, self).__init__(
            model,
            target_layers,
            reshape_transform,
            uses_gradients=True,
        )

    def get_cam_image(
        self,
        input_tensor,
        target_layer,
        target_category,
        activations,
        grads,
        eigen_smooth,
    ):
        # Step 1 — Gradient Weighting (Eq. 10):
        # |grads| ⊙ activations captures pixel-wise spatial importance,
        # using absolute values to include both positive and negative gradient
        # contributions (unlike Grad-CAM which discards negative gradients).
        weighted_activations = np.abs(grads) * activations

        # Steps 2 & 3 — SVD + Sign Correction (Eq. 11-13)
        return get_2d_projection_with_sign_correction(weighted_activations)