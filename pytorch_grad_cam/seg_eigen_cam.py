from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection

# Based in this paper: 
# https://doi.org/10.3390/app15137562
# Chung, C.-T.; Ying, J.J.-C. 
# Seg-Eigen-CAM: Eigen-Value-Based Visual Explanations for Semantic Segmentation Models.


class SegEigenCAM(BaseCAM):
    def __init__(self, model, target_layers, 
                 reshape_transform=None):
        super(SegEigenCAM, self).__init__(model, target_layers,
                                          reshape_transform)

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        return get_2d_projection(grads * activations)
