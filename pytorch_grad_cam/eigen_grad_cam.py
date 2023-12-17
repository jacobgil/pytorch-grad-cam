from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection

# Like Eigen CAM: https://arxiv.org/abs/2008.00299
# But multiply the activations x gradients


class EigenGradCAM(BaseCAM):
    def __init__(self, model, target_layers, 
                 reshape_transform=None):
        super(EigenGradCAM, self).__init__(model, target_layers,
                                           reshape_transform)

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        return get_2d_projection(grads * activations)
