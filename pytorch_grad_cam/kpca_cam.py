from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection_kernel

class KPCA_CAM(BaseCAM):
    def __init__(self, model, target_layers, 
                 reshape_transform=None, kernel='sigmoid', gamma=None):
        super(KPCA_CAM, self).__init__(model,
                                       target_layers,
                                       reshape_transform,
                                       uses_gradients=False)
        self.kernel=kernel
        self.gamma=gamma

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        return get_2d_projection_kernel(activations, self.kernel, self.gamma)
