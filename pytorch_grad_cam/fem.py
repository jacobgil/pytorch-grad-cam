import numpy as np
from pytorch_grad_cam.base_cam import BaseCAM

"""Feature Explanation Method.
Fuad, K. A. A., Martin, P. E., Giot, R., Bourqui, R., Benois-Pineau, J., & Zemmari, A. (2020, November). Features understanding in 3D CNNS for actions recognition in video. In 2020 Tenth International Conference on Image Processing Theory, Tools and Applications (IPTA) (pp. 1-6). IEEE.
https://hal.science/hal-02963298/document
"""

class FEM(BaseCAM):
    def __init__(self, model, target_layers, 
                 reshape_transform=None, k=2):
        super(FEM, self).__init__(model,
                                       target_layers,
                                       reshape_transform,
                                       uses_gradients=False)
        self.k = k

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        
        
        # 2D image
        if len(activations.shape) == 4:
            axis = (2, 3)
        # 3D image
        elif len(activations.shape) == 5:
            axis = (2, 3, 4)
        else:
            raise ValueError("Invalid activations shape." 
                             "Shape of activations should be 4 (2D image) or 5 (3D image).")
        means = np.mean(activations, axis=axis)
        stds = np.std(activations, axis=axis)
        # k sigma rule:
        # Add extra dimensions to match activations shape
        th = means + self.k * stds
        weights_shape = list(means.shape) + [1] * len(axis)
        th = th.reshape(weights_shape)
        binary_mask = activations > th
        weights = binary_mask.mean(axis=axis)
        return (weights.reshape(weights_shape) * activations).sum(axis=1)