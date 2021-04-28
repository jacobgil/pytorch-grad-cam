import cv2
import numpy as np
import torch
from pytorch_grad_cam.base_cam import BaseCAM

# https://arxiv.org/abs/2008.00299
class EigenCAM(BaseCAM):
    def __init__(self, model, target_layer, use_cuda=False, 
        reshape_transform=None):
        super(EigenCAM, self).__init__(model, target_layer, use_cuda, 
            reshape_transform)

    def get_cam_image(self,
                      input_tensor,
                      target_category,
                      activations,
                      grads):
        reshaped_activations = activations.reshape(activations.shape[0], -1).transpose()

        # Centering before the SVD seems to be important here,
        # Otherwise the image returned is negative
        reshaped_activations = reshaped_activations - reshaped_activations.mean(axis=0)
        U, S, VT = np.linalg.svd(reshaped_activations, full_matrices=True)    
        projection = reshaped_activations @ VT[0, :]
        projection = projection.reshape(activations.shape[1 : ])
        return projection