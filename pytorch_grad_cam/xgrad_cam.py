import cv2
import numpy as np
import torch
from pytorch_grad_cam.base_cam import BaseCAM

class XGradCAM(BaseCAM):
    def __init__(self, model, target_layer, use_cuda=False):
        super(XGradCAM, self).__init__(model, target_layer, use_cuda)

    def get_cam_weights(self, input_tensor, 
                              target_category, 
                              activations, 
                              grads):

        sum_activations = np.sum(activations, axis=(1, 2))
        eps = 1e-7
        weights = grads * activations / (sum_activations[:, None, None] + eps)
        weights = weights.sum(axis =(1, 2))
        return weights