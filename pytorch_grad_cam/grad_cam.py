import cv2
import numpy as np
import torch
from pytorch_grad_cam.base_cam import BaseCAM

class GradCAM(BaseCAM):
    def __init__(self, model, target_layer, use_cuda=False):
        super(GradCAM, self).__init__(model, target_layer, use_cuda)

    def get_cam_weights(self, input_tensor, 
                              target_category, 
                              activations, grads):
        return np.mean(grads, axis=(1, 2))