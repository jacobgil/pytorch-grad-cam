import cv2
import numpy as np
import torch
from pytorch_grad_cam.base_cam import BaseCAM

class XGradCAM(BaseCAM):
    def __init__(self, model, target_layer, use_cuda=False, reshape_transform=None):
        super(XGradCAM, self).__init__(model, target_layer, use_cuda, reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_category,
                        activations,
                        grads):
        sum_activations = np.sum(activations, axis=(2, 3))
        eps = 1e-7
        weights = grads * activations / (sum_activations[:, :, None, None] + eps)
        weights = weights.sum(axis=(2, 3))
        return weights