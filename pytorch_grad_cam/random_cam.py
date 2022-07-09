import numpy as np
from pytorch_grad_cam.base_cam import BaseCAM


class RandomCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(
            RandomCAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        return np.random.uniform(-1, 1, size=(grads.shape[0], grads.shape[1]))
