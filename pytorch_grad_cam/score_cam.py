import cv2
import numpy as np
import torch
from pytorch_grad_cam.base_cam import BaseCAM

class ScoreCAM(BaseCAM):
    def __init__(self, model, target_layer, use_cuda=False):
        super(ScoreCAM, self).__init__(model, target_layer, use_cuda)

    def get_cam_weights(self, input_tensor, 
                              target_category, 
                              activations, grads):
        with torch.no_grad():
            upsample = torch.nn.UpsamplingBilinear2d(size=input_tensor.shape[2 : ])
            activation_tensor = torch.from_numpy(activations).unsqueeze(0)
            if self.cuda:
                activation_tensor = activation_tensor.cuda()

            upsampled = upsample(activation_tensor)
            upsampled = upsampled[0, ]
            
            maxs = upsampled.view(upsampled.size(0), -1).max(dim=-1)[0]
            mins = upsampled.view(upsampled.size(0), -1).min(dim=-1)[0]
            maxs, mins = maxs[:, None, None], mins[:, None, None]
            upsampled = (upsampled - mins) / (maxs - mins)

            input_tensors = input_tensor*upsampled[:, None, :, :]

            if hasattr(self, "batch_size"):
                BATCH_SIZE = self.batch_size
            else: 
                BATCH_SIZE = 16

            scores = []
            for i in range(0, input_tensors.size(0), BATCH_SIZE):
                batch = input_tensors[i : i + BATCH_SIZE, :]
                outputs = self.model(batch).cpu().numpy()[:, target_category]
                scores.append(outputs)
            scores = torch.from_numpy(np.concatenate(scores))
            weights = torch.nn.Softmax(dim=-1)(scores).numpy()
            return weights
        