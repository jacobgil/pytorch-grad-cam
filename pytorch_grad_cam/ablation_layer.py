import torch
from collections import OrderedDict
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
import time
import numpy as np
import cv2

class AblationLayerFasterRCNN(torch.nn.Module):
    def __init__(self, layer):
        super(AblationLayerFasterRCNN, self).__init__()
        self.layer = layer

    def activations_to_be_ablated(self, activations, top_percent=1.0):
        index = 0
        indices = []

        t0 = time.time()
        projection = get_2d_projection(activations[None, :])[0, :]
        projection = projection - projection.min()
        projection = projection / projection.max()
        projection = projection > 0.1
        projection = np.float32(projection)
        scores = []
        for channel in activations:
            normalized = channel - channel.min()
            normalized = normalized / channel.max()
            score = (projection*normalized).sum()
            scores.append(score)
        scores = np.float32(scores)
        self.indices = np.argsort(scores)[::-1]
        self.indices = self.indices[: int(len(self.indices) * top_percent)]
        return self.indices

    def set_next_batch(self, input_batch_index, activations, num_channels_to_ablate):
        self.activations = OrderedDict()
        for key, value in activations.items():
            self.activations[key] = value[input_batch_index, :, :, :].clone().unsqueeze(0).repeat(num_channels_to_ablate, 1, 1, 1)


    def __call__(self, x):
        result = self.activations
        #result = self.layer(x)
        layers = {0: '0', 1: '1', 2:'2', 3:'3', 4:'pool'}
        num_channels_to_ablate = result['pool'].size(0)
        for i in range(num_channels_to_ablate):
            pyramid_layer = int(self.indices[i]/256)
            index_in_pyramid_layer = int(self.indices[i] % 256)
            result[layers[pyramid_layer]] [i, index_in_pyramid_layer, :, :]  = -1000
        self.indices = self.indices[num_channels_to_ablate : ]
        return result


class AblationLayer(torch.nn.Module):
    def __init__(self, layer):
        super(AblationLayer, self).__init__()
        self.layer = layer


    def activations_to_be_ablated(self, activations, top_percent=0.05):
        index = 0
        indices = []

        t0 = time.time()
        self.indices = list(range(activations.shape[0]))
        return self.indices

        projection = get_2d_projection(activations[None, :])[0, :]
        projection = projection - projection.min()
        projection = projection / projection.max()
        projection = projection > 0.1
        projection = np.float32(projection)
        scores = []
        for channel in activations:
            normalized = channel - channel.min()
            normalized = normalized / channel.max()
            score = (projection*normalized).sum()
            scores.append(score)
        scores = np.float32(scores)
        self.indices = np.argsort(scores)[::-1]
        self.indices = self.indices[: int(len(self.indices) * top_percent)]
        return self.indices

    def set_next_batch(self, input_batch_index, activations, num_channels_to_ablate):
        self.activations = activations[input_batch_index, :, :, :].clone().unsqueeze(0).repeat(num_channels_to_ablate, 1, 1, 1)

    def __call__(self, x):
        if self.activations is None:
            output = self.layer(x)
        else:
            output = self.activations

        for i in range(output.size(0)):
            # Commonly the minimum activation will be 0,
            # And then it makes sense to zero it out.
            # However depending on the architecture,
            # If the values can be negative, we use very negative values
            # to perform the ablation, deviating from the paper.
            if torch.min(output) == 0:
                output[i, self.indices[i], :] = 0
            else:
                ABLATION_VALUE = 1e7
                output[i, self.indices[i], :] = torch.min(
                    output) - ABLATION_VALUE

        return output

class AblationLayerVit(AblationLayer):
    def __init__(self, layer):
        super(AblationLayerVit, self).__init__(layer)

    def __call__(self, x):
        output = self.layer(x)
        output = output.transpose(1, 2)        
        for i in range(output.size(0)):

            # Commonly the minimum activation will be 0,
            # And then it makes sense to zero it out.
            # However depending on the architecture,
            # If the values can be negative, we use very negative values
            # to perform the ablation, deviating from the paper.
            if torch.min(output) == 0:
                output[i, self.indices[i], :] = 0
            else:
                ABLATION_VALUE = 1e7
                output[i, self.indices[i], :] = torch.min(
                    output) - ABLATION_VALUE

        output = output.transpose(2, 1)

        return output
