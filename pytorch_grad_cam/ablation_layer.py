import torch
from collections import OrderedDict
import numpy as np
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection


class AblationLayer(torch.nn.Module):
    def __init__(self):
        super(AblationLayer, self).__init__()

    def objectiveness_mask_from_svd(self, activations, threshold=0.01):
        """ Experimental method to get a binary mask to compare if the activation is worth ablating.
            The idea is to apply the EigenCAM method by doing PCA on the activations.
            Then we create a binary mask by comparing to a low threshold.
            Areas that are masked out, are probably not interesting anyway.
        """

        projection = get_2d_projection(activations[None, :])[0, :]
        projection = np.abs(projection)
        projection = projection - projection.min()
        projection = projection / projection.max()
        projection = projection > threshold
        return projection

    def activations_to_be_ablated(
            self,
            activations,
            ratio_channels_to_ablate=1.0):
        """ Experimental method to get a binary mask to compare if the activation is worth ablating.
            Create a binary CAM mask with objectiveness_mask_from_svd.
            Score each Activation channel, by seeing how much of its values are inside the mask.
            Then keep the top channels.

        """
        if ratio_channels_to_ablate == 1.0:
            self.indices = np.int32(range(activations.shape[0]))
            return self.indices

        projection = self.objectiveness_mask_from_svd(activations)

        scores = []
        for channel in activations:
            normalized = np.abs(channel)
            normalized = normalized - normalized.min()
            normalized = normalized / np.max(normalized)
            score = (projection * normalized).sum() / normalized.sum()
            scores.append(score)
        scores = np.float32(scores)

        indices = list(np.argsort(scores))
        high_score_indices = indices[::-
                                     1][: int(len(indices) *
                                              ratio_channels_to_ablate)]
        low_score_indices = indices[: int(
            len(indices) * ratio_channels_to_ablate)]
        self.indices = np.int32(high_score_indices + low_score_indices)
        return self.indices

    def set_next_batch(
            self,
            input_batch_index,
            activations,
            num_channels_to_ablate):
        """ This creates the next batch of activations from the layer.
            Just take corresponding batch member from activations, and repeat it num_channels_to_ablate times.
        """
        self.activations = activations[input_batch_index, :, :, :].clone(
        ).unsqueeze(0).repeat(num_channels_to_ablate, 1, 1, 1)

    def __call__(self, x):
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
    def __init__(self):
        super(AblationLayerVit, self).__init__()

    def __call__(self, x):
        output = self.activations
        output = output.transpose(1, len(output.shape) - 1)
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

        output = output.transpose(len(output.shape) - 1, 1)

        return output

    def set_next_batch(
            self,
            input_batch_index,
            activations,
            num_channels_to_ablate):
        """ This creates the next batch of activations from the layer.
            Just take corresponding batch member from activations, and repeat it num_channels_to_ablate times.
        """
        repeat_params = [num_channels_to_ablate] + \
            len(activations.shape[:-1]) * [1]
        self.activations = activations[input_batch_index, :, :].clone(
        ).unsqueeze(0).repeat(*repeat_params)


class AblationLayerFasterRCNN(AblationLayer):
    def __init__(self):
        super(AblationLayerFasterRCNN, self).__init__()

    def set_next_batch(
            self,
            input_batch_index,
            activations,
            num_channels_to_ablate):
        """ Extract the next batch member from activations,
            and repeat it num_channels_to_ablate times.
        """
        self.activations = OrderedDict()
        for key, value in activations.items():
            fpn_activation = value[input_batch_index,
                                   :, :, :].clone().unsqueeze(0)
            self.activations[key] = fpn_activation.repeat(
                num_channels_to_ablate, 1, 1, 1)

    def __call__(self, x):
        result = self.activations
        layers = {0: '0', 1: '1', 2: '2', 3: '3', 4: 'pool'}
        num_channels_to_ablate = result['pool'].size(0)
        for i in range(num_channels_to_ablate):
            pyramid_layer = int(self.indices[i] / 256)
            index_in_pyramid_layer = int(self.indices[i] % 256)
            result[layers[pyramid_layer]][i,
                                          index_in_pyramid_layer, :, :] = -1000
        return result
