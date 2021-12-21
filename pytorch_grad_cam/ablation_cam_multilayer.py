import cv2
import numpy as np
import torch
import tqdm
from pytorch_grad_cam.base_cam import BaseCAM


class AblationLayer(torch.nn.Module):
    def __init__(self, layer, reshape_transform, indices):
        super(AblationLayer, self).__init__()

        self.layer = layer
        self.reshape_transform = reshape_transform
        # The channels to zero out:
        self.indices = indices

    def forward(self, x):
        self.__call__(x)

    def __call__(self, x):
        output = self.layer(x)

        # Hack to work with ViT,
        # Since the activation channels are last and not first like in CNNs
        # Probably should remove it?
        if self.reshape_transform is not None:
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
                ABLATION_VALUE = 1e5
                output[i, self.indices[i], :] = torch.min(
                    output) - ABLATION_VALUE

        if self.reshape_transform is not None:
            output = output.transpose(2, 1)

        return output


def replace_layer_recursive(model, old_layer, new_layer):
    for name, layer in model._modules.items():
        if layer == old_layer:
            model._modules[name] = new_layer
            return True
        elif replace_layer_recursive(layer, old_layer, new_layer):
            return True
    return False


class AblationCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(AblationCAM, self).__init__(model, target_layers, use_cuda,
                                          reshape_transform)

        if len(target_layers) > 1:
            print(
                "Warning. You are usign Ablation CAM with more than 1 layers. "
                "This is supported only if all layers have the same output shape")

    def set_ablation_layers(self):
        self.ablation_layers = []
        for target_layer in self.target_layers:
            ablation_layer = AblationLayer(target_layer,
                                           self.reshape_transform, indices=[])
            self.ablation_layers.append(ablation_layer)
            replace_layer_recursive(self.model, target_layer, ablation_layer)

    def unset_ablation_layers(self):
        # replace the model back to the original state
        for ablation_layer, target_layer in zip(
                self.ablation_layers, self.target_layers):
            replace_layer_recursive(self.model, ablation_layer, target_layer)

    def set_ablation_layer_batch_indices(self, indices):
        for ablation_layer in self.ablation_layers:
            ablation_layer.indices = indices

    def trim_ablation_layer_batch_indices(self, keep):
        for ablation_layer in self.ablation_layers:
            ablation_layer.indices = ablation_layer.indices[:keep]

    def get_cam_weights(self,
                        input_tensor,
                        target_category,
                        activations,
                        grads):
        with torch.no_grad():
            outputs = self.model(input_tensor).cpu().numpy()
            original_scores = []
            for i in range(input_tensor.size(0)):
                original_scores.append(outputs[i, target_category[i]])
        original_scores = np.float32(original_scores)

        self.set_ablation_layers()

        if hasattr(self, "batch_size"):
            BATCH_SIZE = self.batch_size
        else:
            BATCH_SIZE = 32

        number_of_channels = activations.shape[1]
        weights = []

        with torch.no_grad():
            # Iterate over the input batch
            for tensor, category in zip(input_tensor, target_category):
                batch_tensor = tensor.repeat(BATCH_SIZE, 1, 1, 1)
                for i in tqdm.tqdm(range(0, number_of_channels, BATCH_SIZE)):
                    self.set_ablation_layer_batch_indices(
                        list(range(i, i + BATCH_SIZE)))

                    if i + BATCH_SIZE > number_of_channels:
                        keep = number_of_channels - i
                        batch_tensor = batch_tensor[:keep]
                        self.trim_ablation_layer_batch_indices(self, keep)
                    score = self.model(batch_tensor)[:, category].cpu().numpy()
                    weights.extend(score)

        weights = np.float32(weights)
        weights = weights.reshape(activations.shape[:2])
        original_scores = original_scores[:, None]
        weights = (original_scores - weights) / original_scores

        # replace the model back to the original state
        self.unset_ablation_layers()
        return weights
