import numpy as np
import torch
import tqdm
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.find_layers import replace_layer_recursive
from pytorch_grad_cam.ablation_layer import AblationLayer

class AblationCAM(BaseCAM):
    def __init__(self,
                 model,
                 target_layers,
                 use_cuda=False,
                 reshape_transform=None,
                 ablation_layer=AblationLayer):
        super(AblationCAM, self).__init__(model, 
                                          target_layers, 
                                          use_cuda,
                                          reshape_transform, 
                                          uses_gradients=False)
        self.ablation_layer = ablation_layer    

    def get_cam_weights(self,
                        input_tensor,
                        target_layers,
                        targets,
                        activations,
                        grads):
        with torch.no_grad():
            outputs = self.model(input_tensor)
            original_scores = np.float32([target(output).cpu().item() for target, output in zip(targets, outputs)])

        ablation_layer = self.ablation_layer(target_layers, indices=[])
        replace_layer_recursive(self.model, target_layers, ablation_layer)

        if hasattr(self, "batch_size"):
            BATCH_SIZE = self.batch_size
        else:
            BATCH_SIZE = 32


        number_of_channels = activations.shape[1]
        weights = []

        with torch.no_grad():
            # Iterate over the input batch
            for target, tensor in zip(targets, input_tensor):
                batch_tensor = tensor.repeat(BATCH_SIZE, 1, 1, 1)
                for i in tqdm.tqdm(range(0, number_of_channels, BATCH_SIZE)):
                    ablation_layer.indices = list(range(i, i + BATCH_SIZE))

                    if i + BATCH_SIZE > number_of_channels:
                        keep = number_of_channels - i
                        batch_tensor = batch_tensor[:keep]
                        ablation_layer.indices = ablation_layer.indices[:keep]
                    score = [target(o).cpu().item() for o in self.model(batch_tensor)]
                    weights.extend(score)

        weights = np.float32(weights)
        weights = weights.reshape(activations.shape[:2])
        original_scores = original_scores[:, None]
        weights = (original_scores - weights) / original_scores

        # Replace the model back to the original state
        replace_layer_recursive(self.model, ablation_layer, target_layers)
        return weights


