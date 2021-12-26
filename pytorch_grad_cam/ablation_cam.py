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
                 ablation_layer=AblationLayer,
                 batch_size=32):
        super(AblationCAM, self).__init__(model, 
                                          target_layers, 
                                          use_cuda,
                                          reshape_transform, 
                                          uses_gradients=False)
        self.batch_size = batch_size
        self.ablation_layer = ablation_layer    

    def save_activation(self, module, input, output):
        self.activations = output

    def merge_prunned_channels(self, weights, channels, number_of_channels):
        index = 0
        result = []
        for i in range(number_of_channels):
            weight = weights[index]
            index = index + 1
            # if index < len(channels) and channels[index] == i:
            #     weight = weights[index]
            #     index = index + 1
            # else:
            #     weight = 0
            result.append(weight)
        return result


    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        targets,
                        activations,
                        grads):


        handle = target_layer.register_forward_hook(self.save_activation)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            handle.remove()
            original_scores = np.float32([target(output).cpu().item() for target, output in zip(targets, outputs)])
        

        ablation_layer = self.ablation_layer(target_layer)
        replace_layer_recursive(self.model, target_layer, ablation_layer)

        number_of_channels = activations.shape[1]
        weights = []

        with torch.no_grad():
            # Iterate over the input batch
            batch_weights = []
            for batch_index, (target, tensor) in enumerate(zip(targets, input_tensor)):
                batch_tensor = tensor.repeat(self.batch_size, 1, 1, 1)
                #channels_to_ablate = ablation_layer.activations_to_be_ablated(activations[batch_index, :])
                
                channels_to_ablate = list(range(number_of_channels))
                number_channels_to_ablate = len(channels_to_ablate)
                ablation_layer.indices = channels_to_ablate

                print("number_channels_to_ablate", number_channels_to_ablate)

                for i in tqdm.tqdm(range(0, number_channels_to_ablate, self.batch_size)):
                    if i + self.batch_size > number_channels_to_ablate:
                        keep = number_channels_to_ablate - i
                        batch_tensor = batch_tensor[:keep]
                        #ablation_layer.indices = ablation_layer.indices[:keep]

                    ablation_layer.set_next_batch(input_batch_index=batch_index, 
                                                  activations=self.activations,
                                                  num_channels_to_ablate=batch_tensor.size(0))
                    score = [target(o).cpu().item() for o in self.model(batch_tensor)]
                    batch_weights.extend(score)

                #batch_weights = self.merge_prunned_channels(batch_weights, channels_to_ablate, number_of_channels)
                weights.extend(batch_weights)


        weights = np.float32(weights)
        weights = weights.reshape(activations.shape[:2])
        original_scores = original_scores[:, None]
        weights = (original_scores - weights) / original_scores
        print(weights)
        # Replace the model back to the original state
        replace_layer_recursive(self.model, ablation_layer, target_layer)
        return weights


