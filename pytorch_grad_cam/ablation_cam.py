import numpy as np
import torch
import tqdm
from typing import Callable, List
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.find_layers import replace_layer_recursive
from pytorch_grad_cam.ablation_layer import AblationLayer


""" Implementation of AblationCAM
https://openaccess.thecvf.com/content_WACV_2020/papers/Desai_Ablation-CAM_Visual_Explanations_for_Deep_Convolutional_Network_via_Gradient-free_Localization_WACV_2020_paper.pdf

Ablate individual activations, and then measure the drop in the target score.

In the current implementation, the target layer activations is cached, so it won't be re-computed.
However layers before it, if any, will not be cached.
This means that if the target layer is a large block, for example model.featuers (in vgg), there will
be a large save in run time.

Since we have to go over many channels and ablate them, and every channel ablation requires a forward pass,
it would be nice if we could avoid doing that for channels that won't contribute anwyay, making it much faster.
The parameter ratio_channels_to_ablate controls how many channels should be ablated, using an experimental method
(to be improved). The default 1.0 value means that all channels will be ablated.
"""


class AblationCAM(BaseCAM):
    def __init__(self,
                 model: torch.nn.Module,
                 target_layers: List[torch.nn.Module],
                 use_cuda: bool = False,
                 reshape_transform: Callable = None,
                 ablation_layer: torch.nn.Module = AblationLayer(),
                 batch_size: int = 32,
                 ratio_channels_to_ablate: float = 1.0) -> None:

        super(AblationCAM, self).__init__(model,
                                          target_layers,
                                          use_cuda,
                                          reshape_transform,
                                          uses_gradients=False)
        self.batch_size = batch_size
        self.ablation_layer = ablation_layer
        self.ratio_channels_to_ablate = ratio_channels_to_ablate

    def save_activation(self, module, input, output) -> None:
        """ Helper function to save the raw activations from the target layer """
        self.activations = output

    def assemble_ablation_scores(self,
                                 new_scores: list,
                                 original_score: float ,
                                 ablated_channels: np.ndarray,
                                 number_of_channels: int) -> np.ndarray:
        """ Take the value from the channels that were ablated,
            and just set the original score for the channels that were skipped """

        index = 0
        result = []
        sorted_indices = np.argsort(ablated_channels)
        ablated_channels = ablated_channels[sorted_indices]
        new_scores = np.float32(new_scores)[sorted_indices]

        for i in range(number_of_channels):
            if index < len(ablated_channels) and ablated_channels[index] == i:
                weight = new_scores[index]
                index = index + 1
            else:
                weight = original_score
            result.append(weight)

        return result

    def get_cam_weights(self,
                        input_tensor: torch.Tensor,
                        target_layer: torch.nn.Module,
                        targets: List[Callable],
                        activations: torch.Tensor,
                        grads: torch.Tensor) -> np.ndarray:
        
        # Do a forward pass, compute the target scores, and cache the activations
        handle = target_layer.register_forward_hook(self.save_activation)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            handle.remove()
            original_scores = np.float32([target(output).cpu().item() for target, output in zip(targets, outputs)])

        # Replace the layer with the ablation layer.
        # When we finish, we will replace it back, so the original model is unchanged.
        ablation_layer = self.ablation_layer
        replace_layer_recursive(self.model, target_layer, ablation_layer)

        number_of_channels = activations.shape[1]
        weights = []
        # This is a "gradient free" method, so we don't need gradients here.
        with torch.no_grad():
            # Loop over each of the batch images and ablate activations for it.
            for batch_index, (target, tensor) in enumerate(zip(targets, input_tensor)):
                new_scores = []
                batch_tensor = tensor.repeat(self.batch_size, 1, 1, 1)

                # Check which channels should be ablated. Normally this will be all channels,
                # But we can also try to speed this up by using a low ratio_channels_to_ablate.
                channels_to_ablate = ablation_layer.activations_to_be_ablated(activations[batch_index, :],
                                                                              self.ratio_channels_to_ablate)
                number_channels_to_ablate = len(channels_to_ablate)

                for i in tqdm.tqdm(range(0, number_channels_to_ablate, self.batch_size)):
                    if i + self.batch_size > number_channels_to_ablate:
                        batch_tensor = batch_tensor[:(number_channels_to_ablate - i)]

                    # Change the state of the ablation layer so it ablates the next channels.
                    # TBD: Move this into the ablation layer forward pass.
                    ablation_layer.set_next_batch(input_batch_index=batch_index,
                                                  activations=self.activations,
                                                  num_channels_to_ablate=batch_tensor.size(0))
                    score = [target(o).cpu().item() for o in self.model(batch_tensor)]
                    new_scores.extend(score)
                    ablation_layer.indices = ablation_layer.indices[batch_tensor.size(0):]

                new_scores = self.assemble_ablation_scores(new_scores,
                                                           original_scores[batch_index],
                                                           channels_to_ablate,
                                                           number_of_channels)
                weights.extend(new_scores)

        weights = np.float32(weights)
        weights = weights.reshape(activations.shape[:2])
        original_scores = original_scores[:, None]
        weights = (original_scores - weights) / original_scores

        # Replace the model back to the original state
        replace_layer_recursive(self.model, ablation_layer, target_layer)
        return weights
