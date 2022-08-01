import numpy as np
import torch
from typing import Callable, List, Tuple
from sklearn.decomposition import NMF
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.image import scale_cam_image


def dff(activations: np.ndarray, n_components: int = 5):
    """ Compute Deep Feature Factorization on a 2d Activations tensor.

    :param activations: A numpy array of shape batch x channels x height x width
    :param n_components: The number of components for the non negative matrix factorization
    :returns: A tuple of the concepts (a numpy array with shape channels x components),
              and the explanation heatmaps (a numpy arary with shape batch x height x width)
    """

    batch_size, channels, h, w = activations.shape
    reshaped_activations = activations.transpose((1, 0, 2, 3))
    reshaped_activations[np.isnan(reshaped_activations)] = 0
    reshaped_activations = reshaped_activations.reshape(
        reshaped_activations.shape[0], -1)
    reshaped_activations = reshaped_activations - \
        reshaped_activations.min(axis=0)
    model = NMF(n_components=n_components, init='random', random_state=0)
    W = model.fit_transform(reshaped_activations)
    H = model.components_
    concepts = W
    explanations = H.reshape(batch_size, n_components, h, w)
    return concepts, explanations


class DeepFeatureFactorization:
    """ Deep Feature Factorization: https://arxiv.org/abs/1806.10206
        This gets a model andcomputes the 2D activations for a target layer,
        and computes Non Negative Matrix Factorization on the activations.

        Optionally it runs a computation on the concept embeddings,
        like running a classifier on them.

        The explanation heatmaps are scalled to the range [0, 1]
        and to the input tensor width and height.
     """

    def __init__(self,
                 model: torch.nn.Module,
                 target_layer: torch.nn.Module,
                 reshape_transform: Callable = None,
                 computation_on_concepts=None
                 ):
        self.model = model
        self.computation_on_concepts = computation_on_concepts
        self.activations_and_grads = ActivationsAndGradients(
            self.model, [target_layer], reshape_transform)

    def __call__(self,
                 input_tensor: torch.Tensor,
                 n_components: int = 16):
        batch_size, channels, h, w = input_tensor.size()
        _ = self.activations_and_grads(input_tensor)
        activations = self.activations_and_grads.activations[0].cpu().numpy()
        concepts, explanations = dff(activations, n_components=n_components)

        processed_explanations = []
        for batch in explanations:
            processed_explanations.append(scale_cam_image(batch, (w, h)))

        if self.computation_on_concepts:
            with torch.no_grad():
                concept_tensors = torch.from_numpy(
                    np.float32(concepts).transpose((1, 0)))
                concept_outputs = self.computation_on_concepts(
                    concept_tensors).cpu().numpy()
            return concepts, processed_explanations, concept_outputs
        else:
            return concepts, processed_explanations

    def __del__(self):
        self.activations_and_grads.release()

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in ActivationSummary with block: {exc_type}. Message: {exc_value}")
            return True
