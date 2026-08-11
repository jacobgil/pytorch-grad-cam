from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import resize

from pytorch_grad_cam.grad_cam import GradCAM
from pytorch_grad_cam.utils.image import scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class SESS:
    """
    SESS: Saliency Enhancing with Scaling and Sliding.

    A meta-method that makes any CAM method robust to scale variance, to multiple
    occurrences of the target object and to distractors.
    The input image is resized to several scales, a sliding window extracts fixed
    size patches from every scale, and the CAM of every patch is computed with the
    `base_method`. The patch CAMs are weighted by the classification score of their
    patch (channel-wise weight), pasted back into the coordinates they were taken
    from, and fused with a spatial weighted average (or a max) over all patches.

    Paper: https://arxiv.org/abs/2207.01769
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: List[torch.nn.Module],
        reshape_transform: Optional[Callable] = None,
        base_method=GradCAM,
        scales: Optional[List[int]] = None,
        num_scales: int = 12,
        scale_step_ratio: float = 2.0 / 7.0,
        window_size: Optional[int] = None,
        step_size: Optional[int] = None,
        pre_filter_ratio: float = 0.0,
        theta: float = 0.0,
        pool: str = "mean",
        smooth: bool = True,
        use_softmax_scores: bool = True,
        batch_size: int = 32,
        **kwargs,
    ) -> None:
        """
        :param base_method: The CAM method used for every patch.
        :param scales: The sizes the shorter side of the image is resized to.
            Defaults to `num_scales` scales, starting at the window size and
            growing by `scale_step_ratio * window_size` (224, 288, ... for a
            224x224 input, as in the paper).
        :param window_size: The size of the (square) sliding window.
            Defaults to the shorter side of the input, so that the smallest scale
            is the input image itself.
        :param step_size: The stride of the sliding window. Defaults to the window
            size, i.e. non overlapping windows (windows at the right/bottom border
            are shifted back into the image, so they may overlap).
        :param pre_filter_ratio: The ratio of the lowest scoring patches to discard
            before computing their CAMs. 0.0 keeps all of the patches.
        :param theta: Values below this threshold don't participate in the
            spatial weighted average of the 'mean' pooling.
        :param pool: 'mean' for the spatial weighted average fusion, 'max' for a
            maximum over the patch CAMs.
        :param smooth: Apply a gaussian blur on the fused CAM.
        :param use_softmax_scores: Compute the channel-wise weight of a patch by
            applying the target on the softmax of the model output, as in the paper.
            Set to False for models/targets where a softmax is meaningless, in
            which case the raw target scores are used (clipped at 0).
        :param batch_size: The number of patches processed in a single forward pass.

        Any additional kwargs are passed to the `base_method` during initialization.
        """
        if pool not in ("mean", "max"):
            raise ValueError(f"pool has to be either 'mean' or 'max'. Got: {pool}.")
        if not 0.0 <= pre_filter_ratio < 1.0:
            raise ValueError(f"pre_filter_ratio has to be in [0, 1). Got: {pre_filter_ratio}.")
        if num_scales < 1:
            raise ValueError(f"num_scales has to be at least 1. Got: {num_scales}.")

        self.base_cam = base_method(model, target_layers, reshape_transform, **kwargs)
        self.model = self.base_cam.model
        self.device = self.base_cam.device
        self.scales = scales
        self.num_scales = num_scales
        self.scale_step_ratio = scale_step_ratio
        self.window_size = window_size
        self.step_size = step_size
        self.pre_filter_ratio = pre_filter_ratio
        self.theta = theta
        self.pool = pool
        self.smooth = smooth
        self.use_softmax_scores = use_softmax_scores
        self.batch_size = batch_size

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.forward(*args, **kwargs)

    def forward(
        self,
        input_tensor: torch.Tensor,
        targets: Optional[List[torch.nn.Module]] = None,
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        # The patches are new tensors anyway, gradients w.r.t the input are not used.
        input_tensor = input_tensor.detach().to(self.device)
        if targets is None:
            targets = [None] * input_tensor.size(0)
        if len(targets) != input_tensor.size(0):
            raise ValueError(
                f"Expected one target per image. Got {len(targets)} targets "
                f"for {input_tensor.size(0)} images."
            )

        # Every image has its own set of patches and scores, so they are
        # enhanced one by one.
        cams = [
            self.forward_single_image(input_tensor[i: i + 1], target, eigen_smooth)
            for i, target in enumerate(targets)
        ]
        return scale_cam_image(np.float32(cams))

    def forward_single_image(
        self,
        input_tensor: torch.Tensor,
        target: Optional[torch.nn.Module],
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        height, width = input_tensor.shape[-2:]
        window_size = self.window_size if self.window_size is not None else min(height, width)
        step_size = self.step_size if self.step_size is not None else window_size
        if target is None:
            target = self.get_target(input_tensor)

        patches, coordinates = self.collect_patches(input_tensor, window_size, step_size)
        scores = self.get_scores(patches, target)

        # Pre-filtering: only the highest scoring patches are visualized.
        order = np.argsort(scores)[int(len(scores) * self.pre_filter_ratio):]
        max_score = scores[order[-1]]
        # Channel-wise weights, in [0, 1].
        weights = scores[order] / max_score if max_score > 0 else np.ones(len(order), np.float32)

        fusion = _CAMFusion((height, width), self.pool, self.theta)
        for batch_start in range(0, len(order), self.batch_size):
            indices = order[batch_start: batch_start + self.batch_size]
            batch = torch.cat([patches[i] for i in indices])
            batch_cams = self.base_cam(batch, [target] * len(indices), eigen_smooth=eigen_smooth)
            batch_weights = weights[batch_start: batch_start + self.batch_size]
            for cam, weight, index in zip(batch_cams, batch_weights, indices):
                fusion.add(self.paste_cam(cam * weight, coordinates[index], (height, width)))

        cam = fusion.result()
        if self.smooth:
            cam = cv2.GaussianBlur(cam, (11, 11), 5.0, 0)
        return cam

    def get_scales(self, window_size: int) -> List[int]:
        if self.scales is not None:
            return self.scales
        step = int(round(window_size * self.scale_step_ratio))
        return [window_size + step * i for i in range(self.num_scales)]

    def collect_patches(
        self, input_tensor: torch.Tensor, window_size: int, step_size: int
    ) -> Tuple[List[torch.Tensor], List[Tuple[int, int, int, int]]]:
        """
        Scale the image to every scale, and slide a window over it.

        :returns: The patches, and for every patch the coordinates it was taken from,
            in the format (x1, y1, scaled_width, scaled_height).
        """
        patches, coordinates = [], []
        for scale in self.get_scales(window_size):
            if scale < window_size:
                raise ValueError(
                    f"Every scale has to be at least the window size {window_size}. Got: {scale}."
                )
            scaled = resize(input_tensor, scale)
            scaled_height, scaled_width = scaled.shape[-2:]
            for x1, y1 in sliding_window(scaled_height, scaled_width, step_size, window_size):
                patches.append(scaled[:, :, y1: y1 + window_size, x1: x1 + window_size])
                coordinates.append((x1, y1, scaled_width, scaled_height))
        return patches, coordinates

    def get_target(self, input_tensor: torch.Tensor) -> torch.nn.Module:
        """The highest scoring category of the image, like in BaseCAM."""
        with torch.no_grad():
            outputs = self.model(input_tensor)
        return ClassifierOutputTarget(np.argmax(outputs[0].cpu().data.numpy(), axis=-1))

    def get_scores(self, patches: List[torch.Tensor], target: torch.nn.Module) -> np.ndarray:
        """The score of the target category for every patch."""
        scores = []
        with torch.no_grad():
            for batch_start in range(0, len(patches), self.batch_size):
                batch = torch.cat(patches[batch_start: batch_start + self.batch_size])
                outputs = self.model(batch)
                if self.use_softmax_scores:
                    outputs = torch.softmax(outputs, dim=-1)
                scores.append(target(outputs).cpu().data.numpy())
        # Negative scores can't be used as weights.
        return np.maximum(np.concatenate(scores), 0)

    def paste_cam(
        self,
        cam: np.ndarray,
        coordinates: Tuple[int, int, int, int],
        target_size: Tuple[int, int],
    ) -> np.ndarray:
        """Place a patch CAM back into the coordinates of the original image."""
        x1, y1, scaled_width, scaled_height = coordinates
        height, width = target_size
        full_cam = np.zeros((scaled_height, scaled_width), dtype=np.float32)
        full_cam[y1: y1 + cam.shape[0], x1: x1 + cam.shape[1]] = cam
        return cv2.resize(full_cam, (width, height))

    def __del__(self):
        # The base method isn't created if the arguments were invalid.
        if hasattr(self, "base_cam"):
            self.base_cam.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        return self.base_cam.__exit__(exc_type, exc_value, exc_tb)


def sliding_window(height: int, width: int, step_size: int, window_size: int):
    """
    The top left corners of the windows covering an image.

    Windows that would cross the right or the bottom border are shifted back
    into the image, so that all of the windows have the same size.
    """
    return [
        (x, y)
        for y in window_starts(height, step_size, window_size)
        for x in window_starts(width, step_size, window_size)
    ]


def window_starts(length: int, step_size: int, window_size: int) -> List[int]:
    starts = []
    for start in range(0, length, step_size):
        if start + window_size >= length:
            starts.append(max(length - window_size, 0))
            break
        starts.append(start)
    return starts


class _CAMFusion:
    """
    Fuses the patch CAMs one by one, so that the memory usage doesn't depend
    on the number of patches.
    """

    def __init__(self, size: Tuple[int, int], pool: str, theta: float) -> None:
        self.pool = pool
        self.theta = theta
        self.total = np.zeros(size, dtype=np.float32)
        self.counts = np.zeros(size, dtype=np.float32)

    def add(self, cam: np.ndarray) -> None:
        if self.pool == "max":
            np.maximum(self.total, cam, out=self.total)
        else:
            mask = cam > self.theta
            self.total[mask] += cam[mask]
            self.counts += mask

    def result(self) -> np.ndarray:
        if self.pool == "max":
            return self.total
        # Spatial weighted average: every pixel is averaged over the patches
        # that actually cover it.
        return np.divide(self.total, self.counts, out=np.zeros_like(self.total), where=self.counts > 0)
