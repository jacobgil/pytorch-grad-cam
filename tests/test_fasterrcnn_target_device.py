"""Regression test: FasterRCNNBoxScoreTarget must not hardcode cuda/mps.

Mirrors the fix for SemanticSegmentationTarget (#546). The target must
inherit the device of the model outputs instead of assuming cuda/mps.
"""
import numpy as np
import torch

from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget


def test_target_inherits_output_device_cpu():
    target = FasterRCNNBoxScoreTarget(
        labels=[1],
        bounding_boxes=[np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32)],
    )
    outputs = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]], device="cpu"),
        "labels": torch.tensor([1], device="cpu"),
        "scores": torch.tensor([0.9], device="cpu"),
    }
    score = target(outputs)
    assert score.device.type == "cpu"
    assert score.item() > 0


def test_target_returns_zero_on_empty_boxes():
    target = FasterRCNNBoxScoreTarget(labels=[1], bounding_boxes=[np.zeros(4)])
    outputs = {
        "boxes": torch.zeros((0, 4), device="cpu"),
        "labels": torch.zeros((0,), dtype=torch.int64, device="cpu"),
        "scores": torch.zeros((0,), device="cpu"),
    }
    score = target(outputs)
    assert score.device.type == "cpu"
    assert score.item() == 0.0


def test_target_preserves_boxes_dtype():
    """box_iou requires both inputs to share dtype; the target must not
    silently upcast/downcast relative to model_outputs['boxes']."""
    target = FasterRCNNBoxScoreTarget(
        labels=[1],
        bounding_boxes=[np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64)],
    )
    outputs = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float64),
        "labels": torch.tensor([1]),
        "scores": torch.tensor([0.9], dtype=torch.float64),
    }
    score = target(outputs)
    assert score.dtype == torch.float64
