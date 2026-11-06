"""Tests for show_cam_on_image overlay helpers."""

from __future__ import annotations

import numpy as np

from pytorch_grad_cam.utils.image import show_cam_on_image


def test_show_cam_on_image_accepts_list_mask() -> None:
    """A nested list mask must scale like an ndarray, not repeat via ``*``."""
    img = np.zeros((8, 8, 3), dtype=np.float32)
    mask = [[0.0] * 8 for _ in range(8)]
    for r in range(2, 6):
        for c in range(2, 6):
            mask[r][c] = 1.0

    out = show_cam_on_image(img, mask, use_rgb=True)
    assert out.shape == (8, 8, 3)
    assert out.dtype == np.uint8


def test_show_cam_on_image_ndarray_mask_unchanged() -> None:
    """Existing ndarray mask inputs keep working."""
    img = np.zeros((8, 8, 3), dtype=np.float32)
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[2:6, 2:6] = 1.0

    out = show_cam_on_image(img, mask, use_rgb=True)
    assert out.shape == (8, 8, 3)
    assert out.dtype == np.uint8
