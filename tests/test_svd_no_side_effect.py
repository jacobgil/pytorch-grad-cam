import numpy as np

from pytorch_grad_cam.utils.svd_on_activations import (
    get_2d_projection,
    get_2d_projection_kernel,
    get_2d_projection_with_sign_correction,
)


def _sample_batch_with_nan():
    a = np.zeros((1, 2, 2, 2), dtype=np.float32)
    a[0, 0, 0, 0] = np.nan
    a[0, 0, 1, 1] = 1.0
    a[0, 1] = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    return a


def test_get_2d_projection_does_not_mutate_caller():
    a = _sample_batch_with_nan()
    assert np.isnan(a).any()
    snapshot = a.copy()
    _ = get_2d_projection(a)
    assert np.array_equal(a, snapshot, equal_nan=True), (
        "get_2d_projection mutated caller's activation_batch in place"
    )


def test_get_2d_projection_kernel_does_not_mutate_caller():
    a = _sample_batch_with_nan()
    snapshot = a.copy()
    _ = get_2d_projection_kernel(a)
    assert np.array_equal(a, snapshot, equal_nan=True), (
        "get_2d_projection_kernel mutated caller's activation_batch in place"
    )


def test_get_2d_projection_with_sign_correction_does_not_mutate_caller():
    a = _sample_batch_with_nan()
    snapshot = a.copy()
    _ = get_2d_projection_with_sign_correction(a)
    assert np.array_equal(a, snapshot, equal_nan=True), (
        "get_2d_projection_with_sign_correction mutated caller's "
        "activation_batch in place"
    )
