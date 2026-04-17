"""Regression tests for FinerCAM:

1. Supports the `with FinerCAM(...) as cam:` protocol used by cam.py (K3).
2. Does not IndexError on models with fewer than 4 output classes (K8).
3. References self.base_cam._htcore (single underscore) so name-mangling does
   not break HPU users (K10 — static check since no HPU hardware in CI).
"""

import inspect

import torch
import torch.nn as nn

from pytorch_grad_cam import FinerCAM


def _cnn(num_classes: int) -> nn.Module:
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(4, num_classes)

        def forward(self, x):
            return self.fc(self.pool(torch.relu(self.conv(x))).flatten(1))

    m = M()
    m.eval()
    return m


def test_finercam_supports_context_manager():
    model = _cnn(10)
    with FinerCAM(model=model, target_layers=[model.conv]) as cam:
        assert cam is not None


def test_finercam_runs_on_binary_classifier_without_index_error():
    model = _cnn(2)
    x = torch.randn(1, 3, 16, 16)
    with FinerCAM(model=model, target_layers=[model.conv]) as cam:
        out = cam(input_tensor=x, targets=None)
    assert out.shape == (1, 16, 16)


def test_finercam_runs_on_ternary_classifier_without_index_error():
    model = _cnn(3)
    x = torch.randn(1, 3, 16, 16)
    with FinerCAM(model=model, target_layers=[model.conv]) as cam:
        out = cam(input_tensor=x, targets=None)
    assert out.shape == (1, 16, 16)


def test_finercam_htcore_is_single_underscore():
    """Guard against re-introducing the name-mangled `__htcore` attribute."""
    src = inspect.getsource(FinerCAM)
    assert "self.base_cam.__htcore" not in src, (
        "FinerCAM must not reference self.base_cam.__htcore (name-mangled)"
    )
    assert "self.base_cam._htcore" in src, "FinerCAM should call into _htcore"
