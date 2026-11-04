from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from pytorch_grad_cam.feature_factorization import deep_feature_factorization
from pytorch_grad_cam.feature_factorization.deep_feature_factorization import run_dff_on_image


def test_run_dff_on_image_without_reshape_transform(monkeypatch):
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(4, 3),
    )
    model.config = SimpleNamespace(id2label={0: "a", 1: "b", 2: "c"})
    classifier = model[-1]

    img_pil = Image.fromarray(np.uint8(np.random.rand(16, 16, 3) * 255))
    img_tensor = torch.rand(3, 16, 16)

    # Only the reshape_transform default is under test here, not the rendering.
    monkeypatch.setattr(deep_feature_factorization, "show_factorization_on_image",
                        lambda img, *args, **kwargs: np.uint8(img * 255))

    result = run_dff_on_image(model=model,
                              target_layer=model[1],
                              classifier=classifier,
                              img_pil=img_pil,
                              img_tensor=img_tensor,
                              n_components=2,
                              top_k=1)

    assert result.shape == (16, 32, 3)
