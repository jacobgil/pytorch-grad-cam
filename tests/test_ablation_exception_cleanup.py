import numpy as np
import pytest
import torch
import torch.nn as nn

from pytorch_grad_cam import AblationCAM
from pytorch_grad_cam.ablation_layer import AblationLayer


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 1, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.tensor([1.0, 2.0]).reshape(2, 1, 1, 1))
        self.calls = 0
        self.fail_on_call = 0
        self.error = RuntimeError("injected forward failure")

    def forward(self, x):
        self.calls += 1
        x = self.conv(x)
        if self.calls == self.fail_on_call:
            raise self.error
        return x.mean(dim=(2, 3)).sum(dim=1, keepdim=True)


class FailingTarget:
    def __init__(self, fail_on_call):
        self.calls = 0
        self.fail_on_call = fail_on_call
        self.error = RuntimeError("injected target failure")

    def __call__(self, output):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise self.error
        return output[0]


@pytest.mark.parametrize("forward_failure,target_failure", [
    pytest.param(2, 0, id="cache-forward-failure"),
    pytest.param(0, 1, id="cached-target-failure"),
    pytest.param(3, 0, id="ablated-forward-failure"),
    pytest.param(0, 2, id="ablated-target-failure"),
    pytest.param(0, 0, id="success"),
])
def test_ablation_restores_model_and_hooks(forward_failure, target_failure):
    # Given a deterministic CPU model with a caller-owned forward hook.
    model = TinyModel().eval()
    original_layer = model.conv
    input_tensor = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)
    hook_outputs = []
    user_hook = original_layer.register_forward_hook(
        lambda module, inputs, output: hook_outputs.append(output.detach().clone()))
    original_hooks = dict(original_layer._forward_hooks)
    with torch.no_grad():
        expected_output = model(input_tensor)
    model.calls = 0
    model.fail_on_call = forward_failure
    target = FailingTarget(target_failure)

    try:
        with AblationCAM(model, [original_layer], ablation_layer=AblationLayer(),
                         batch_size=2) as cam:
            expected_hooks = dict(original_layer._forward_hooks)

            # When the public CAM API completes or propagates the injected error.
            if forward_failure or target_failure:
                expected_error = model.error if forward_failure else target.error
                with pytest.raises(RuntimeError) as caught:
                    cam(input_tensor=input_tensor, targets=[target])
                assert caught.value is expected_error
            else:
                result = cam(input_tensor=input_tensor, targets=[target])
                np.testing.assert_allclose(result, input_tensor[:, 0].numpy() / 3,
                                           atol=1e-6)

            # Then only temporary state is removed, even before context exit.
            assert model.conv is original_layer
            assert dict(original_layer._forward_hooks) == expected_hooks
            model.fail_on_call = 0
            hook_count = len(hook_outputs)
            with torch.no_grad():
                torch.testing.assert_close(model(input_tensor), expected_output)
            assert len(hook_outputs) == hook_count + 1

        assert dict(original_layer._forward_hooks) == original_hooks
        with torch.no_grad():
            torch.testing.assert_close(model(input_tensor), expected_output)
    finally:
        user_hook.remove()
