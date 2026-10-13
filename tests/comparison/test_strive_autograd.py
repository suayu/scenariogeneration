"""eval RNN的潜变量梯度、后端范围和网络状态回归。"""
import copy

import pytest
import torch

from experiments.comparison.strive_official_worker import configure_rnn_autograd


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_eval_rnn_has_finite_input_gradient_without_training_mode(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    original = torch.nn.GRU(4, 8, 2, batch_first=True).to(device).eval()
    model = configure_rnn_autograd(copy.deepcopy(original), 'native')
    x = torch.randn(2, 3, 4, device=device, requires_grad=True)
    backend_before = torch.backends.cudnn.enabled
    with torch.no_grad():
        expected, _ = original(x)
    actual, _ = model(x)
    actual.square().sum().backward()
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert not model.training
    assert torch.backends.cudnn.enabled == backend_before
    for key, value in original.state_dict().items():
        assert torch.equal(value, model.state_dict()[key])
