"""验证真实扩散内环的动作梯度方向和跳步去噪边界。"""
from types import SimpleNamespace

import pytest
import torch
from tbsim.models.diffusion import DiffusionTraj, VarianceSchedule
from tbsim.utils.guidance_utils import Guidance


class QuadraticGuide(Guidance):
    # 使用凸损失隔离网络雅可比，检查更新是否作用于正确变量。
    def calculate_loss(self, action, state, data_batch_for_guidance):
        return action.square().sum(-1)


class IdentityDynamics:
    def forward_dynamics(self, current_states, action, dt, **kwargs):
        return action, None


class FixedDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dyn = IdentityDynamics()
        self.step_time = 0.1

    def forward(self, tau, context, **kwargs):
        # 已知干净输出用于检验最后一次采样是否精确落到 x0。
        return torch.ones_like(tau[..., :2]) * 0.3


def test_clean_update_descends_despite_negative_denoiser_jacobian():
    noisy = torch.ones(2, 4, 2, requires_grad=True)
    clean = -2 * noisy
    guide = QuadraticGuide()
    sampler = DiffusionTraj(FixedDenoiser(), VarianceSchedule(100))
    updated = sampler.n_step_guided_p_sample(
        noisy, clean, guide, beta=1.0, current_states=torch.zeros(2, 4),
        grad_wrt="clean_guide", inner_lr=0.1, inner_beta=0.5, n_guide_steps=3,
    )
    assert updated.square().sum() < clean.square().sum()


@pytest.mark.parametrize("stride", [1, 4, 6])
@pytest.mark.parametrize("mode", ["ddpm", "ddim"])
def test_last_reverse_step_returns_clean_prediction(stride, mode):
    # 无引导与固定网络去除模型质量影响，涵盖不能整除训练步数的步长。
    sampler = DiffusionTraj(FixedDenoiser(), VarianceSchedule(100))
    result = sampler.sample(
        context=torch.zeros(2, 3), num_points=4, bestof=True, num_samples=3,
        current_states=torch.zeros(2, 4), forward_mode="eval",
        sampling_mode=mode, sample_step=stride,
        guide_config=SimpleNamespace(params=SimpleNamespace(partial_t=None)),
    )
    torch.testing.assert_close(result, torch.full_like(result, 0.3), atol=1e-6, rtol=1e-6)
