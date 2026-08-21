"""在不修改 Safe-Sim 核心代码的前提下注册 LLM 轨迹锚点损失。"""

from typing import Any, Dict

import torch
import torch.nn.functional as F


def register_llm_anchor_guidance():
    """注册锚点损失并返回损失计算器类型，重复调用不会重复注册。"""
    from tbsim.configs import guidance_config
    from tbsim.utils.guidance_utils import (
        GUIDANCE_REGISTRY,
        Guidance,
        register_guidance,
    )
    from tbsim.utils.trajdata_utils import enlarge_batch_samples

    default_config = {
        "loss_timesteps": None,
        "filter_timesteps": None,
        "loss_scale": 1.0,
        "robust_delta": 1.0,
    }
    guidance_config.DEFAULT_GUIDANCE_CONFIGS.setdefault("llm_anchor", default_config)
    if "llm_anchor" in GUIDANCE_REGISTRY:
        return GUIDANCE_REGISTRY["llm_anchor"]

    @register_guidance("llm_anchor")
    class LLMAnchorLossCalculator(Guidance):
        """计算生成轨迹与 LLM 锚点之间的径向 Huber 位置损失。"""

        def __init__(
            self,
            loss_timesteps=None,
            filter_timesteps=None,
            loss_scale=1.0,
            robust_delta=1.0,
        ):
            super().__init__(loss_timesteps, filter_timesteps, loss_scale)
            if loss_scale < 0:
                raise ValueError("llm_anchor loss_scale must be non-negative")
            if robust_delta <= 0:
                raise ValueError("llm_anchor robust_delta must be positive")
            self.robust_delta = float(robust_delta)
            self.anchor_positions = None
            self.anchor_mask = None
            self.num_samples = None

        def set_anchor_targets(self, positions, mask, num_samples):
            """设置本次采样使用的局部坐标锚点及有效时间掩码。"""
            self.anchor_positions = positions
            self.anchor_mask = mask
            self.num_samples = int(num_samples)

        def clear_anchor_targets(self):
            """清除逐帧目标，避免跨场景或跨计划复用旧锚点。"""
            self.anchor_positions = None
            self.anchor_mask = None
            self.num_samples = None

        def update_params(self, params: Dict[str, torch.Tensor]) -> None:
            pass

        def update_config(self, config: Dict[str, Any]) -> None:
            pass

        def calculate_loss(self, action, state, data_batch_for_guidance):
            if self.anchor_positions is None or self.anchor_mask is None:
                return state[..., 0] * 0.0
            batch_size = int(data_batch_for_guidance["batch_size"])
            if self.anchor_positions.shape[0] != batch_size:
                raise ValueError("LLM anchor batch size does not match guidance batch size")

            anchor_positions = enlarge_batch_samples(
                self.anchor_positions.to(device=state.device, dtype=state.dtype),
                batch_size,
                self.num_samples,
            )
            anchor_mask = enlarge_batch_samples(
                self.anchor_mask.to(device=state.device, dtype=torch.bool),
                batch_size,
                self.num_samples,
            )
            horizon = min(state.shape[1], anchor_positions.shape[1])
            distance = torch.linalg.norm(
                state[:, :horizon, :2] - anchor_positions[:, :horizon],
                dim=-1,
            )

            # 近距离采用二次项保证平滑，远距离采用线性项限制极端梯度。
            delta = self.robust_delta
            robust_loss = torch.where(
                distance <= delta,
                0.5 * distance.square() / delta,
                distance - 0.5 * delta,
            )
            robust_loss = robust_loss * anchor_mask[:, :horizon].to(robust_loss.dtype)
            if self.loss_timesteps is not None:
                valid_horizon = min(horizon, int(self.loss_timesteps))
                timestep_mask = torch.arange(horizon, device=state.device) < valid_horizon
                robust_loss = robust_loss * timestep_mask.unsqueeze(0)
            if horizon < state.shape[1]:
                robust_loss = F.pad(robust_loss, (0, state.shape[1] - horizon))
            return robust_loss * self.loss_scale

        def calculate_grad(self, action, state, a_t, grad_wrt, data_batch_for_guidance):
            """直接对即将更新的干净控制量求负损失梯度。"""
            loss = self.calculate_loss(action, state, data_batch_for_guidance)
            grad = torch.autograd.grad(-loss.sum(), action, retain_graph=True)[0]
            if not torch.isfinite(grad).all():
                raise FloatingPointError("LLM anchor guidance produced a non-finite gradient")
            return grad

    return LLMAnchorLossCalculator
