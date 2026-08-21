"""注册适配 Scenario Dreamer 闭环状态的数据一致性与风险引导损失。"""

from typing import Any, Dict

import torch
import torch.nn.functional as F


def register_scenario_guidance():
    """注册联合碰撞约束和自车 TTC 风险损失，重复调用保持幂等。"""
    from tbsim.configs import guidance_config
    from tbsim.utils.geometry_utils import transform_points_tensor
    from tbsim.utils.guidance_utils import GUIDANCE_REGISTRY, Guidance, register_guidance
    from tbsim.utils.trajdata_utils import enlarge_batch_samples

    defaults = {
        "scenario_collision": {
            "safety_margin": 0.5,
            "temperature": 0.5,
            "loss_timesteps": 20,
            "filter_timesteps": 20,
            "loss_scale": 1.0,
        },
        "scenario_ttc": {
            "distance_bandwidth": 2.0,
            "time_bandwidth": 2.0,
            "min_velocity_diff": 0.1,
            "max_ttc": 6.0,
            "loss_timesteps": 20,
            "filter_timesteps": 20,
            "loss_scale": 1.0,
        },
    }
    for name, config in defaults.items():
        guidance_config.DEFAULT_GUIDANCE_CONFIGS.setdefault(name, config)

    def _world_positions(state, params):
        return transform_points_tensor(state[..., :2], params["world_from_agent"])

    def _ego_future(params, horizon, dtype, device):
        ego_state = params.get("scenario_ego_state")
        if ego_state is None:
            raise ValueError("联合风险引导要求 scenario_ego_state")
        ego_state = ego_state[0].to(device=device, dtype=dtype)
        times = torch.arange(1, horizon + 1, device=device, dtype=dtype) * float(params["dt"])
        return ego_state[None, :2] + times[:, None] * ego_state[None, 2:4]

    if "scenario_collision" not in GUIDANCE_REGISTRY:
        @register_guidance("scenario_collision")
        class ScenarioCollisionLossCalculator(Guidance):
            """避免背景车彼此或与固定自车发生几何重叠，同时允许形成近失事件。"""

            def __init__(
                self,
                safety_margin=0.5,
                temperature=0.5,
                loss_timesteps=20,
                filter_timesteps=20,
                loss_scale=1.0,
            ):
                super().__init__(loss_timesteps, filter_timesteps, loss_scale)
                if safety_margin < 0 or temperature <= 0:
                    raise ValueError("碰撞损失参数必须满足 safety_margin>=0 且 temperature>0")
                self.safety_margin = float(safety_margin)
                self.temperature = float(temperature)

            def update_params(self, params: Dict[str, torch.Tensor]) -> None:
                pass

            def update_config(self, config: Dict[str, Any]) -> None:
                pass

            def calculate_loss(self, action, state, params):
                batch_size = int(params["batch_size"])
                num_samples = int(params["num_samples"])
                horizon = state.shape[1]
                world = _world_positions(state, params).reshape(
                    batch_size, num_samples, horizon, 2
                )
                extents = params["ego_extents"].to(world).clamp_min(0.1)
                radii = 0.5 * torch.linalg.norm(extents, dim=-1)

                pair_distance = torch.linalg.norm(
                    world[:, None] - world[None, :], dim=-1
                )
                pair_margin = radii[:, None, None, None] + radii[None, :, None, None]
                pair_margin = pair_margin + self.safety_margin
                pair_penalty = F.softplus(
                    (pair_margin - pair_distance) / self.temperature
                ) * self.temperature
                identity = torch.eye(batch_size, device=world.device, dtype=torch.bool)
                pair_penalty = pair_penalty.masked_fill(identity[:, :, None, None], 0.0)
                loss = pair_penalty.sum(dim=1)

                ego_future = _ego_future(params, horizon, world.dtype, world.device)
                ego_extent = params["scenario_ego_state"][0, 5:7].to(world).clamp_min(0.1)
                ego_radius = 0.5 * torch.linalg.norm(ego_extent)
                ego_distance = torch.linalg.norm(world - ego_future[None, None], dim=-1)
                ego_margin = radii[:, None, None] + ego_radius + self.safety_margin
                loss = loss + F.softplus(
                    (ego_margin - ego_distance) / self.temperature
                ) * self.temperature

                loss = loss.reshape(batch_size * num_samples, horizon)
                if self.loss_timesteps is not None:
                    loss[:, int(self.loss_timesteps):] = 0.0
                return loss * self.loss_scale

    if "scenario_ttc" not in GUIDANCE_REGISTRY:
        @register_guidance("scenario_ttc")
        class ScenarioTTCLossCalculator(Guidance):
            """提升目标背景车与固定自车之间的有限时域 TTC 风险。"""

            def __init__(
                self,
                distance_bandwidth=2.0,
                time_bandwidth=2.0,
                min_velocity_diff=0.1,
                max_ttc=6.0,
                loss_timesteps=20,
                filter_timesteps=20,
                loss_scale=1.0,
            ):
                super().__init__(loss_timesteps, filter_timesteps, loss_scale)
                if min(distance_bandwidth, time_bandwidth, min_velocity_diff, max_ttc) <= 0:
                    raise ValueError("TTC 损失带宽、速度阈值和最大 TTC 必须为正")
                self.distance_bandwidth = float(distance_bandwidth)
                self.time_bandwidth = float(time_bandwidth)
                self.min_velocity_diff = float(min_velocity_diff)
                self.max_ttc = float(max_ttc)

            def update_params(self, params: Dict[str, torch.Tensor]) -> None:
                pass

            def update_config(self, config: Dict[str, Any]) -> None:
                pass

            def calculate_loss(self, action, state, params):
                batch_size = int(params["batch_size"])
                num_samples = int(params["num_samples"])
                horizon = state.shape[1]
                world = _world_positions(state, params).reshape(
                    batch_size, num_samples, horizon, 2
                )
                ego_future = _ego_future(params, horizon, world.dtype, world.device)
                ego_velocity = params["scenario_ego_state"][0, 2:4].to(world)

                current = params["world_from_agent"][:, :2, -1].reshape(
                    batch_size, num_samples, 2
                )
                previous = torch.cat([current[:, :, None], world[:, :, :-1]], dim=2)
                background_velocity = (world - previous) / float(params["dt"])
                relative_position = world - ego_future[None, None]
                relative_velocity = background_velocity - ego_velocity[None, None, None]
                speed_squared = relative_velocity.square().sum(dim=-1).clamp_min(
                    self.min_velocity_diff ** 2
                )
                ttc = -(relative_position * relative_velocity).sum(dim=-1) / speed_squared
                closest_vector = relative_position + ttc[..., None] * relative_velocity
                closest_distance = torch.linalg.norm(closest_vector, dim=-1)
                valid = (ttc > 0.0) & (ttc <= self.max_ttc)
                risk = torch.exp(-ttc.clamp_min(0.0) / self.time_bandwidth)
                risk = risk * torch.exp(-closest_distance / self.distance_bandwidth) * valid

                target_mask = params.get("guidance_target_mask")
                if target_mask is not None:
                    target_mask = enlarge_batch_samples(
                        target_mask.to(device=world.device, dtype=world.dtype),
                        batch_size,
                        num_samples,
                    ).reshape(batch_size, num_samples)
                    risk = risk * target_mask[:, :, None]
                loss = -risk.reshape(batch_size * num_samples, horizon)
                if self.loss_timesteps is not None:
                    loss[:, int(self.loss_timesteps):] = 0.0
                return loss * self.loss_scale

    return {
        name: GUIDANCE_REGISTRY[name]
        for name in ("scenario_collision", "scenario_ttc")
    }
