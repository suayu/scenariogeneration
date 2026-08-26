"""注册适配 Scenario Dreamer 闭环状态的数据一致性与风险引导损失。"""

from typing import Any, Dict

import torch
import torch.nn.functional as F


def resolve_attack_contact_schedule(
    attack_age_frames,
    protection_frames,
    relaxation_frames,
    contact_ramp_frames,
):
    """根据攻击持续帧数计算目标车防碰撞和受控接触的阶段参数。"""
    attack_age_frames = max(int(attack_age_frames), 0)
    protection_frames = max(int(protection_frames), 0)
    relaxation_frames = max(int(relaxation_frames), 0)
    contact_ramp_frames = max(int(contact_ramp_frames), 1)

    if attack_age_frames < protection_frames:
        return {
            "phase": "protection",
            "avoidance_weight": 1.0,
            "contact_weight": 0.0,
            "contact_step": None,
        }

    relaxation_age = attack_age_frames - protection_frames
    if relaxation_age < relaxation_frames:
        avoidance_weight = (relaxation_frames - relaxation_age - 1) / max(
            relaxation_frames, 1
        )
        return {
            "phase": "relaxation",
            "avoidance_weight": float(avoidance_weight),
            "contact_weight": 0.0,
            "contact_step": None,
        }

    contact_age = relaxation_age - relaxation_frames
    return {
        "phase": "contact",
        "avoidance_weight": 0.0,
        "contact_weight": float(
            min((contact_age + 1) / contact_ramp_frames, 1.0)
        ),
        # 接触目标随闭环推进逐帧移近，截止帧后固定要求第一预测步接触。
        "contact_step": max(contact_ramp_frames - contact_age - 1, 0),
    }


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
            "attack_protection_frames": 2,
            "attack_relaxation_frames": 6,
            "attack_contact_ramp_frames": 2,
            "controlled_contact_penetration": 0.05,
            "max_contact_penetration": 0.15,
            "contact_loss_scale": 1.0,
            "penetration_loss_scale": 5.0,
            "safety_penetration_loss_scale": 100.0,
            "max_speed": 20.0,
            "max_acceleration": 6.0,
            "max_jerk": 12.0,
            "max_step_distance": 2.0,
            "kinematics_loss_scale": 1.0,
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
                attack_protection_frames=2,
                attack_relaxation_frames=6,
                attack_contact_ramp_frames=2,
                controlled_contact_penetration=0.05,
                max_contact_penetration=0.15,
                contact_loss_scale=1.0,
                penetration_loss_scale=5.0,
                safety_penetration_loss_scale=100.0,
                max_speed=20.0,
                max_acceleration=6.0,
                max_jerk=12.0,
                max_step_distance=2.0,
                kinematics_loss_scale=1.0,
            ):
                super().__init__(loss_timesteps, filter_timesteps, loss_scale)
                if safety_margin < 0 or temperature <= 0:
                    raise ValueError("碰撞损失参数必须满足 safety_margin>=0 且 temperature>0")
                if min(
                    attack_protection_frames,
                    attack_relaxation_frames,
                    controlled_contact_penetration,
                    max_contact_penetration,
                    contact_loss_scale,
                    penetration_loss_scale,
                    safety_penetration_loss_scale,
                    kinematics_loss_scale,
                ) < 0:
                    raise ValueError("攻击阶段帧数、接触深度和损失强度不能为负")
                if attack_contact_ramp_frames < 1:
                    raise ValueError("接触引导渐强帧数必须至少为一帧")
                if controlled_contact_penetration > max_contact_penetration:
                    raise ValueError("受控接触深度不能超过硬穿透上限")
                if min(max_speed, max_acceleration, max_jerk, max_step_distance) <= 0:
                    raise ValueError("运动学速度、加速度、加加速度和位移阈值必须为正")
                self.safety_margin = float(safety_margin)
                self.temperature = float(temperature)
                self.attack_protection_frames = int(attack_protection_frames)
                self.attack_relaxation_frames = int(attack_relaxation_frames)
                self.attack_contact_ramp_frames = int(attack_contact_ramp_frames)
                self.controlled_contact_penetration = float(
                    controlled_contact_penetration
                )
                self.max_contact_penetration = float(max_contact_penetration)
                self.contact_loss_scale = float(contact_loss_scale)
                self.penetration_loss_scale = float(penetration_loss_scale)
                self.safety_penetration_loss_scale = float(
                    safety_penetration_loss_scale
                )
                self.max_speed = float(max_speed)
                self.max_acceleration = float(max_acceleration)
                self.max_jerk = float(max_jerk)
                self.max_step_distance = float(max_step_distance)
                self.kinematics_loss_scale = float(kinematics_loss_scale)

            def update_params(self, params: Dict[str, torch.Tensor]) -> None:
                pass

            def update_config(self, config: Dict[str, Any]) -> None:
                pass

            def _kinematics_loss(self, world, params, batch_size, num_samples):
                """约束速度、加速度、加加速度和单帧位移，阻止瞬移与高速穿模。"""
                horizon = world.shape[2]
                dt = float(params["dt"])
                current_position = params["world_from_agent"][:, :2, -1].reshape(
                    batch_size, num_samples, 2
                )
                previous_position = torch.cat(
                    [current_position[:, :, None], world[:, :, :-1]], dim=2
                )
                displacement = world - previous_position
                velocity = displacement / dt
                speed = torch.linalg.norm(velocity, dim=-1)

                current_speed = enlarge_batch_samples(
                    params["curr_speed"].to(world), batch_size, num_samples
                ).reshape(batch_size, num_samples)
                current_yaw = params["yaw"].to(world).reshape(
                    batch_size, num_samples
                )
                current_velocity = torch.stack(
                    [
                        current_speed * torch.cos(current_yaw),
                        current_speed * torch.sin(current_yaw),
                    ],
                    dim=-1,
                )
                previous_velocity = torch.cat(
                    [current_velocity[:, :, None], velocity[:, :, :-1]], dim=2
                )
                acceleration = (velocity - previous_velocity) / dt
                acceleration_norm = torch.linalg.norm(acceleration, dim=-1)

                current_acceleration = params.get(
                    "scenario_curr_acceleration_world"
                )
                if current_acceleration is None:
                    current_acceleration = acceleration[:, :, 0].detach()
                else:
                    current_acceleration = enlarge_batch_samples(
                        current_acceleration.to(world), batch_size, num_samples
                    ).reshape(batch_size, num_samples, 2)
                previous_acceleration = torch.cat(
                    [current_acceleration[:, :, None], acceleration[:, :, :-1]],
                    dim=2,
                )
                jerk_norm = torch.linalg.norm(
                    (acceleration - previous_acceleration) / dt,
                    dim=-1,
                )

                # 可行域内损失严格为零，避免对 Safe-Sim 的正常轨迹分布施加无谓梯度。
                loss = F.relu(speed - self.max_speed).square()
                loss = loss + F.relu(
                    acceleration_norm - self.max_acceleration
                ).square()
                loss = loss + F.relu(jerk_norm - self.max_jerk).square()
                loss = loss + F.relu(
                    torch.linalg.norm(displacement, dim=-1) - self.max_step_distance
                ).square()
                return loss * self.kinematics_loss_scale

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
                # 软安全边界负责提前避让，二次穿透屏障负责强烈排斥已经重叠的背景车。
                pair_overlap = F.relu(
                    radii[:, None, None, None]
                    + radii[None, :, None, None]
                    - pair_distance
                )
                pair_penalty = pair_penalty + (
                    pair_overlap.square() * self.safety_penetration_loss_scale
                )
                identity = torch.eye(batch_size, device=world.device, dtype=torch.bool)
                pair_penalty = pair_penalty.masked_fill(identity[:, :, None, None], 0.0)
                loss = pair_penalty.sum(dim=1)

                ego_future = _ego_future(params, horizon, world.dtype, world.device)
                ego_extent = params["scenario_ego_state"][0, 5:7].to(world).clamp_min(0.1)
                ego_radius = 0.5 * torch.linalg.norm(ego_extent)
                ego_distance = torch.linalg.norm(world - ego_future[None, None], dim=-1)
                ego_margin = radii[:, None, None] + ego_radius + self.safety_margin
                ego_avoidance = F.softplus(
                    (ego_margin - ego_distance) / self.temperature
                ) * self.temperature

                # 非攻击车辆始终避让自车；只有 LLM 明确指定的攻击者进入分阶段调度。
                attack_mask = params.get("guidance_attack_active")
                if attack_mask is None:
                    attack_mask = torch.zeros(batch_size, device=world.device)
                attack_mask = enlarge_batch_samples(
                    attack_mask.to(device=world.device, dtype=world.dtype),
                    batch_size,
                    num_samples,
                ).reshape(batch_size, num_samples)
                attack_mask = attack_mask.clamp(0.0, 1.0)

                attack_age = params.get("guidance_attack_age_frames")
                if attack_age is None or not bool((attack_mask > 0.5).any()):
                    schedule = {
                        "avoidance_weight": 1.0,
                        "contact_weight": 0.0,
                        "contact_step": None,
                    }
                else:
                    schedule = resolve_attack_contact_schedule(
                        int(torch.as_tensor(attack_age).max().item()),
                        self.attack_protection_frames,
                        self.attack_relaxation_frames,
                        self.attack_contact_ramp_frames,
                    )

                avoidance_weight = 1.0 - attack_mask[:, :, None] + (
                    attack_mask[:, :, None] * schedule["avoidance_weight"]
                )
                if schedule["contact_step"] is not None:
                    # 接触目标尚在预测远端时，目标帧之前仍完整防碰撞，避免提前撞击。
                    contact_step = min(int(schedule["contact_step"]), horizon - 1)
                    protected_steps = (
                        torch.arange(horizon, device=world.device) < contact_step
                    ).to(world.dtype)
                    avoidance_weight = torch.maximum(
                        avoidance_weight,
                        attack_mask[:, :, None] * protected_steps[None, None],
                    )
                loss = loss + ego_avoidance * avoidance_weight

                # 非攻击车辆始终使用强穿透屏障；攻击者只按当前避碰阶段等比例减弱。
                ego_overlap = F.relu(
                    radii[:, None, None] + ego_radius - ego_distance
                ).square()
                loss = loss + (
                    ego_overlap
                    * avoidance_weight
                    * self.safety_penetration_loss_scale
                )

                # 即使进入接触阶段也禁止深度穿透；该项只约束攻击者，不改变其他车辆的安全逻辑。
                hard_min_distance = (
                    radii[:, None, None]
                    + ego_radius
                    - self.max_contact_penetration
                ).clamp_min(0.0)
                penetration_loss = F.relu(hard_min_distance - ego_distance).square()
                loss = loss + (
                    penetration_loss
                    * attack_mask[:, :, None]
                    * self.penetration_loss_scale
                )

                if schedule["contact_weight"] > 0.0:
                    contact_step = min(int(schedule["contact_step"]), horizon - 1)
                    desired_distance = (
                        radii[:, None]
                        + ego_radius
                        - self.controlled_contact_penetration
                    ).clamp_min(0.0)
                    contact_error = ego_distance[:, :, contact_step] - desired_distance
                    contact_loss = F.smooth_l1_loss(
                        contact_error,
                        torch.zeros_like(contact_error),
                        reduction="none",
                        beta=self.temperature,
                    )
                    loss[:, :, contact_step] = loss[:, :, contact_step] + (
                        contact_loss
                        * attack_mask
                        * schedule["contact_weight"]
                        * self.contact_loss_scale
                    )

                static_obstacles = params.get("static_obstacles_world")
                static_obstacle_mask = params.get("static_obstacle_mask")
                if static_obstacles is not None and static_obstacles.shape[1] > 0:
                    # 使用障碍物外接圆半径：保守、可微，且不改变原有车辆矩形碰撞逻辑。
                    obstacles = static_obstacles.to(device=world.device, dtype=world.dtype)
                    obstacle_mask = static_obstacle_mask.to(device=world.device, dtype=torch.bool)
                    obstacles = obstacles.reshape(batch_size, num_samples, -1, 5)
                    obstacle_mask = obstacle_mask.reshape(batch_size, num_samples, -1)
                    obstacle_radii = 0.5 * torch.linalg.norm(obstacles[..., 3:5], dim=-1)
                    obstacle_distance = torch.linalg.norm(
                        world[:, :, :, None, :] - obstacles[:, :, None, :, :2],
                        dim=-1,
                    )
                    obstacle_margin = (
                        radii[:, None, None, None]
                        + obstacle_radii[:, :, None, :]
                        + self.safety_margin
                    )
                    obstacle_penalty = F.softplus(
                        (obstacle_margin - obstacle_distance) / self.temperature
                    ) * self.temperature
                    obstacle_overlap = F.relu(
                        radii[:, None, None, None] + obstacle_radii[:, :, None, :]
                        - obstacle_distance
                    )
                    obstacle_penalty = obstacle_penalty + (
                        obstacle_overlap.square() * self.safety_penetration_loss_scale
                    )
                    obstacle_penalty = obstacle_penalty * obstacle_mask[:, :, None, :].to(world.dtype)
                    loss = loss + obstacle_penalty.sum(dim=-1)

                # 运动学约束在所有阶段和所有背景车辆上始终有效。
                loss = loss + self._kinematics_loss(
                    world, params, batch_size, num_samples
                )

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
