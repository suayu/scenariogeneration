import json
import sys
from types import SimpleNamespace

import numpy as np
import torch
from policies.llm_adversarial_planner import LLMAdversarialPlanner
from policies.llm_anchor_guidance import register_llm_anchor_guidance
from policies.diffusion_model_wrapper import DiffusionModelWrapper
from policies.scenario_guidance import (
    register_scenario_guidance,
    resolve_attack_contact_schedule,
)
from policies.obstacle_wrapper import CarlaObstacleBackend, ObstacleWrapper, ScenarioDreamerObstacleBackend
from policies.obstacles import ObstacleCatalog, StaticObstacle
from policies.safe_sim_adapter import SafeSimBatchAdapter
from policies.traffic_types import JointTrajectory, ScenarioFrame
from scenario_generator import AdversarialScenarioGenerator
from simulator import Simulator
from utils.llm_scene_viz import render_llm_scene_png


LLMAnchorLossCalculator = register_llm_anchor_guidance()
ScenarioGuidanceCalculators = register_scenario_guidance()


def _frame(history_frames=4):
    states = np.array(
        [
            [10.0, 5.0, 2.0, 0.0, 0.0, 4.5, 2.0, 1.0],
            [14.0, 5.0, 1.0, 0.0, np.pi / 2, 4.5, 2.0, 1.0],
            [40.0, 5.0, 0.0, 0.0, 0.0, 4.5, 2.0, 1.0],
        ],
        dtype=np.float32,
    )
    history = np.repeat(states[:, None], history_frames, axis=1)
    for step in range(history_frames):
        history[:, step, 0] -= (history_frames - 1 - step) * 0.2
    mask = np.ones((3, history_frames), dtype=bool)
    mask[0, :2] = False
    history[0, :2] = 0.0
    return ScenarioFrame(
        scene_id="unit-scene",
        step=3,
        dt=0.1,
        agent_ids=np.arange(3, dtype=np.int64),
        states_global=states,
        agent_types=np.eye(5, dtype=np.float32)[[1, 1, 1]],
        active_mask=np.array([True, True, False]),
        history_global=history,
        history_mask=mask,
        lanes_global=np.array(
            [[[-10.0, 5.0], [10.0, 5.0], [30.0, 5.0], [50.0, 5.0]]],
            dtype=np.float32,
        ),
        route_global=np.array([[-10.0, 5.0], [50.0, 5.0]], dtype=np.float32),
        ego_state_global=np.array([0.0, 5.0, 4.0, 0.0, 0.0, 4.8, 2.0, 1.0], dtype=np.float32),
    )


def _guidance_wrapper(mode, config=None):
    wrapper = DiffusionModelWrapper.__new__(DiffusionModelWrapper)
    wrapper.guidance_mode = mode
    wrapper.guidance_config = {} if config is None else config
    wrapper.prediction_horizon = 32
    wrapper.anchor_guidance_strength = 0.2
    wrapper.anchor_robust_delta = 1.0
    wrapper.active_guidance_functions = ["scenario_collision", "route", "scenario_ttc"]
    return wrapper


def test_four_guidance_modes_resolve_to_expected_losses():
    assert _guidance_wrapper("unguided")._resolve_guidance_profile()["functions"] == []
    assert _guidance_wrapper("default")._resolve_guidance_profile()["functions"] == [
        "scenario_collision", "route", "scenario_ttc"
    ]
    assert _guidance_wrapper("llm_joint")._resolve_guidance_profile()["functions"] == [
        "scenario_collision", "route", "scenario_ttc", "llm_anchor"
    ]
    manual = _guidance_wrapper(
        "manual",
        {"manual": {"functions": ["route", "drivearea"], "weights": [1.0, 0.5]}},
    )._resolve_guidance_profile()
    assert manual["functions"] == ["route", "drivearea"]
    assert manual["weights"] == [1.0, 0.5]


def test_scenario_ttc_targets_only_selected_background_agent():
    calculator = ScenarioGuidanceCalculators["scenario_ttc"](
        loss_timesteps=3,
        filter_timesteps=3,
    )
    state = torch.zeros((2, 3, 4), dtype=torch.float32, requires_grad=True)
    transforms = torch.eye(3).repeat(2, 1, 1)
    transforms[0, 0, 2] = 5.0
    transforms[1, 0, 2] = 8.0
    params = {
        "batch_size": 2,
        "num_samples": 1,
        "world_from_agent": transforms,
        "scenario_ego_state": torch.tensor(
            [[0.0, 0.0, 2.0, 0.0, 0.0, 4.8, 2.0, 1.0]]
        ).repeat(2, 1),
        "guidance_target_mask": torch.tensor([1.0, 0.0]),
        "dt": 0.1,
    }

    loss = calculator.calculate_loss(state[..., :2], state, params)

    assert torch.all(loss[0] < 0.0)
    torch.testing.assert_close(loss[1], torch.zeros_like(loss[1]))
    assert torch.isfinite(loss).all()


def _scenario_collision_params(batch_size, attack_mask, attack_age):
    transforms = torch.eye(3).repeat(batch_size, 1, 1)
    return {
        "batch_size": batch_size,
        "num_samples": 1,
        "world_from_agent": transforms,
        "scenario_ego_state": torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 2.0, 1.0]]
        ).repeat(batch_size, 1),
        "ego_extents": torch.tensor([[4.0, 2.0]]).repeat(batch_size, 1),
        "curr_speed": torch.zeros(batch_size),
        "yaw": torch.zeros(batch_size),
        "guidance_attack_active": torch.tensor(attack_mask, dtype=torch.float32),
        "guidance_attack_age_frames": torch.full((batch_size,), attack_age),
        "dt": 0.1,
    }


def test_attack_contact_schedule_has_protection_relaxation_and_contact_phases():
    protection = resolve_attack_contact_schedule(0, 2, 6, 2)
    relaxation_start = resolve_attack_contact_schedule(2, 2, 6, 2)
    relaxation_end = resolve_attack_contact_schedule(7, 2, 6, 2)
    contact_start = resolve_attack_contact_schedule(8, 2, 6, 2)
    contact_now = resolve_attack_contact_schedule(9, 2, 6, 2)

    assert protection == {
        "phase": "protection",
        "avoidance_weight": 1.0,
        "contact_weight": 0.0,
        "contact_step": None,
    }
    assert relaxation_start["phase"] == "relaxation"
    assert relaxation_start["avoidance_weight"] == 5.0 / 6.0
    assert relaxation_end["avoidance_weight"] == 0.0
    assert contact_start["contact_weight"] == 0.5
    assert contact_start["contact_step"] == 1
    assert contact_now["contact_weight"] == 1.0
    assert contact_now["contact_step"] == 0


def test_non_attacker_keeps_ego_avoidance_during_late_contact_phase():
    calculator = ScenarioGuidanceCalculators["scenario_collision"](
        loss_timesteps=2,
        max_speed=1000.0,
        max_acceleration=10000.0,
        max_jerk=100000.0,
        max_step_distance=100.0,
    )
    state = torch.zeros((2, 2, 4), dtype=torch.float32)
    state[0, :, 0] = 20.0
    params = _scenario_collision_params(2, [1.0, 0.0], attack_age=9)

    loss = calculator.calculate_loss(state[..., :2], state, params)

    # 非攻击车辆 1 与自车重叠，末段也必须保留显著防碰撞损失。
    assert torch.all(loss[1] > 1.0)


def test_background_pair_collision_avoidance_is_preserved_for_attacker():
    calculator = ScenarioGuidanceCalculators["scenario_collision"](
        loss_timesteps=2,
        max_speed=1000.0,
        max_acceleration=10000.0,
        max_jerk=100000.0,
        max_step_distance=100.0,
    )
    state = torch.zeros((2, 2, 4), dtype=torch.float32)
    state[:, :, 0] = 20.0
    params = _scenario_collision_params(2, [1.0, 0.0], attack_age=9)

    loss = calculator.calculate_loss(state[..., :2], state, params)

    # 两辆背景车在远离自车处重叠；攻击者身份不能关闭背景车之间的约束。
    assert torch.all(loss > 100.0)


def test_late_contact_prefers_shallow_contact_and_rejects_deep_penetration():
    calculator = ScenarioGuidanceCalculators["scenario_collision"](
        loss_timesteps=1,
        max_speed=1000.0,
        max_acceleration=10000.0,
        max_jerk=100000.0,
        max_step_distance=100.0,
    )
    params = _scenario_collision_params(1, [1.0], attack_age=9)
    desired_distance = float(np.sqrt(5.0) * 2.0 - 0.05)

    shallow = torch.zeros((1, 1, 4), dtype=torch.float32)
    shallow[..., 0] = desired_distance
    far = shallow.clone()
    far[..., 0] += 3.0
    deep = shallow.clone()
    deep[..., 0] = 0.0

    shallow_loss = calculator.calculate_loss(shallow[..., :2], shallow, params)
    far_loss = calculator.calculate_loss(far[..., :2], far, params)
    deep_loss = calculator.calculate_loss(deep[..., :2], deep, params)

    assert shallow_loss.item() < far_loss.item()
    assert shallow_loss.item() < deep_loss.item()


def test_kinematics_loss_penalizes_teleportation_in_every_attack_phase():
    calculator = ScenarioGuidanceCalculators["scenario_collision"](
        loss_timesteps=2,
        max_speed=20.0,
        max_acceleration=6.0,
        max_jerk=12.0,
        max_step_distance=2.0,
    )
    params = _scenario_collision_params(1, [1.0], attack_age=9)
    params["scenario_ego_state"][:, 0] = 1000.0
    params["curr_speed"][:] = 1.0
    normal = torch.zeros((1, 2, 4), dtype=torch.float32)
    normal[0, :, 0] = torch.tensor([0.1, 0.2])
    teleport = torch.zeros((1, 2, 4), dtype=torch.float32)
    teleport[0, :, 0] = torch.tensor([5.0, 10.0])

    normal_loss = calculator.calculate_loss(normal[..., :2], normal, params)
    teleport_loss = calculator.calculate_loss(teleport[..., :2], teleport, params)

    assert teleport_loss.sum() > normal_loss.sum() + 100.0


def test_collision_guidance_supports_multiple_diffusion_samples():
    calculator = ScenarioGuidanceCalculators["scenario_collision"](
        loss_timesteps=4,
    )
    batch_size = 2
    num_samples = 3
    state = torch.zeros(
        (batch_size * num_samples, 4, 4), dtype=torch.float32, requires_grad=True
    )
    transforms = torch.eye(3).repeat(batch_size * num_samples, 1, 1)
    transforms[:num_samples, 0, 2] = 10.0
    transforms[num_samples:, 0, 2] = 20.0
    params = _scenario_collision_params(batch_size, [1.0, 0.0], attack_age=0)
    params["num_samples"] = num_samples
    params["world_from_agent"] = transforms
    params["yaw"] = torch.zeros(batch_size * num_samples)

    loss = calculator.calculate_loss(state[..., :2], state, params)

    assert loss.shape == (batch_size * num_samples, 4)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert torch.isfinite(state.grad).all()


def test_wrapper_selects_one_shared_sample_from_joint_guidance_loss():
    class FakeLossCalculator:
        def calculate_loss(self, action, state, guidance_data):
            return state[..., 0].square()

    class FakePolicyNet:
        Loss_Calculater = FakeLossCalculator()

        @staticmethod
        def _prepare_guidance_data(model_batch):
            return {"batch_size": 2}

    wrapper = DiffusionModelWrapper.__new__(DiffusionModelWrapper)
    wrapper.guidance_enabled = True
    positions = torch.zeros((2, 3, 1, 2))
    positions[:, 0, 0, 0] = torch.tensor([0.0, 10.0])
    positions[:, 1, 0, 0] = torch.tensor([2.0, 2.0])
    positions[:, 2, 0, 0] = torch.tensor([5.0, 5.0])
    yaws = torch.zeros((2, 3, 1, 1))
    action = SimpleNamespace(
        positions=torch.zeros((2, 1, 2)),
        yaws=torch.zeros((2, 1, 1)),
    )

    selected, selected_index = wrapper._select_joint_guided_action(
        FakePolicyNet(),
        {},
        action,
        {"action_samples": {"positions": positions, "yaws": yaws}},
    )

    assert selected_index == 1
    torch.testing.assert_close(selected.positions, positions[:, 1])


def test_llm_joint_without_attack_does_not_create_fallback_adversarial_target():
    frame = _frame()
    adapter = SafeSimBatchAdapter(
        history_frames=4,
        max_neighbors=2,
        raster_size=64,
        pixel_size=0.5,
    )
    batch = adapter.build(frame)
    inactive_anchor = {"active": False, "target_id": None, "valid_steps": 0}

    target_id = _guidance_wrapper("llm_joint")._select_adversarial_target_id(
        frame, batch, inactive_anchor
    )

    assert target_id is None


def test_default_guidance_keeps_nearest_agent_ttc_target():
    frame = _frame()
    adapter = SafeSimBatchAdapter(
        history_frames=4,
        max_neighbors=2,
        raster_size=64,
        pixel_size=0.5,
    )
    batch = adapter.build(frame)
    inactive_anchor = {"active": False, "target_id": None, "valid_steps": 0}

    target_id = _guidance_wrapper("default")._select_adversarial_target_id(
        frame, batch, inactive_anchor
    )

    assert target_id == 0


def test_llm_joint_uses_valid_llm_attack_target():
    frame = _frame()
    adapter = SafeSimBatchAdapter(
        history_frames=4,
        max_neighbors=2,
        raster_size=64,
        pixel_size=0.5,
    )
    batch = adapter.build(frame)
    active_anchor = {"active": True, "target_id": 1, "valid_steps": 5}

    target_id = _guidance_wrapper("llm_joint")._select_adversarial_target_id(
        frame, batch, active_anchor
    )

    assert target_id == 1


def test_adapter_contract_and_global_round_trip():
    frame = _frame()
    adapter = SafeSimBatchAdapter(
        history_frames=4,
        max_neighbors=2,
        raster_size=64,
        pixel_size=0.5,
    )
    batch = adapter.build(frame)

    assert batch.row_to_agent_id.tolist() == [0, 1]
    assert tuple(batch.data["image"].shape) == (2, 7, 64, 64)
    assert tuple(batch.data["agent_hist"].shape) == (2, 4, 9)
    assert tuple(batch.data["scenario_curr_acceleration_world"].shape) == (2, 2)
    assert tuple(batch.data["neigh_hist"].shape) == (2, 2, 4, 9)
    assert batch.data["image"].isfinite().all()

    local_positions = np.array(
        [
            [[1.0, 0.0], [2.0, 0.0]],
            [[1.0, 0.0], [2.0, 0.0]],
        ],
        dtype=np.float32,
    )
    local_yaws = np.zeros((2, 2, 1), dtype=np.float32)
    joint = adapter.decode(frame, batch, local_positions, local_yaws)
    np.testing.assert_allclose(joint.positions_global[0, 0], [11.0, 5.0], atol=1e-5)
    np.testing.assert_allclose(joint.positions_global[1, 0], [14.0, 6.0], atol=1e-5)
    np.testing.assert_allclose(joint.yaws_global[:, 0, 0], [0.0, np.pi / 2], atol=1e-5)


def test_static_obstacle_is_rasterized_and_forwarded_to_safe_sim():
    frame = _frame()
    frame = ScenarioFrame(
        **{**frame.__dict__, "static_obstacles_global": np.array(
            [[12.0, 5.0, 0.0, 2.0, 1.0]], dtype=np.float32
        )}
    )
    adapter = SafeSimBatchAdapter(
        history_frames=4, max_neighbors=2, raster_size=64, pixel_size=0.5
    )
    batch = adapter.build(frame)

    # 对第一辆车而言，障碍物位于局部坐标 (2, 0)，应占据所有历史动态通道。
    pixel = adapter._to_pixels(np.array([2.0, 0.0], dtype=np.float32))
    assert torch.all(batch.data["image"][0, :4, pixel[1], pixel[0]] == -1.0)
    assert tuple(batch.data["static_obstacles_world"].shape) == (2, 1, 5)
    assert torch.all(batch.data["static_obstacle_mask"])
    torch.testing.assert_close(
        batch.data["static_obstacles_world"][0, 0],
        torch.tensor([12.0, 5.0, 0.0, 2.0, 1.0]),
    )


def test_empty_static_obstacles_preserve_adapter_raster_output():
    frame = _frame()
    adapter = SafeSimBatchAdapter(
        history_frames=4, max_neighbors=2, raster_size=64, pixel_size=0.5
    )
    first = adapter.build(frame)
    second = adapter.build(frame)

    torch.testing.assert_close(first.data["image"], second.data["image"])
    assert tuple(first.data["static_obstacles_world"].shape) == (2, 0, 5)
    assert tuple(first.data["static_obstacle_mask"].shape) == (2, 0)


def test_static_obstacle_collision_penalty_rejects_crossing_trajectory():
    calculator = ScenarioGuidanceCalculators["scenario_collision"](
        loss_timesteps=2,
        max_speed=1000.0,
        max_acceleration=10000.0,
        max_jerk=100000.0,
        max_step_distance=100.0,
    )
    params = _scenario_collision_params(1, [0.0], attack_age=0)
    params["scenario_ego_state"][:, 0] = 1000.0
    params["static_obstacles_world"] = torch.tensor(
        [[[0.0, 0.0, 0.0, 2.0, 2.0]]], dtype=torch.float32
    )
    params["static_obstacle_mask"] = torch.tensor([[True]])
    crossing = torch.zeros((1, 2, 4), dtype=torch.float32)
    bypass = torch.zeros((1, 2, 4), dtype=torch.float32)
    bypass[..., 0] = 20.0

    crossing_loss = calculator.calculate_loss(crossing[..., :2], crossing, params)
    bypass_loss = calculator.calculate_loss(bypass[..., :2], bypass, params)

    assert crossing_loss.sum() > bypass_loss.sum() + 100.0


def test_simulator_consumes_only_first_joint_step():
    sim = Simulator.__new__(Simulator)
    sim.t = 3
    sim.agent_active = np.array([True, True, False])
    sim.pending_joint_trajectory = None
    initial = _frame().states_global.copy()
    sim.data_dict = {"agent": [initial]}

    joint = JointTrajectory(
        source_step=3,
        agent_ids=np.array([0, 1], dtype=np.int64),
        positions_global=np.array(
            [[[11.0, 5.0], [99.0, 99.0]], [[14.0, 6.0], [88.0, 88.0]]],
            dtype=np.float32,
        ),
        yaws_global=np.zeros((2, 2, 1), dtype=np.float32),
        velocities_global=np.ones((2, 2, 2), dtype=np.float32),
        valid_mask=np.ones((2, 2), dtype=bool),
    )
    sim.inject_joint_trajectory(joint)
    next_states = sim._consume_joint_trajectory(source_step=3)

    np.testing.assert_allclose(next_states[:2, :2], [[11.0, 5.0], [14.0, 6.0]])
    np.testing.assert_allclose(next_states[2], initial[2])
    assert sim.pending_joint_trajectory is None


def test_adapter_interpolates_global_llm_anchors_for_target_only():
    frame = _frame()
    adapter = SafeSimBatchAdapter(
        history_frames=4,
        max_neighbors=2,
        raster_size=64,
        pixel_size=0.5,
    )
    batch = adapter.build(frame)
    tensors, metadata = adapter.build_anchor_guidance(
        frame=frame,
        safe_batch=batch,
        attack_intent={
            "target_id": 0,
            "anchors": [[10.0, 5.0], [11.0, 5.0], [12.0, 5.0], [13.0, 5.0]],
            "strategy": "hard_brake",
            "source_step": frame.step,
        },
        prediction_horizon=5,
        anchor_interval_seconds=1.0,
    )

    positions = tensors["llm_anchor_positions_local"].numpy()
    mask = tensors["llm_anchor_mask"].numpy()
    np.testing.assert_allclose(positions[0, :, 0], np.arange(0.1, 0.6, 0.1), atol=1e-5)
    np.testing.assert_allclose(positions[0, :, 1], 0.0, atol=1e-5)
    assert mask[0].all()
    assert not mask[1].any()
    assert metadata == {
        "active": True,
        "target_id": 0,
        "valid_steps": 5,
        "elapsed_seconds": 0.0,
    }


def test_adapter_disables_expired_or_inactive_anchor_guidance():
    frame = _frame()
    adapter = SafeSimBatchAdapter(
        history_frames=4,
        max_neighbors=2,
        raster_size=64,
        pixel_size=0.5,
    )
    batch = adapter.build(frame)
    tensors, metadata = adapter.build_anchor_guidance(
        frame=frame,
        safe_batch=batch,
        attack_intent={
            "target_id": 2,
            "anchors": [[40.0, 5.0], [41.0, 5.0]],
            "strategy": "cut_in",
            "source_step": frame.step,
        },
        prediction_horizon=5,
        anchor_interval_seconds=1.0,
    )

    assert metadata["active"] is False
    assert not tensors["llm_anchor_mask"].any()


def test_llm_anchor_loss_affects_only_target_and_scales_with_strength():
    state = torch.zeros((4, 3, 4), dtype=torch.float32)
    state[:2, :, 0] = 2.0
    state.requires_grad_(True)
    guidance_data = {
        "batch_size": 2,
    }
    calculator = LLMAnchorLossCalculator(loss_scale=2.0, robust_delta=1.0)
    calculator.set_anchor_targets(
        torch.zeros((2, 3, 2), dtype=torch.float32),
        torch.tensor(
            [[True, True, True], [False, False, False]],
            dtype=torch.bool,
        ),
        num_samples=2,
    )

    loss = calculator.calculate_loss(torch.zeros_like(state[..., :2]), state, guidance_data)

    torch.testing.assert_close(loss[:2], torch.full((2, 3), 3.0))
    torch.testing.assert_close(loss[2:], torch.zeros((2, 3)))
    loss.sum().backward()
    assert torch.all(state.grad[:2, :, 0] > 0)
    torch.testing.assert_close(state.grad[2:], torch.zeros_like(state.grad[2:]))


def test_llm_anchor_direct_clean_action_gradient_reduces_position_error():
    action = torch.zeros((2, 2, 2), dtype=torch.float32, requires_grad=True)
    state = torch.cat([action + 2.0, torch.zeros_like(action)], dim=-1)
    noisy_action = torch.zeros_like(action, requires_grad=True)
    guidance_data = {
        "batch_size": 2,
    }
    calculator = LLMAnchorLossCalculator(loss_scale=1.0, robust_delta=1.0)
    calculator.set_anchor_targets(
        torch.zeros((2, 2, 2), dtype=torch.float32),
        torch.tensor(
            [[True, True], [False, False]],
            dtype=torch.bool,
        ),
        num_samples=1,
    )

    gradient = calculator.calculate_grad(
        action,
        state,
        noisy_action,
        "clean_action_guide",
        guidance_data,
    )

    assert torch.all(gradient[0] < 0)
    torch.testing.assert_close(gradient[1], torch.zeros_like(gradient[1]))


def test_simulator_can_clear_attack_intent():
    sim = Simulator.__new__(Simulator)
    sim.attack_intent = {"target_id": 1}

    sim.clear_attack_intent()

    assert sim.attack_intent is None


def test_llm_limit_disables_attack_without_stopping_simulation():
    generator = AdversarialScenarioGenerator(None, [], llm_planner=object())
    generator._llm_call_times = generator._max_call_times

    assert generator.step(env=object(), current_t=0) is True
    assert generator._attacks_enabled is False


def test_attack_intent_is_cleared_exactly_at_ten_frame_boundary():
    generator = AdversarialScenarioGenerator(None, [], llm_planner=object())
    generator.last_attack_frame = 0
    generator.last_query_frame = 10
    env = SimpleNamespace(attack_intent={"target_id": 1})

    def clear_attack_intent():
        env.attack_intent = None

    env.clear_attack_intent = clear_attack_intent

    assert generator.step(env=env, current_t=9) is True
    assert env.attack_intent is not None
    assert generator.step(env=env, current_t=10) is True
    assert env.attack_intent is None


def test_llm_json_plan_is_parsed_with_injected_client():
    raw_plan = {
        "attack": False,
        "attack_target_id": -1,
        "strategy": "none",
        "anchors": [],
        "reason": "当前没有合适目标",
    }
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(raw_plan)))]
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response))
    )
    planner = LLMAdversarialPlanner(client=fake_client)
    env_state = {
        "ego_state": [0.0, 0.0, 1.0, 0.0, 0.0],
        "route": [[0.0, 0.0], [10.0, 0.0]],
        "agents": [],
    }

    assert planner.generate_attack_plan(env_state, ["生成危险场景"]) == raw_plan


def test_llm_parser_accepts_markdown_json_fence():
    raw = """```json
{"attack": false, "attack_target_id": -1, "strategy": "none", "anchors": []}
```"""

    parsed = LLMAdversarialPlanner._parse_json_object(raw)

    assert parsed["attack"] is False
    assert parsed["attack_target_id"] == -1


def test_llm_local_coordinates_preserve_history_and_rotate_velocity():
    planner = LLMAdversarialPlanner(client=SimpleNamespace())
    normalized = planner._normalize_env_state({
        "ego_state": [0.0, 0.0, 2.0, 0.0, 0.0],
        "route": [[1.0, 0.0]],
        "agents": [{
            "id": 3,
            "type": "vehicle",
            "state": [1.0, 0.0, 1.0, 0.0, 0.0],
            "history": [{
                "state": [0.5, 0.0, 1.0, 0.0, 0.0],
                "valid": True,
            }],
        }],
        "history_order": "oldest_to_newest",
        "current_step": 9,
    })

    np.testing.assert_allclose(normalized["ego_state"][2:4], [0.0, 2.0], atol=1e-6)
    np.testing.assert_allclose(normalized["agents"][0]["state"][:4], [0.0, 1.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(
        normalized["agents"][0]["history"][0]["state"][:4],
        [0.0, 0.5, 0.0, 1.0],
        atol=1e-6,
    )
    assert normalized["agents"][0]["type"] == 0
    assert normalized["current_step"] == 9


def test_prompt_switches_explicitly_between_text_and_multimodal_modes():
    env_state = {
        "ego_state": [0.0, 0.0, 1.0, 0.0, 0.0],
        "route": [[0.0, 0.0], [10.0, 0.0]],
        "agents": [],
    }
    planner = LLMAdversarialPlanner(client=SimpleNamespace(), use_multimodal=False)
    text_prompt = planner._build_prompt(env_state, ["优先生成切入场景"], include_image=False)
    image_prompt = planner._build_prompt(env_state, ["优先生成切入场景"], include_image=True)

    assert "No image is provided in text-only mode" in text_prompt
    assert "attached BEV image" not in text_prompt
    assert "A current-frame BEV image" in image_prompt
    assert "优先生成切入场景" in text_prompt


def test_multimodal_request_contains_current_prompt_and_png():
    raw_plan = {
        "attack": False,
        "attack_target_id": -1,
        "strategy": "none",
        "anchors": [],
        "reason": "当前没有合适目标",
    }
    captured = {}

    def create_completion(**kwargs):
        captured.update(kwargs)
        message = SimpleNamespace(content=json.dumps(raw_plan), reasoning_content="")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_completion),
        )
    )
    planner = LLMAdversarialPlanner(client=fake_client, use_multimodal=True)
    env_state = {
        "ego_state": [0.0, 0.0, 1.0, 0.0, 0.0],
        "route": [[0.0, 0.0], [10.0, 0.0]],
        "agents": [],
    }
    result = planner.generate_attack_plan(
        env_state,
        ["生成危险场景"],
        scene_image=b"\x89PNG\r\n\x1a\n",
    )

    content = captured["messages"][0]["content"]
    assert result == raw_plan
    assert content[0]["type"] == "text"
    assert "生成危险场景" in content[0]["text"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_llm_scene_renderer_outputs_png_without_route_input():
    ego = np.array([0.0, 0.0, 0.0, 2.0, np.pi / 2, 4.8, 2.0, 1.0])
    agents = np.array([[5.0, 2.0, 1.0, 0.0, 0.0, 4.5, 1.9, 1.0]])
    lanes = np.array([[[-10.0, 0.0], [0.0, 0.0], [10.0, 0.0]]])
    masks = np.ones((1, 3), dtype=bool)

    image = render_llm_scene_png(ego, agents, np.array([7]), lanes, masks)

    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(image) > 1000


def test_llm_rejects_semantically_invalid_hard_brake_plan():
    plan = {
        "attack": True,
        "attack_target_id": 3,
        "strategy": "hard_brake",
        "anchors": [[-16.0, 31.0], [-16.0, 29.0], [-16.0, 27.0], [-16.0, 25.0]],
        "reason": "横向远处的静止目标不应被用于急刹攻击",
    }
    env_state = {
        "ego_state": [0.0, 0.0, 0.0, 8.0, np.pi / 2],
        "route": [[0.0, 0.0], [0.0, 30.0]],
        "agents": [{
            "id": 3,
            "type": 0,
            "state": [-16.0, 31.0, 0.0, 0.0, np.pi / 2],
            "history": [],
        }],
    }

    try:
        LLMAdversarialPlanner._validate_attack_plan(plan, env_state)
    except ValueError as error:
        assert "同车道前方" in str(error) or "有效交互" in str(error)
    else:
        raise AssertionError("不合理的 hard_brake 计划未被拒绝")


def test_llm_scene_filter_does_not_mutate_simulator_active_mask():
    sim = Simulator.__new__(Simulator)
    sim.ego_state = np.array([0.0, 0.0, 0.0, 2.0, np.pi / 2, 4.8, 2.0, 1.0])
    sim.local_frame = {"center": np.zeros(2), "yaw": np.pi / 2}
    sim.viz_state = {
        "agent_active": np.array([True, True]),
        "agent_states": np.array([
            [5.0, 2.0, 1.0, 0.0, 0.0, 4.5, 1.9, 1.0],
            [-5.0, 2.0, -1.0, 0.0, np.pi, 4.5, 1.9, 1.0],
        ]),
        "lanes": np.array([[[-10.0, 0.0], [0.0, 0.0], [10.0, 0.0]]]),
        "lanes_mask": np.ones((1, 3), dtype=bool),
    }

    image = sim.render_llm_scene_image(agent_ids=[1])

    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    assert sim.viz_state["agent_active"].tolist() == [True, True]


def test_obstacle_catalog_materializes_six_abstract_templates():
    catalog = ObstacleCatalog()
    expected_types = {
        "transverse_barrier",
        "longitudinal_barrier",
        "cone_line",
        "disabled_vehicle",
        "debris_cluster",
        "construction_block",
    }

    assert expected_types == set(catalog.type_names)
    for primitive_type in expected_types:
        obstacles = catalog.materialize(
            {
                "type": primitive_type,
                "center": [1.0, 12.0],
                "yaw": 0.0,
                "count": 1,
                "spacing": 1.0,
            },
            lambda name: f"{name}-0",
        )
        assert obstacles
        if primitive_type == "construction_block":
            assert {item.primitive_type for item in obstacles} == {"transverse_barrier", "cone_line"}
        else:
            assert all(item.primitive_type == primitive_type for item in obstacles)


def test_scenario_dreamer_obstacle_wrapper_supports_crud_and_collision():
    simulator = SimpleNamespace(static_obstacle_elements={})
    wrapper = ObstacleWrapper(ScenarioDreamerObstacleBackend(simulator))
    obstacle = StaticObstacle(
        obstacle_id="manual-0",
        primitive_type="disabled_vehicle",
        object_kind="disabled_vehicle",
        center_global=(0.0, 0.0),
        yaw=0.0,
        length=4.8,
        width=2.0,
    )

    wrapper.create(obstacle)
    assert wrapper.query("manual-0") == obstacle
    assert "manual-0" in simulator.static_obstacle_elements
    assert wrapper.colliding_ids(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 4.8, 2.0, 1.0])) == ["manual-0"]
    updated = wrapper.update("manual-0", center_global=(10.0, 0.0))
    assert updated.center_global == (10.0, 0.0)
    assert wrapper.delete("manual-0") is True
    assert wrapper.list() == []


def test_carla_obstacle_backend_uses_the_same_create_update_delete_interface(monkeypatch):
    class FakeTransform:
        def __init__(self, location, rotation):
            self.location = location
            self.rotation = rotation

    fake_carla = SimpleNamespace(
        Location=lambda **kwargs: kwargs,
        Rotation=lambda **kwargs: kwargs,
        Transform=FakeTransform,
    )
    monkeypatch.setitem(sys.modules, "carla", fake_carla)

    class FakeActor:
        is_alive = True

        def __init__(self):
            self.transforms = []
            self.destroyed = False

        def set_transform(self, transform):
            self.transforms.append(transform)

        def destroy(self):
            self.destroyed = True

    class FakeWorld:
        def __init__(self):
            self.actor = FakeActor()
            self.requests = []

        def get_blueprint_library(self):
            return SimpleNamespace(find=lambda name: name)

        def try_spawn_actor(self, blueprint, transform):
            self.requests.append((blueprint, transform))
            return self.actor

    world = FakeWorld()
    backend = CarlaObstacleBackend(world)
    obstacle = StaticObstacle(
        obstacle_id="carla-0",
        primitive_type="cone_line",
        object_kind="traffic_cone",
        center_global=(3.0, 8.0),
        yaw=0.0,
        length=0.45,
        width=0.45,
    )

    handle = backend.create(obstacle)
    assert handle is world.actor
    assert world.requests[0][0] == "static.prop.trafficcone01"
    assert backend.update(handle, obstacle) is handle
    backend.delete(handle)
    assert world.actor.destroyed is True


def test_obstacle_prompt_is_disabled_by_default_and_enabled_explicitly():
    env_state = {
        "ego_state": [0.0, 0.0, 0.0, 2.0, np.pi / 2],
        "route": [[0.0, 0.0], [0.0, 20.0]],
        "agents": [],
        "static_obstacles": [],
    }
    default_planner = LLMAdversarialPlanner(client=SimpleNamespace())
    enabled_planner = LLMAdversarialPlanner(client=SimpleNamespace(), use_obstacles=True)

    default_prompt = default_planner._build_prompt(env_state, ["生成危险场景"])
    enabled_prompt = enabled_planner._build_prompt(env_state, ["生成危险场景"])

    assert "Static-obstacle scheduling is disabled" in default_prompt
    assert "`obstacle_plan` (list)" not in default_prompt
    assert "Optional Static-Obstacle Scheduling" in enabled_prompt
    assert "transverse_barrier" in enabled_prompt
    assert '"obstacle_plan"' in enabled_prompt


def test_llm_validates_and_converts_an_enabled_obstacle_plan():
    planner = LLMAdversarialPlanner(client=SimpleNamespace(), use_obstacles=True)
    local_plan = [{
        "type": "cone_line",
        "center": [0.0, 17.0],
        "yaw": np.pi / 2,
        "count": 3,
        "spacing": 1.0,
    }]
    env_state = {
        "ego_state": [100.0, 200.0, 0.0, 5.0, np.pi / 2],
        "route": [[100.0, 200.0], [100.0, 220.0]],
        "agents": [],
        "static_obstacles": [],
    }

    normalized = planner._normalize_env_state(env_state)
    planner._validate_obstacle_plan(local_plan, normalized)
    converted = planner._convert_obstacle_plan_to_global(local_plan, 100.0, 200.0, np.pi / 2)

    np.testing.assert_allclose(converted[0]["center"], [100.0, 217.0])
    assert converted[0]["yaw"] == np.pi / 2


def test_obstacle_validation_uses_materialized_occupancy_and_filters_only_group():
    planner = LLMAdversarialPlanner(client=SimpleNamespace(), use_obstacles=True)
    normalized = {"static_obstacles": []}
    # 中心在 17 m 处的故障车辆前缘仅 14.6 m，必须按真实矩形占用被过滤。
    near_vehicle = [{
        "type": "disabled_vehicle", "center": [0.0, 17.0], "yaw": np.pi / 2,
        "count": 1, "spacing": 1.0,
    }]
    assert planner._validate_obstacle_plan(near_vehicle, normalized) == []
    # 锥桶队列的最近锥桶同样不能借由组中心绕过 15 m 阈值。
    near_cones = [{
        "type": "cone_line", "center": [0.0, 16.0], "yaw": np.pi / 2,
        "count": 3, "spacing": 1.0,
    }]
    assert planner._validate_obstacle_plan(near_cones, normalized) == []
    safe_cones = [{
        "type": "cone_line", "center": [0.0, 18.0], "yaw": np.pi / 2,
        "count": 3, "spacing": 1.0,
    }]
    assert planner._validate_obstacle_plan(safe_cones, normalized) == safe_cones


def test_unsafe_obstacle_group_does_not_discard_valid_dynamic_plan():
    planner = LLMAdversarialPlanner(client=SimpleNamespace(), use_obstacles=True)
    raw_plan = {
        "attack": True,
        "attack_target_id": 1,
        "strategy": "lane_change",
        "anchors": [[0.0, 8.0], [0.0, 12.0], [0.0, 16.0], [0.0, 20.0]],
        "reason": "动态攻击保持有效",
        # 中心虽为 17m，故障车前缘却只有 14.6m，必须只过滤该障碍物组。
        "obstacle_plan": [{
            "type": "disabled_vehicle", "center": [0.0, 17.0], "yaw": np.pi / 2,
            "count": 1, "spacing": 1.0,
        }],
    }
    planner._request_attack_plan = lambda *args, **kwargs: ("", dict(raw_plan))
    env_state = {
        "ego_state": [100.0, 200.0, 0.0, 5.0, np.pi / 2],
        "route": [[100.0, 200.0], [100.0, 230.0]],
        "agents": [{"id": 1, "type": 0, "state": [100.0, 208.0, 0.0, 4.0, np.pi / 2]}],
        "static_obstacles": [],
    }
    result = planner.generate_attack_plan(env_state, ["生成动态攻击"])
    assert result is not None
    assert result["attack"] is True
    assert result["obstacle_plan"] == []
    assert result["attack_target_id"] == 1


def test_llm_provider_aliases_select_explicit_contracts():
    """三类提供方必须显式选择，避免模型失败后跨服务静默切换。"""
    client = SimpleNamespace()
    assert LLMAdversarialPlanner(provider="gpt", client=client).provider == "openai"
    assert LLMAdversarialPlanner(provider="deepseek", client=client).provider == "deepseek"
    assert LLMAdversarialPlanner(provider="aliyun", client=client).provider == "dashscope"


def test_openai_provider_uses_responses_api_for_json_plan():
    raw_plan = {"attack": False, "attack_target_id": -1, "strategy": "none", "anchors": []}
    captured = {}

    def create_response(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(output_text=json.dumps(raw_plan))

    client = SimpleNamespace(responses=SimpleNamespace(create=create_response))
    planner = LLMAdversarialPlanner(provider="openai", client=client)
    reasoning, plan = planner._request_attack_plan_once("gpt-5.6", "仅返回 JSON")

    assert reasoning == ""
    assert plan == raw_plan
    assert captured["input"][0]["content"][0]["type"] == "input_text"
    assert captured["reasoning"]["effort"] == "low"


def test_deepseek_provider_keeps_compatible_chat_contract():
    raw_plan = {"attack": False, "attack_target_id": -1, "strategy": "none", "anchors": []}
    captured = {}

    def create_completion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(raw_plan)))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_completion)))
    planner = LLMAdversarialPlanner(provider="deepseek", client=client)
    _, plan = planner._request_attack_plan_once("deepseek-v4-pro", "仅返回 JSON")

    assert plan == raw_plan
    assert "extra_body" not in captured


def test_gemini_provider_uses_compatible_chat_without_dashscope_fields():
    """Gemini 通过官方 OpenAI 兼容端点复用 JSON 规划契约。"""
    raw_plan = {"attack": False, "attack_target_id": -1, "strategy": "none", "anchors": []}
    captured = {}

    def create_completion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(raw_plan)))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_completion)))
    planner = LLMAdversarialPlanner(provider="gemini", client=client)
    _, plan = planner._request_attack_plan_once("gemini-2.5-flash", "仅返回 JSON")

    assert plan == raw_plan
    assert planner.api_key_env == "GEMINI_API_KEY"
    assert "extra_body" not in captured


def test_qwen_provider_uses_modelscope_contract_and_disables_thinking():
    """Qwen3.6 经 ModelScope API Inference 走 OpenAI 兼容请求。"""
    raw_plan = {"attack": False, "attack_target_id": -1, "strategy": "none", "anchors": []}
    captured = {}

    def create_completion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(raw_plan)))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_completion)))
    planner = LLMAdversarialPlanner(provider="modelscope", client=client)
    _, plan = planner._request_attack_plan_once("Qwen/Qwen3.6-27B-FP8", "仅返回 JSON")

    assert plan == raw_plan
    assert planner.provider == "qwen"
    assert planner.api_key_env == "MODELSCOPE_ACCESS_TOKEN"
    assert captured["extra_body"] == {"enable_thinking": False}
