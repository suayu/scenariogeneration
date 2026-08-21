import json
import sys
from types import SimpleNamespace

import numpy as np
import torch
from policies.llm_adversarial_planner import LLMAdversarialPlanner
from policies.llm_anchor_guidance import register_llm_anchor_guidance
from policies.diffusion_model_wrapper import DiffusionModelWrapper
from policies.scenario_guidance import register_scenario_guidance
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


def test_llm_json_plan_is_parsed_with_injected_client():
    raw_plan = {
        "attack": False,
        "attack_target_id": -1,
        "strategy": "none",
        "anchors": [],
        "reason": "当前没有合适目标",
    }
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(text=json.dumps(raw_plan))],
            )
        ]
    )
    fake_client = SimpleNamespace(
        responses=SimpleNamespace(create=lambda **kwargs: response)
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
        "center": [0.0, 10.0],
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

    np.testing.assert_allclose(converted[0]["center"], [100.0, 210.0])
    assert converted[0]["yaw"] == np.pi / 2
