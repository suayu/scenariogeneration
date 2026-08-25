from types import SimpleNamespace

import numpy as np

from policies.llm_adversarial_planner import LLMAdversarialPlanner
from policies.risk_metrics import (
    AdversarialRiskMetrics,
    EvasiveAcceleration,
    RoadUserState,
    RouteReachability,
    compute_scenario_danger_score,
)
from policies.traffic_types import JointTrajectory
from scenario_generator import AdversarialScenarioGenerator


class _UnavailableModelError(RuntimeError):
    """模拟单个模型额度耗尽。"""

    status_code = 429


def test_composite_danger_score_is_zero_for_safe_metrics():
    metrics = {
        "collision rate": 0.0,
        "off route rate": 0.0,
        "completed rate": 1.0,
        "progress": 1.0,
        "near_miss_rate": 0.0,
        "avg_min_ttc": float("inf"),
        "max_evasive_acceleration_mps2": 0.0,
        "mean_evasive_acceleration_mps2": 0.0,
        "reachability_event_count": 0,
        "dangerous_scene_solvable": 0.0,
    }
    assert compute_scenario_danger_score(metrics) == 0.0


def test_composite_danger_score_ignores_uncomputed_reachability_defaults():
    metrics = {
        "collision rate": 1.0,
        "near_miss_rate": 0.0,
        "avg_min_ttc": float("inf"),
        "max_evasive_acceleration_mps2": 0.0,
        "mean_evasive_acceleration_mps2": 0.0,
        "reachability_event_count": 0,
        "reachability_difficulty": float("nan"),
        "dangerous_scene_solvable": 0.0,
    }
    # 可达性未执行且驾驶质量字段缺失时，只在已有有效分量内归一化。
    assert np.isclose(compute_scenario_danger_score(metrics), 0.30 / 0.76)


def test_composite_danger_score_uses_all_independent_risk_components():
    metrics = {
        "collision rate": 1.0,
        "collision_rate": 0.0,
        "near_miss_rate": 1.0,
        "avg_min_ttc": 0.0,
        "max_evasive_acceleration_mps2": 3.0,
        "mean_evasive_acceleration_mps2": 3.0,
        "reachability_event_count": 1,
        "reachability_difficulty": 1.0,
        "mean_reachability_difficulty": 1.0,
        "dangerous_scene_solvable": 0.0,
        "off route rate": 1.0,
        "completed rate": 0.0,
        "progress": 0.0,
    }
    expected = 0.30 + 0.15 + 0.15 + 0.16 * (1.0 - np.exp(-1.0)) + 0.16 + 0.05 + 0.03
    assert np.isclose(compute_scenario_danger_score(metrics), expected)


def test_ttc_uses_positive_closing_speed_and_rejects_departing_agent():
    generator = AdversarialScenarioGenerator.__new__(AdversarialScenarioGenerator)
    env = SimpleNamespace(
        ego_state=np.asarray([0.0, 0.0, 10.0, 0.0]),
        data_dict={"agent": [np.asarray([[10.0, 0.0, 0.0, 0.0]])]},
        agent_active=np.asarray([True]),
    )
    assert np.isclose(generator._compute_min_ttc(env), 1.0)
    env.data_dict["agent"][-1][0, 2] = 20.0
    assert np.isinf(generator._compute_min_ttc(env))


def test_collision_frame_is_also_counted_as_near_miss():
    generator = AdversarialScenarioGenerator.__new__(AdversarialScenarioGenerator)
    generator.risk_metrics = SimpleNamespace(evaluate_ea=lambda env: 0.0)
    generator.collision_list = []
    generator.near_miss_list = []
    generator._current_episode_min_ttc = float("inf")
    generator.evaluate_reaction(SimpleNamespace(), {"collision": True})
    assert generator.collision_list == [1.0]
    assert generator.near_miss_list == [1.0]


def test_reachability_metrics_report_event_maximum_and_mean():
    metrics = AdversarialRiskMetrics.__new__(AdversarialRiskMetrics)
    metrics.ea_values = []
    metrics.ea_undefined_frames = 0
    metrics.reachability_events = [
        {
            "difficulty": 0.2,
            "original_area_m2": 10.0,
            "dangerous_area_m2": 8.0,
            "dangerous_solvable": True,
        },
        {
            "difficulty": 0.8,
            "original_area_m2": 10.0,
            "dangerous_area_m2": 2.0,
            "dangerous_solvable": False,
        },
    ]
    result = metrics.compute_metrics()
    assert np.isclose(result["reachability_difficulty"], 0.8)
    assert np.isclose(result["mean_reachability_difficulty"], 0.5)


class _FallbackPlanner(LLMAdversarialPlanner):
    """通过内存假客户端验证模型轮换，不发起网络请求。"""

    def __init__(self):
        super().__init__(
            model_name="model-a",
            model_names=["model-a", "model-b"],
            client=object(),
        )
        self.calls = []

    def _request_attack_plan_once(self, model_name, prompt, scene_image=None):
        self.calls.append(model_name)
        if model_name == "model-a":
            raise _UnavailableModelError("额度耗尽")
        return "", {"attack": False}


def test_llm_model_pool_falls_back_and_remembers_success():
    planner = _FallbackPlanner()
    assert planner._request_attack_plan("测试") == ("", {"attack": False})
    assert planner.calls == ["model-a", "model-b"]
    assert planner.model_name == "model-b"
    planner.calls.clear()
    planner._request_attack_plan("再次测试")
    assert planner.calls == ["model-b"]


def test_ea_is_zero_without_future_conflict_and_positive_for_head_on_case():
    solver = EvasiveAcceleration({
        "horizon_seconds": 10.0,
        "timestep_seconds": 0.02,
        "coarse_directions": 72,
        "fine_directions": 51,
    })
    ego = RoadUserState(0.0, 0.0, 10.0, 0.0, 4.5, 1.8)
    same_speed = RoadUserState(20.0, 5.0, 10.0, 0.0, 4.5, 1.8)
    head_on = RoadUserState(20.0, 0.0, 8.0, np.pi, 4.7, 1.9)
    assert solver.compute(ego, same_speed) == 0.0
    # 官方单帧示例约为 4.828 m/s^2；时间离散近似应处于其附近。
    assert 4.4 < solver.compute(ego, head_on) < 5.2


def _simple_reachability_env(static_obstacles):
    route = np.stack((np.linspace(0.0, 40.0, 41), np.zeros(41)), axis=-1)
    return SimpleNamespace(
        scenario_dict={"route": route},
        ego_state=np.asarray([0.0, 0.0, 5.0, 0.0, 0.0, 4.5, 1.8, 1.0]),
        data_dict={"agent": [np.empty((0, 8), dtype=np.float64)]},
        agent_active=np.empty((0,), dtype=bool),
        pending_joint_trajectory=None,
        dt=0.1,
        get_static_obstacles=lambda: static_obstacles,
    )


def test_reachability_obstacle_pruning_never_increases_terminal_area():
    solver = RouteReachability({
        "horizon_seconds": 2.0,
        "timestep_seconds": 0.5,
        "lane_half_width": 3.0,
    })
    baseline_env = _simple_reachability_env([])
    obstacle = {
        "center": [7.0, 0.0],
        "yaw": 0.0,
        "length": 2.0,
        "width": 8.0,
    }
    baseline = solver.compute(baseline_env, False, [])
    dangerous = solver.compute(
        _simple_reachability_env([obstacle]),
        False,
        [obstacle],
    )
    assert baseline["area_m2"] > 0.0
    assert dangerous["area_m2"] <= baseline["area_m2"]


def test_reachability_reads_safe_sim_joint_trajectory_for_dynamic_pruning():
    solver = RouteReachability({
        "horizon_seconds": 2.0,
        "timestep_seconds": 0.5,
        "lane_half_width": 3.0,
    })
    env = _simple_reachability_env([])
    # 原始交通参与者远离路由，但 Safe-Sim 联合轨迹将其移入自车前方并横跨车道。
    env.data_dict = {
        "agent": [np.asarray([[30.0, 10.0, 0.0, 0.0, 0.0, 1.0, 8.0, 1.0]])]
    }
    env.agent_active = np.asarray([True])
    horizon = 20
    positions = np.zeros((1, horizon, 2), dtype=np.float32)
    positions[0, :, 0] = np.arange(1, horizon + 1, dtype=np.float32) * 0.5
    env.pending_joint_trajectory = JointTrajectory(
        source_step=0,
        agent_ids=np.asarray([0], dtype=np.int64),
        positions_global=positions,
        yaws_global=np.zeros((1, horizon, 1), dtype=np.float32),
        velocities_global=np.zeros((1, horizon, 2), dtype=np.float32),
        valid_mask=np.ones((1, horizon), dtype=bool),
    )
    baseline = solver.compute(env, False, [])
    dangerous = solver.compute(env, True, [])
    assert dangerous["area_m2"] < baseline["area_m2"]
