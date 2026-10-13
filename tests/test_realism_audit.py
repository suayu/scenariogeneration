import json
from types import SimpleNamespace

import numpy as np

from policies.realism_audit import TrajectoryRealismAudit


class FakeEnv:
    def __init__(self, trajectories, ground_truth, dt=0.1):
        self.current_step = 0
        self.agent_active = np.ones(len(trajectories), dtype=bool)
        self.data_dict = {"agent": [trajectories[:, 0].copy()]}
        self.scenario_dict = {"agents": ground_truth.copy()}
        self.trajectories = trajectories
        self.dt = dt

    def seek(self, step):
        self.current_step = step
        self.data_dict["agent"] = [self.trajectories[:, step].copy()]


def trajectory(acceleration_scale=1.0, frames=40, dt=0.1):
    states = np.zeros((2, frames, 8), dtype=float)
    states[..., 5:7] = [4.5, 2.0]
    states[..., 7] = 1.0
    for agent_id in range(2):
        acceleration = acceleration_scale * (0.4 + 0.2 * agent_id)
        time = np.arange(frames) * dt
        states[agent_id, :, 0] = 3.0 * time + 0.5 * acceleration * time ** 2
        states[agent_id, :, 1] = agent_id * 3.5 + 0.1 * np.sin(time)
        states[agent_id, :, 4] = 0.05 * np.sin(time)
    return states


def settings(enabled=True, minimum_samples=8, reference_histogram_path=None):
    return SimpleNamespace(
        enabled=enabled,
        minimum_samples=minimum_samples,
        reference_histogram_path=reference_histogram_path,
    )


def run(audit, env):
    audit.begin_episode()
    for step in range(env.trajectories.shape[1]):
        env.seek(step)
        audit.observe(env)
    return audit.finish_episode()


def test_matching_real_and_simulated_trajectories_have_zero_deviation():
    real = trajectory()
    result = run(TrajectoryRealismAudit(settings(), 0.1), FakeEnv(real, real))
    assert result["valid"]
    assert result["realism_deviation"] == 0.0
    assert all(value == 0.0 for value in result["wasserstein"].values())
    assert sum(result["histograms"]["simulated_counts"]["velocity"]) > 0
    assert len(result["histogram_edges"]["velocity"]) == 21


def test_changed_motion_distribution_increases_realism_deviation():
    real = trajectory()
    simulated = trajectory(acceleration_scale=4.0)
    result = run(TrajectoryRealismAudit(settings(), 0.1), FakeEnv(simulated, real))
    assert result["valid"]
    assert result["realism_deviation"] > 0.0
    assert result["wasserstein"]["lon_accel"] > 0.0


def test_short_episode_reports_invalid_measurement_instead_of_zero():
    real = trajectory(frames=4)
    result = run(TrajectoryRealismAudit(settings(minimum_samples=8), 0.1), FakeEnv(real, real))
    assert not result["valid"]
    assert result["realism_deviation"] is None
    assert result["reason"] == "insufficient_realism_samples"


def test_disabled_audit_has_no_result():
    audit = TrajectoryRealismAudit(settings(enabled=False), 0.1)
    audit.begin_episode()
    assert audit.finish_episode() is None


def test_initial_state_only_scene_reports_missing_real_reference():
    simulated = trajectory()
    initial_only = simulated[:, :1].copy()
    result = run(TrajectoryRealismAudit(settings(), 0.1), FakeEnv(simulated, initial_only))
    assert result["sample_counts"]["velocity"]["simulated"] > 0
    assert result["reason"] == "missing_real_reference"
    assert result["realism_deviation"] is None


def test_frozen_reference_histogram_supports_generated_scenario_audit(tmp_path):
    simulated = trajectory()
    initial_only = simulated[:, :1].copy()
    reference = {}
    audit_for_reference = TrajectoryRealismAudit(settings(), 0.1)
    paired = run(audit_for_reference, FakeEnv(simulated, simulated))
    for name, edges in audit_for_reference.bins.items():
        values = audit_for_reference._real.values[name]
        reference[name] = np.histogram(values, bins=edges)[0].tolist()
    path = tmp_path / "real_histograms.json"
    path.write_text(json.dumps({"stats": reference}), encoding="utf-8")
    result = run(
        TrajectoryRealismAudit(settings(reference_histogram_path=str(path)), 0.1),
        FakeEnv(simulated, initial_only),
    )
    assert paired["valid"] and result["valid"]
    assert result["realism_deviation"] == 0.0
    assert result["reference_source"] == str(path)


def test_frozen_reference_rejects_mismatched_bins(tmp_path):
    audit = TrajectoryRealismAudit(settings(), 0.1)
    stats = {
        name: np.ones(len(edges) - 1).tolist()
        for name, edges in audit.bins.items()
    }
    ticks = {name: edges.tolist() for name, edges in audit.bins.items()}
    ticks["jerk"][1] += 0.25
    path = tmp_path / "wrong_bins.json"
    path.write_text(json.dumps({"stats": stats, "ticks": ticks}), encoding="utf-8")

    with np.testing.assert_raises_regex(ValueError, "分箱与当前配置不一致"):
        TrajectoryRealismAudit(settings(reference_histogram_path=str(path)), 0.1)


def test_frozen_reference_rejects_mismatched_dt(tmp_path):
    audit = TrajectoryRealismAudit(settings(), 0.1)
    stats = {
        name: np.ones(len(edges) - 1).tolist()
        for name, edges in audit.bins.items()
    }
    path = tmp_path / "wrong_dt.json"
    path.write_text(
        json.dumps({"stats": stats, "dt_seconds": 0.5}), encoding="utf-8"
    )

    with np.testing.assert_raises_regex(ValueError, "时间步长"):
        TrajectoryRealismAudit(settings(reference_histogram_path=str(path)), 0.1)
