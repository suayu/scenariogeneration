"""SAFE-SIM 兼容的轨迹真实性分布审计。"""

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np


DEFAULT_BINS = {
    "velocity": np.linspace(0.0, 30.0, 21),
    "lon_accel": np.linspace(0.0, 10.0, 21),
    "lat_accel": np.linspace(0.0, 10.0, 21),
    "jerk": np.linspace(0.0, 20.0, 21),
}


def _setting(settings, name, default):
    if settings is None:
        return default
    if isinstance(settings, dict):
        return settings.get(name, default)
    return getattr(settings, name, default)


def _normalized_histogram(values, edges):
    """按 SAFE-SIM 固定区间统计并归一化；区间外样本不进入主直方图。"""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    counts = np.histogram(values, bins=edges)[0].astype(np.float64)
    if counts.sum() <= 0:
        return None
    return counts / counts.sum()


def _wasserstein_histogram(left, right, edges):
    """计算一维归一化直方图的 Wasserstein-1 距离。"""
    widths = np.diff(np.asarray(edges, dtype=np.float64))
    return float(np.sum(np.abs(np.cumsum(left) - np.cumsum(right)) * widths))


def _jensen_shannon(left, right):
    """返回自然对数定义的 Jensen--Shannon divergence。"""
    midpoint = 0.5 * (left + right)
    left_mask = left > 0
    right_mask = right > 0
    left_kl = np.sum(left[left_mask] * np.log(left[left_mask] / midpoint[left_mask]))
    right_kl = np.sum(right[right_mask] * np.log(right[right_mask] / midpoint[right_mask]))
    return float(0.5 * (left_kl + right_kl))


@dataclass
class _TrackState:
    step: int
    position: np.ndarray
    yaw: float
    velocity: np.ndarray | None = None
    acceleration_norm: float | None = None


@dataclass
class _DomainAccumulator:
    previous: dict = field(default_factory=dict)
    values: dict = field(default_factory=lambda: {
        "velocity": [], "lon_accel": [], "lat_accel": [], "jerk": [],
    })

    def observe(self, agent_id, step, state, dt):
        position = np.asarray(state[:2], dtype=np.float64)
        yaw = float(state[4])
        previous = self.previous.get(agent_id)
        current = _TrackState(step=step, position=position.copy(), yaw=yaw)
        if previous is not None and step == previous.step + 1:
            velocity = (position - previous.position) / dt
            current.velocity = velocity
            self.values["velocity"].append(float(np.linalg.norm(velocity)))
            if previous.velocity is not None:
                acceleration = (velocity - previous.velocity) / dt
                acceleration_norm = float(np.linalg.norm(acceleration))
                current.acceleration_norm = acceleration_norm
                # 与 SAFE-SIM 官方实现保持完全相同的画像统计口径。
                self.values["lon_accel"].append(acceleration_norm * np.cos(yaw))
                self.values["lat_accel"].append(acceleration_norm * np.sin(yaw))
                if previous.acceleration_norm is not None:
                    self.values["jerk"].append(
                        (acceleration_norm - previous.acceleration_norm) / dt
                    )
        self.previous[agent_id] = current


class TrajectoryRealismAudit:
    """比较闭环模拟背景轨迹与冻结真实数据直方图的分布距离。"""

    def __init__(self, settings, dt):
        self.enabled = bool(_setting(settings, "enabled", False))
        self.dt = float(dt)
        if self.dt <= 0 or not np.isfinite(self.dt):
            raise ValueError("真实性审计需要有效且为正的仿真 dt")
        self.minimum_samples = int(_setting(settings, "minimum_samples", 8))
        if self.minimum_samples < 1:
            raise ValueError("真实性审计 minimum_samples 必须至少为 1")
        self.bins = {}
        for name, default in DEFAULT_BINS.items():
            configured = _setting(settings, f"{name}_bins", default)
            edges = np.asarray(configured, dtype=np.float64)
            if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or not np.all(np.diff(edges) > 0):
                raise ValueError(f"真实性审计 {name} 分箱必须严格递增")
            self.bins[name] = edges
        self.reference_path = _setting(settings, "reference_histogram_path", None)
        self.reference_histograms = None
        self.reference_metadata = None
        if self.enabled and self.reference_path:
            payload = json.loads(Path(self.reference_path).read_text(encoding="utf-8"))
            stats = payload.get("stats", payload)
            ticks = payload.get("ticks")
            if ticks is not None:
                for name in DEFAULT_BINS:
                    reference_edges = np.asarray(ticks.get(name), dtype=np.float64)
                    if reference_edges.shape != self.bins[name].shape or not np.allclose(
                        reference_edges, self.bins[name], rtol=0.0, atol=1e-12
                    ):
                        raise ValueError(f"真实参考直方图 {name} 分箱与当前配置不一致")
            reference_dt = payload.get("dt_seconds")
            if reference_dt is not None and not np.isclose(
                float(reference_dt), self.dt, rtol=0.0, atol=1e-12
            ):
                raise ValueError("真实参考直方图时间步长与当前仿真 dt 不一致")
            self.reference_histograms = {}
            for name in DEFAULT_BINS:
                values = np.asarray(stats[name], dtype=np.float64)
                if values.shape != (len(self.bins[name]) - 1,) or not np.isfinite(values).all() or values.sum() <= 0:
                    raise ValueError(f"真实参考直方图 {name} 无效")
                self.reference_histograms[name] = values / values.sum()
            # 只记录不含原始轨迹和凭据的可复现性元数据。
            self.reference_metadata = {
                key: payload.get(key)
                for key in (
                    "schema_version", "dataset_label", "dt_seconds", "file_count",
                    "source_manifest_sha256", "sample_counts", "range_audit",
                )
                if key in payload
            }
        self._simulated = _DomainAccumulator()
        self._real = _DomainAccumulator()
        self._observed_steps = set()

    def begin_episode(self):
        """每次场景重放独立计量，防止失败重放污染最终分布。"""
        self._simulated = _DomainAccumulator()
        self._real = _DomainAccumulator()
        self._observed_steps = set()

    def observe(self, env):
        if not self.enabled:
            return
        step = int(env.current_step)
        if step in self._observed_steps:
            return
        self._observed_steps.add(step)
        simulated = np.asarray(env.data_dict["agent"][-1], dtype=np.float64)
        ground_truth = np.asarray(env.scenario_dict["agents"], dtype=np.float64)
        active = np.asarray(env.agent_active, dtype=bool)
        # Simulator 的最后一个轨迹槽位是自车；真实性只统计背景车辆。
        background_count = min(len(ground_truth), len(simulated), len(active))
        for agent_id in np.flatnonzero(active[:background_count] & np.isfinite(simulated[:background_count, :5]).all(axis=1)):
            self._simulated.observe(int(agent_id), step, simulated[agent_id], self.dt)
        # 某些回放数据自带完整真实未来，可用于测试；Scenario Dreamer 生成场景通常只有初态。
        if ground_truth.ndim != 3 or step >= ground_truth.shape[1]:
            return
        real_frame = ground_truth[:, step]
        valid = active[:background_count] & np.isfinite(real_frame[:background_count, :5]).all(axis=1)
        if real_frame.shape[1] > 7:
            valid &= real_frame[:background_count, 7] > 0
        for agent_id in np.flatnonzero(valid):
            self._real.observe(int(agent_id), step, real_frame[agent_id], self.dt)

    def finish_episode(self):
        if not self.enabled:
            return None
        distances = {}
        divergences = {}
        sample_counts = {}
        histograms = {"simulated_counts": {}, "reference_probability": {}}
        range_audit = {}
        missing = []
        for name, edges in self.bins.items():
            simulated_values = self._simulated.values[name]
            real_values = self._real.values[name]
            reference_hist = self.reference_histograms.get(name) if self.reference_histograms else None
            finite_simulated = np.asarray(simulated_values, dtype=np.float64)
            finite_simulated = finite_simulated[np.isfinite(finite_simulated)]
            simulated_counts = np.histogram(finite_simulated, bins=edges)[0].astype(np.int64)
            histograms["simulated_counts"][name] = simulated_counts.tolist()
            range_audit[name] = {
                "below_count": int(np.sum(finite_simulated < edges[0])),
                "above_count": int(np.sum(finite_simulated > edges[-1])),
                "in_range_count": int(simulated_counts.sum()),
            }
            sample_counts[name] = {
                "simulated": len(simulated_values),
                "real": len(real_values) if reference_hist is None else None,
            }
            if len(simulated_values) < self.minimum_samples or (
                reference_hist is None and len(real_values) < self.minimum_samples
            ):
                distances[name] = None
                divergences[name] = None
                missing.append(name)
                continue
            simulated_hist = _normalized_histogram(simulated_values, edges)
            real_hist = reference_hist if reference_hist is not None else _normalized_histogram(real_values, edges)
            histograms["reference_probability"][name] = real_hist.tolist()
            distances[name] = _wasserstein_histogram(simulated_hist, real_hist, edges)
            divergences[name] = _jensen_shannon(simulated_hist, real_hist)
        required = ("lon_accel", "lat_accel", "jerk")
        valid = all(distances.get(name) is not None for name in required)
        if self.reference_histograms is None and not any(self._real.values[name] for name in required):
            reason = "missing_real_reference"
        elif not valid:
            reason = "insufficient_realism_samples"
        else:
            reason = None
        return {
            "enabled": True,
            "valid": valid,
            "reason": reason,
            "reference_source": (
                str(self.reference_path) if self.reference_histograms is not None
                else ("paired_ground_truth" if any(self._real.values[name] for name in required) else None)
            ),
            "reference_metadata": self.reference_metadata,
            "observed_frame_count": len(self._observed_steps),
            "sample_counts": sample_counts,
            "histogram_edges": {name: edges.tolist() for name, edges in self.bins.items()},
            "histograms": histograms,
            "range_audit": range_audit,
            "wasserstein": distances,
            "jensen_shannon_diagnostic": divergences,
            "realism_deviation": (
                float(np.mean([distances[name] for name in required])) if valid else None
            ),
            "definition": "SAFE-SIM normalized-histogram Wasserstein mean over longitudinal acceleration, lateral acceleration, and jerk",
            "missing_metrics": missing,
        }
