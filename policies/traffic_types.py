"""Scenario Dreamer 与 Safe-Sim 接合代码共用的强类型数据契约。"""

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


SCENARIO_STATE_DIM = 8
SCENARIO_TYPE_DIM = 5


def _require_shape(name, value, ndim, trailing_shape=None):
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(value).__name__}")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {value.shape}")
    if trailing_shape is not None and value.shape[-len(trailing_shape):] != trailing_shape:
        raise ValueError(f"{name} must end with shape {trailing_shape}, got {value.shape}")


@dataclass(frozen=True)
class ScenarioFrame:
    """全局坐标系下唯一可信的 Scenario Dreamer 状态快照。

    交通参与者编号在单个场景内始终对应 ``states_global`` 的稳定索引。历史轨迹按
    从旧到新的顺序排列；不足 ``H`` 帧时向右对齐。
    """

    scene_id: str
    step: int
    dt: float
    agent_ids: np.ndarray
    states_global: np.ndarray
    agent_types: np.ndarray
    active_mask: np.ndarray
    history_global: np.ndarray
    history_mask: np.ndarray
    lanes_global: np.ndarray
    route_global: np.ndarray
    ego_state_global: np.ndarray = None

    def __post_init__(self):
        _require_shape("agent_ids", self.agent_ids, 1)
        _require_shape("states_global", self.states_global, 2, (SCENARIO_STATE_DIM,))
        _require_shape("agent_types", self.agent_types, 2, (SCENARIO_TYPE_DIM,))
        _require_shape("active_mask", self.active_mask, 1)
        _require_shape("history_global", self.history_global, 3, (SCENARIO_STATE_DIM,))
        _require_shape("history_mask", self.history_mask, 2)
        _require_shape("lanes_global", self.lanes_global, 3, (2,))
        _require_shape("route_global", self.route_global, 2, (2,))
        if self.ego_state_global is not None:
            _require_shape("ego_state_global", self.ego_state_global, 1, (SCENARIO_STATE_DIM,))

        agent_count = self.states_global.shape[0]
        expected_first_dims = {
            "agent_ids": self.agent_ids.shape[0],
            "agent_types": self.agent_types.shape[0],
            "active_mask": self.active_mask.shape[0],
            "history_global": self.history_global.shape[0],
            "history_mask": self.history_mask.shape[0],
        }
        mismatches = {name: size for name, size in expected_first_dims.items() if size != agent_count}
        if mismatches:
            raise ValueError(f"agent dimension mismatch: states={agent_count}, others={mismatches}")
        if self.history_global.shape[:2] != self.history_mask.shape:
            raise ValueError(
                "history_global and history_mask must share [A, H], got "
                f"{self.history_global.shape} and {self.history_mask.shape}"
            )
        if len(np.unique(self.agent_ids)) != len(self.agent_ids):
            raise ValueError("agent_ids must be unique and stable within a scene")
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")


@dataclass(frozen=True)
class SafeSimBatch:
    """已解析的 Safe-Sim 推理批次，以及稳定编号和坐标变换元数据。"""

    data: Dict[str, Any]
    row_to_agent_id: np.ndarray
    agent_from_world: np.ndarray
    world_from_agent: np.ndarray
    raster_source: str = "scenario_dreamer_lane_raster"

    def __post_init__(self):
        _require_shape("row_to_agent_id", self.row_to_agent_id, 1)
        _require_shape("agent_from_world", self.agent_from_world, 3, (3, 3))
        _require_shape("world_from_agent", self.world_from_agent, 3, (3, 3))
        batch_size = self.row_to_agent_id.shape[0]
        if self.agent_from_world.shape[0] != batch_size or self.world_from_agent.shape[0] != batch_size:
            raise ValueError("Safe-Sim transform batch dimension must match row_to_agent_id")


@dataclass(frozen=True)
class JointTrajectory:
    """所有当前受控非自车交通参与者的联合未来轨迹。

    位置、速度和航向角均使用 Scenario Dreamer 全局坐标系。仿真器只执行索引 0
    对应的第一帧，并在下一帧重新规划。
    """

    source_step: int
    agent_ids: np.ndarray
    positions_global: np.ndarray
    yaws_global: np.ndarray
    velocities_global: np.ndarray
    valid_mask: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        _require_shape("agent_ids", self.agent_ids, 1)
        _require_shape("positions_global", self.positions_global, 3, (2,))
        _require_shape("yaws_global", self.yaws_global, 3, (1,))
        _require_shape("velocities_global", self.velocities_global, 3, (2,))
        _require_shape("valid_mask", self.valid_mask, 2)
        batch_size = self.agent_ids.shape[0]
        horizon = self.positions_global.shape[1]
        expected = (batch_size, horizon)
        if self.yaws_global.shape[:2] != expected:
            raise ValueError("yaws_global must share [A, T] with positions_global")
        if self.velocities_global.shape[:2] != expected:
            raise ValueError("velocities_global must share [A, T] with positions_global")
        if self.valid_mask.shape != expected:
            raise ValueError("valid_mask must share [A, T] with positions_global")
        if len(np.unique(self.agent_ids)) != len(self.agent_ids):
            raise ValueError("JointTrajectory agent_ids must be unique")

    @classmethod
    def empty(cls, source_step):
        return cls(
            source_step=source_step,
            agent_ids=np.empty((0,), dtype=np.int64),
            positions_global=np.empty((0, 0, 2), dtype=np.float32),
            yaws_global=np.empty((0, 0, 1), dtype=np.float32),
            velocities_global=np.empty((0, 0, 2), dtype=np.float32),
            valid_mask=np.empty((0, 0), dtype=bool),
        )

    def trajectory_for(self, agent_id):
        matches = np.flatnonzero(self.agent_ids == agent_id)
        if len(matches) == 0:
            return None
        row = int(matches[0])
        return np.concatenate(
            [self.positions_global[row], self.velocities_global[row], self.yaws_global[row]],
            axis=-1,
        )
