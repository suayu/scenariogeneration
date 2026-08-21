"""统一静态障碍物接口及 Scenario Dreamer、CARLA 后端实现。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import count
from typing import Any, Dict, Iterable, List, Mapping, Optional

import math
import numpy as np

from policies.obstacles import ObstacleCatalog, StaticObstacle


class ObstacleBackend(ABC):
    """外部仿真器障碍物 API 的统一适配协议。"""

    @abstractmethod
    def create(self, obstacle: StaticObstacle) -> Any:
        """在后端创建障碍物，并返回后端句柄。"""

    @abstractmethod
    def delete(self, backend_handle: Any) -> None:
        """从后端删除障碍物。"""

    @abstractmethod
    def update(self, backend_handle: Any, obstacle: StaticObstacle) -> Any:
        """在后端更新障碍物，并返回可能变更的后端句柄。"""

    def colliding_ids(self, ego_state: np.ndarray, obstacles: Iterable[StaticObstacle]) -> List[str]:
        """返回与自车相交的障碍物编号；不支持时返回空列表。"""
        return []


class ScenarioDreamerObstacleBackend(ObstacleBackend):
    """Scenario Dreamer 静态障碍物后端，以抽象碰撞体参与环境判定。"""

    def __init__(self, simulator: Any):
        self.simulator = simulator

    def create(self, obstacle: StaticObstacle) -> str:
        # Scenario Dreamer 没有独立 actor；字典中的领域对象即为该后端的持久状态。
        self.simulator.static_obstacle_elements[obstacle.obstacle_id] = obstacle
        return obstacle.obstacle_id

    def delete(self, backend_handle: str) -> None:
        self.simulator.static_obstacle_elements.pop(str(backend_handle), None)

    def update(self, backend_handle: str, obstacle: StaticObstacle) -> str:
        self.simulator.static_obstacle_elements.pop(str(backend_handle), None)
        self.simulator.static_obstacle_elements[obstacle.obstacle_id] = obstacle
        return obstacle.obstacle_id

    def colliding_ids(self, ego_state, obstacles):
        return [
            obstacle.obstacle_id
            for obstacle in obstacles
            if obstacle.collision_enabled and _oriented_boxes_overlap(ego_state, obstacle)
        ]


class CarlaObstacleBackend(ObstacleBackend):
    """CARLA 静态障碍物后端；仅在传入 CARLA world 时加载 CARLA 模块。"""

    _BLUEPRINTS = {
        # 领域 object_kind 到 CARLA 蓝图的映射集中在后端，规划器无需了解蓝图名称。
        "barrier": "static.prop.streetbarrier",
        "traffic_cone": "static.prop.trafficcone01",
        "disabled_vehicle": "vehicle.audi.tt",
        "debris": "static.prop.cardboardbox01",
    }

    def __init__(self, world: Any, blueprint_library: Any = None):
        self.world = world
        self.blueprint_library = blueprint_library or world.get_blueprint_library()

    def _blueprint_for(self, obstacle: StaticObstacle):
        blueprint_id = self._BLUEPRINTS[obstacle.object_kind]
        return self.blueprint_library.find(blueprint_id)

    @staticmethod
    def _transform_for(obstacle: StaticObstacle):
        try:
            import carla
        except ImportError as error:
            raise RuntimeError("CARLA 后端需要已安装 carla Python API") from error
        # 领域层 yaw 使用弧度，CARLA Rotation 使用角度；z 取物体高度中心以落在地面上。
        return carla.Transform(
            carla.Location(x=obstacle.center_global[0], y=obstacle.center_global[1], z=obstacle.height / 2.0),
            carla.Rotation(yaw=math.degrees(obstacle.yaw)),
        )

    def create(self, obstacle: StaticObstacle) -> Any:
        actor = self.world.try_spawn_actor(self._blueprint_for(obstacle), self._transform_for(obstacle))
        if actor is None:
            raise RuntimeError(f"CARLA 未能创建障碍物 {obstacle.obstacle_id}")
        return actor

    def delete(self, backend_handle: Any) -> None:
        if backend_handle is not None and getattr(backend_handle, "is_alive", True):
            backend_handle.destroy()

    def update(self, backend_handle: Any, obstacle: StaticObstacle) -> Any:
        backend_handle.set_transform(self._transform_for(obstacle))
        return backend_handle


class ObstacleWrapper:
    """统一管理障碍物的存储、创建、删除、修改、查询和碰撞判定。"""

    def __init__(self, backend: ObstacleBackend, catalog: Optional[ObstacleCatalog] = None):
        self.backend = backend
        self.catalog = catalog or ObstacleCatalog()
        # 领域对象与后端句柄分开保存，查询接口不会泄露 CARLA actor 等后端对象。
        self._obstacles: Dict[str, StaticObstacle] = {}
        self._backend_handles: Dict[str, Any] = {}
        self._id_counter = count()

    def _new_id(self, primitive_type: str) -> str:
        return f"obstacle-{primitive_type}-{next(self._id_counter):04d}"

    def create(self, obstacle: StaticObstacle) -> StaticObstacle:
        if obstacle.obstacle_id in self._obstacles:
            raise ValueError(f"障碍物编号已存在：{obstacle.obstacle_id}")
        self._backend_handles[obstacle.obstacle_id] = self.backend.create(obstacle)
        self._obstacles[obstacle.obstacle_id] = obstacle
        return obstacle

    def create_plan(self, placements: Iterable[Mapping[str, Any]], max_groups: int = 2) -> List[StaticObstacle]:
        placements = list(placements)
        if len(placements) > max_groups:
            raise ValueError(f"一次最多创建 {max_groups} 组障碍物")
        created: List[StaticObstacle] = []
        try:
            for placement in placements:
                for obstacle in self.catalog.materialize(placement, self._new_id):
                    created.append(self.create(obstacle))
        except Exception:
            # 组内任一单体创建失败时回滚已创建单体，避免留下半个障碍物组合。
            for obstacle in reversed(created):
                self.delete(obstacle.obstacle_id)
            raise
        return created

    def delete(self, obstacle_id: str) -> bool:
        obstacle = self._obstacles.pop(obstacle_id, None)
        if obstacle is None:
            return False
        handle = self._backend_handles.pop(obstacle_id, None)
        self.backend.delete(handle)
        return True

    def update(self, obstacle_id: str, **changes: Any) -> StaticObstacle:
        obstacle = self.query(obstacle_id)
        if obstacle is None:
            raise KeyError(f"未找到障碍物：{obstacle_id}")
        updated = obstacle.with_updates(**changes)
        handle = self.backend.update(self._backend_handles[obstacle_id], updated)
        self._obstacles[obstacle_id] = updated
        self._backend_handles[obstacle_id] = handle
        return updated

    def query(self, obstacle_id: str) -> Optional[StaticObstacle]:
        return self._obstacles.get(obstacle_id)

    def list(self) -> List[StaticObstacle]:
        return list(self._obstacles.values())

    def public_state(self) -> List[Dict[str, Any]]:
        return [obstacle.to_public_dict() for obstacle in self.list()]

    def colliding_ids(self, ego_state: np.ndarray) -> List[str]:
        return self.backend.colliding_ids(ego_state, self.list())

    def clear(self) -> None:
        for obstacle_id in list(self._obstacles):
            self.delete(obstacle_id)


def _oriented_box_corners(state: np.ndarray, length: float, width: float) -> np.ndarray:
    center = np.asarray(state[:2], dtype=float)
    yaw = float(state[4])
    forward = np.array([math.cos(yaw), math.sin(yaw)])
    lateral = np.array([-math.sin(yaw), math.cos(yaw)])
    return np.array([
        center + forward * length / 2.0 + lateral * width / 2.0,
        center + forward * length / 2.0 - lateral * width / 2.0,
        center - forward * length / 2.0 - lateral * width / 2.0,
        center - forward * length / 2.0 + lateral * width / 2.0,
    ])


def _oriented_boxes_overlap(ego_state: np.ndarray, obstacle: StaticObstacle) -> bool:
    ego_length = float(ego_state[5])
    ego_width = float(ego_state[6])
    ego_corners = _oriented_box_corners(ego_state, ego_length, ego_width)
    obstacle_state = np.array([*obstacle.center_global, 0.0, 0.0, obstacle.yaw], dtype=float)
    obstacle_corners = _oriented_box_corners(obstacle_state, obstacle.length, obstacle.width)
    # 采用分离轴定理检测两个带朝向矩形；因此不依赖任一外部碰撞引擎。
    axes = []
    for corners in (ego_corners, obstacle_corners):
        for start, end in ((corners[0], corners[1]), (corners[1], corners[2])):
            edge = end - start
            norm = np.linalg.norm(edge)
            if norm > 1e-8:
                axes.append(np.array([-edge[1], edge[0]]) / norm)
    for axis in axes:
        ego_projection = ego_corners @ axis
        obstacle_projection = obstacle_corners @ axis
        if ego_projection.max() < obstacle_projection.min() or obstacle_projection.max() < ego_projection.min():
            return False
    return True
