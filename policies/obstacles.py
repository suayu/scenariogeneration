"""与具体仿真器无关的静态障碍物领域模型。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import math
import numpy as np


@dataclass(frozen=True)
class StaticObstacle:
    """可由任意仿真后端创建的单个静态碰撞体。"""

    # 此处始终保存全局坐标；局部坐标只存在于 LLM 输入输出阶段。
    obstacle_id: str
    primitive_type: str
    object_kind: str
    center_global: Tuple[float, float]
    yaw: float
    length: float
    width: float
    height: float = 1.0
    collision_enabled: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.obstacle_id:
            raise ValueError("障碍物编号不能为空")
        if self.length <= 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("障碍物长、宽、高必须为正数")
        if not np.isfinite([*self.center_global, self.yaw, self.length, self.width, self.height]).all():
            raise ValueError("障碍物参数必须为有限数值")

    def with_updates(self, **changes: Any) -> "StaticObstacle":
        """返回修改后的不可变障碍物实例。"""
        return replace(self, **changes)

    def to_public_dict(self) -> Dict[str, Any]:
        """导出不含后端句柄的抽象障碍物描述。"""
        return {
            "id": self.obstacle_id,
            "type": self.primitive_type,
            "object_kind": self.object_kind,
            "center": [float(self.center_global[0]), float(self.center_global[1])],
            "yaw": float(self.yaw),
            "length": float(self.length),
            "width": float(self.width),
            "height": float(self.height),
        }


@dataclass(frozen=True)
class ObstaclePlacement:
    """大模型可表达的抽象模板摆放请求。"""

    # 该请求不包含任何后端句柄，center 与 yaw 由规划器在进入包装器前转换为全局坐标。
    type: str
    center: Tuple[float, float]
    yaw: float
    count: int = 1
    spacing: float = 1.5

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ObstaclePlacement":
        allowed = {"type", "center", "yaw", "count", "spacing"}
        unknown = set(payload).difference(allowed)
        if unknown:
            raise ValueError(f"障碍物请求包含未知字段：{sorted(unknown)}")
        center = payload.get("center")
        if not isinstance(center, Sequence) or len(center) != 2:
            raise ValueError("障碍物 center 必须是两个坐标组成的列表")
        placement = cls(
            type=str(payload.get("type", "")),
            center=(float(center[0]), float(center[1])),
            yaw=float(payload.get("yaw", 0.0)),
            count=int(payload.get("count", 1)),
            spacing=float(payload.get("spacing", 1.5)),
        )
        if placement.count < 1 or placement.count > 12:
            raise ValueError("障碍物 count 必须在 1 到 12 之间")
        if not 0.3 <= placement.spacing <= 8.0:
            raise ValueError("障碍物 spacing 必须在 0.3 到 8 米之间")
        if not np.isfinite([*placement.center, placement.yaw, placement.spacing]).all():
            raise ValueError("障碍物请求参数必须为有限数值")
        return placement


class ObstacleTemplate(ABC):
    """LLM 可组合使用的障碍物模板基类。"""

    type_name: str
    description: str

    @abstractmethod
    def materialize(
        self,
        placement: ObstaclePlacement,
        new_id: Callable[[str], str],
    ) -> List[StaticObstacle]:
        """将一个抽象摆放请求展开为具体静态碰撞体。"""

    @staticmethod
    def _offset(center: Tuple[float, float], yaw: float, distance: float, lateral: bool = False):
        # yaw 指向模板的纵向；横向偏移用于将多个单体横跨车道排列。
        direction = np.array([-math.sin(yaw), math.cos(yaw)]) if lateral else np.array([math.cos(yaw), math.sin(yaw)])
        return tuple((np.asarray(center, dtype=float) + direction * distance).tolist())

    @staticmethod
    def _symmetric_offsets(count: int, spacing: float) -> Iterable[float]:
        return ((index - (count - 1) / 2.0) * spacing for index in range(count))


class TransverseBarrierTemplate(ObstacleTemplate):
    """横向条形护栏组合，用于阻断一条车道。"""

    type_name = "transverse_barrier"
    description = "横向条形护栏组合，沿横向排列以阻断一条车道"

    def materialize(self, placement, new_id):
        return [
            StaticObstacle(
                obstacle_id=new_id(self.type_name),
                primitive_type=self.type_name,
                object_kind="barrier",
                center_global=self._offset(placement.center, placement.yaw, offset, lateral=True),
                yaw=placement.yaw + math.pi / 2.0,
                length=1.2,
                width=0.45,
                height=0.9,
            )
            for offset in self._symmetric_offsets(placement.count, placement.spacing)
        ]


class LongitudinalBarrierTemplate(ObstacleTemplate):
    """纵向护栏组合，用于分隔两条相邻车道。"""

    type_name = "longitudinal_barrier"
    description = "纵向护栏组合，沿道路方向排列以分隔相邻车道"

    def materialize(self, placement, new_id):
        return [
            StaticObstacle(
                obstacle_id=new_id(self.type_name),
                primitive_type=self.type_name,
                object_kind="barrier",
                center_global=self._offset(placement.center, placement.yaw, offset),
                yaw=placement.yaw,
                length=1.5,
                width=0.4,
                height=0.9,
            )
            for offset in self._symmetric_offsets(placement.count, placement.spacing)
        ]


class ConeLineTemplate(ObstacleTemplate):
    """交通锥线，用于收窄车道或提示临时施工。"""

    type_name = "cone_line"
    description = "交通锥线，用于收窄车道或引导绕行"

    def materialize(self, placement, new_id):
        return [
            StaticObstacle(
                obstacle_id=new_id(self.type_name),
                primitive_type=self.type_name,
                object_kind="traffic_cone",
                center_global=self._offset(placement.center, placement.yaw, offset),
                yaw=placement.yaw,
                length=0.45,
                width=0.45,
                height=0.7,
            )
            for offset in self._symmetric_offsets(placement.count, placement.spacing)
        ]


class DisabledVehicleTemplate(ObstacleTemplate):
    """故障车辆，用于形成可感知且可绕行的静态阻挡。"""

    type_name = "disabled_vehicle"
    description = "故障车辆，可放置在车道中形成可绕行障碍"

    def materialize(self, placement, new_id):
        return [
            StaticObstacle(
                obstacle_id=new_id(self.type_name),
                primitive_type=self.type_name,
                object_kind="disabled_vehicle",
                center_global=placement.center,
                yaw=placement.yaw,
                length=4.8,
                width=2.0,
                height=1.6,
            )
        ]


class DebrisClusterTemplate(ObstacleTemplate):
    """小型散落物组合，用于降低可通行空间。"""

    type_name = "debris_cluster"
    description = "小型散落物组合，用于局部收窄可通行空间"

    def materialize(self, placement, new_id):
        obstacles = []
        for index, offset in enumerate(self._symmetric_offsets(placement.count, placement.spacing)):
            lateral = 0.35 * placement.spacing * (-1 if index % 2 else 1)
            center = self._offset(self._offset(placement.center, placement.yaw, offset), placement.yaw, lateral, lateral=True)
            obstacles.append(
                StaticObstacle(
                    obstacle_id=new_id(self.type_name),
                    primitive_type=self.type_name,
                    object_kind="debris",
                    center_global=center,
                    yaw=placement.yaw + (0.35 if index % 2 else -0.35),
                    length=0.7,
                    width=0.7,
                    height=0.45,
                )
            )
        return obstacles


class ConstructionBlockTemplate(ObstacleTemplate):
    """施工围挡组合，用于封闭局部道路空间。"""

    type_name = "construction_block"
    description = "施工围挡组合，由护栏与交通锥构成局部封闭区域"

    def materialize(self, placement, new_id):
        # 施工围挡复用基础模板，确保 LLM 的组合语义与实际碰撞体保持一致。
        barrier_request = ObstaclePlacement(
            type="transverse_barrier",
            center=placement.center,
            yaw=placement.yaw,
            count=max(2, placement.count),
            spacing=placement.spacing,
        )
        obstacles = TransverseBarrierTemplate().materialize(barrier_request, new_id)
        cones_center = self._offset(placement.center, placement.yaw, 1.5)
        cone_request = ObstaclePlacement(
            type="cone_line",
            center=cones_center,
            yaw=placement.yaw,
            count=max(2, placement.count),
            spacing=placement.spacing,
        )
        return obstacles + ConeLineTemplate().materialize(cone_request, new_id)


class ObstacleCatalog:
    """障碍物模板注册表，负责限制大模型可使用的类型集合。"""

    def __init__(self, templates: Iterable[ObstacleTemplate] | None = None):
        templates = templates or (
            TransverseBarrierTemplate(),
            LongitudinalBarrierTemplate(),
            ConeLineTemplate(),
            DisabledVehicleTemplate(),
            DebrisClusterTemplate(),
            ConstructionBlockTemplate(),
        )
        self._templates = {template.type_name: template for template in templates}

    @property
    def type_names(self) -> Tuple[str, ...]:
        return tuple(self._templates)

    def describe(self) -> List[Dict[str, str]]:
        return [
            {"type": template.type_name, "description": template.description}
            for template in self._templates.values()
        ]

    def materialize(
        self,
        placement_payload: Mapping[str, Any],
        new_id: Callable[[str], str],
    ) -> List[StaticObstacle]:
        # 先将 LLM 字典收敛为领域对象，再选择模板；此层不依赖任何仿真器。
        placement = ObstaclePlacement.from_dict(placement_payload)
        template = self._templates.get(placement.type)
        if template is None:
            raise ValueError(f"不支持的障碍物模板：{placement.type}")
        return template.materialize(placement, new_id)
