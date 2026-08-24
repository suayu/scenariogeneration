"""对抗性场景的二维风险与可达性评估。

EA 遵循论文定义：在二维相对运动空间中，搜索使未来占用区域不再相交的
最小常值相对加速度。可达性部分借鉴 CommonRoad-Reach 的分解动力学和图传播思路，
但直接使用 Scenario Dreamer 路由和障碍物格式，属于轻量兼容实现，不是完整的
CommonRoad 场景转换器。
"""

import math
from dataclasses import dataclass
from itertools import product

import numpy as np


def _config_value(config, name, default):
    """同时兼容 OmegaConf、字典和缺省配置。"""
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _wrap_angle(angle):
    """将角度稳定映射到 [-pi, pi]。"""
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class RoadUserState:
    """EA 所需的单个交通参与者瞬时状态。"""

    x: float
    y: float
    speed: float
    heading: float
    length: float
    width: float
    yaw_rate: float = 0.0


class EvasiveAcceleration:
    """基于 OBB 和 CV/CTRV 短时外推的 EA 数值求解器。"""

    def __init__(self, config=None):
        self.horizon = float(_config_value(config, "horizon_seconds", 7.0))
        self.dt = float(_config_value(config, "timestep_seconds", 0.02))
        self.a_max = float(_config_value(config, "max_acceleration", 100.0))
        self.coarse_directions = int(_config_value(config, "coarse_directions", 72))
        self.fine_directions = int(_config_value(config, "fine_directions", 51))
        self.fine_half_window = math.radians(
            float(_config_value(config, "fine_half_window_degrees", 5.0))
        )
        self.tolerance = float(_config_value(config, "tolerance", 1e-3))
        if self.horizon <= 0 or self.dt <= 0 or self.a_max <= 0:
            raise ValueError("EA 时域、时间步长和加速度上限必须为正")
        if self.coarse_directions < 4 or self.fine_directions < 3:
            raise ValueError("EA 方向离散数过小")

    @staticmethod
    def _predict(user, times, use_ctrv):
        """生成 CV 或 CTRV 的中心与朝向序列。"""
        if use_ctrv and abs(user.yaw_rate) > 1e-6:
            heading = user.heading + user.yaw_rate * times
            radius = user.speed / user.yaw_rate
            x = user.x + radius * (np.sin(heading) - math.sin(user.heading))
            y = user.y - radius * (np.cos(heading) - math.cos(user.heading))
            return x, y, heading
        heading = np.full_like(times, user.heading, dtype=np.float64)
        x = user.x + user.speed * math.cos(user.heading) * times
        y = user.y + user.speed * math.sin(user.heading) * times
        return x, y, heading

    @staticmethod
    def _support_data(user_a, user_b, prediction_a, prediction_b):
        """预计算 OBB 分离轴投影和两车相对位置。"""
        xa, ya, ha = prediction_a
        xb, yb, hb = prediction_b
        relative = np.stack((xb - xa, yb - ya), axis=-1)
        axis_a_long = np.stack((np.cos(ha), np.sin(ha)), axis=-1)
        axis_a_lat = np.stack((-np.sin(ha), np.cos(ha)), axis=-1)
        axis_b_long = np.stack((np.cos(hb), np.sin(hb)), axis=-1)
        axis_b_lat = np.stack((-np.sin(hb), np.cos(hb)), axis=-1)
        axes = np.stack((axis_a_long, axis_a_lat, axis_b_long, axis_b_lat), axis=1)

        bases = (
            (axis_a_long, user_a.length * 0.5),
            (axis_a_lat, user_a.width * 0.5),
            (axis_b_long, user_b.length * 0.5),
            (axis_b_lat, user_b.width * 0.5),
        )
        radii = np.zeros((len(relative), 4), dtype=np.float64)
        for axis_index in range(4):
            normal = axes[:, axis_index]
            for basis, extent in bases:
                radii[:, axis_index] += extent * np.abs(np.sum(basis * normal, axis=-1))
        projections = np.sum(relative[:, None, :] * axes, axis=-1)
        return relative, axes, radii, projections

    @staticmethod
    def _base_collides(projections, radii):
        """判断无额外规避加速度时是否存在未来碰撞。"""
        return bool(np.any(np.all(np.abs(projections) <= radii, axis=1)))

    def _direction_minimum(self, axes, radii, projections, half_t_squared, direction):
        """合并固定方向上的碰撞加速度区间，找到包含原点分量的右边界。"""
        intervals = []
        direction_projection = np.sum(axes * direction[None, None, :], axis=-1)
        for time_index in range(len(half_t_squared)):
            lower = 0.0
            upper = self.a_max
            collision_possible = True
            for axis_index in range(4):
                p_value = projections[time_index, axis_index]
                q_value = half_t_squared[time_index] * direction_projection[time_index, axis_index]
                radius = radii[time_index, axis_index]
                if abs(q_value) <= 1e-15:
                    if abs(p_value) > radius:
                        collision_possible = False
                        break
                    continue
                root_a = (p_value - radius) / q_value
                root_b = (p_value + radius) / q_value
                lower = max(lower, min(root_a, root_b))
                upper = min(upper, max(root_a, root_b))
                if lower > upper:
                    collision_possible = False
                    break
            if collision_possible and upper >= 0.0 and lower <= self.a_max:
                intervals.append((max(0.0, lower), min(self.a_max, upper)))

        if not intervals:
            return 0.0
        intervals.sort(key=lambda item: item[0])
        reach = 0.0
        covered_origin = False
        for lower, upper in intervals:
            if not covered_origin:
                if lower <= self.tolerance:
                    covered_origin = True
                    reach = max(reach, upper)
                else:
                    return 0.0
            elif lower <= reach + self.tolerance:
                reach = max(reach, upper)
            else:
                break
        if not covered_origin:
            return 0.0
        if reach >= self.a_max - self.tolerance:
            return float("nan")
        return float(np.nextafter(reach, np.inf))

    def _mode_value(self, user_a, user_b, use_ctrv_a, use_ctrv_b):
        """计算单个 CV/CTRV 运动组合下的 EA。"""
        times = np.arange(0.0, self.horizon + self.dt * 0.5, self.dt, dtype=np.float64)
        prediction_a = self._predict(user_a, times, use_ctrv_a)
        prediction_b = self._predict(user_b, times, use_ctrv_b)
        _, axes, radii, projections = self._support_data(
            user_a, user_b, prediction_a, prediction_b
        )
        if not self._base_collides(projections, radii):
            return 0.0

        half_t_squared = 0.5 * np.square(times)
        coarse_angles = (
            np.arange(self.coarse_directions, dtype=np.float64) + 0.5
        ) * (2.0 * math.pi / self.coarse_directions)
        coarse_values = []
        for angle in coarse_angles:
            direction = np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)
            coarse_values.append(
                self._direction_minimum(axes, radii, projections, half_t_squared, direction)
            )
        finite_indices = [index for index, value in enumerate(coarse_values) if np.isfinite(value)]
        if not finite_indices:
            return float("nan")
        best_index = min(finite_indices, key=lambda index: coarse_values[index])
        best_value = coarse_values[best_index]
        fine_angles = np.linspace(
            coarse_angles[best_index] - self.fine_half_window,
            coarse_angles[best_index] + self.fine_half_window,
            self.fine_directions,
        )
        for angle in fine_angles:
            direction = np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)
            value = self._direction_minimum(
                axes, radii, projections, half_t_squared, direction
            )
            if np.isfinite(value):
                best_value = min(best_value, value)
        return float(best_value)

    def compute(self, user_a, user_b):
        """返回四种运动组合的算术平均 EA，单位为 m/s^2。"""
        values = [
            self._mode_value(user_a, user_b, False, False),
            self._mode_value(user_a, user_b, False, True),
            self._mode_value(user_a, user_b, True, False),
            self._mode_value(user_a, user_b, True, True),
        ]
        if not all(np.isfinite(value) for value in values):
            return float("nan")
        return float(np.mean(values))


class RouteReachability:
    """在路由 Frenet 坐标中进行有限时域可达图传播。"""

    def __init__(self, config=None):
        self.horizon = float(_config_value(config, "horizon_seconds", 3.0))
        self.dt = float(_config_value(config, "timestep_seconds", 0.5))
        self.s_resolution = float(_config_value(config, "longitudinal_resolution", 1.0))
        self.d_resolution = float(_config_value(config, "lateral_resolution", 0.5))
        self.speed_resolution = float(_config_value(config, "speed_resolution", 1.0))
        self.lateral_speed_resolution = float(
            _config_value(config, "lateral_speed_resolution", 0.5)
        )
        self.lane_half_width = float(_config_value(config, "lane_half_width", 4.0))
        self.max_speed = float(_config_value(config, "max_speed", 25.0))
        self.max_lateral_speed = float(_config_value(config, "max_lateral_speed", 3.0))
        self.longitudinal_accelerations = tuple(
            float(value)
            for value in _config_value(
                config, "longitudinal_accelerations", [-4.0, -2.0, 0.0, 2.0]
            )
        )
        self.lateral_accelerations = tuple(
            float(value)
            for value in _config_value(config, "lateral_accelerations", [-2.5, 0.0, 2.5])
        )
        if min(self.horizon, self.dt, self.s_resolution, self.d_resolution) <= 0:
            raise ValueError("可达性时域、时间步长和网格分辨率必须为正")

    @staticmethod
    def _route_geometry(route):
        """计算路由弧长、单位切线和法线。"""
        route = np.asarray(route, dtype=np.float64)
        if route.ndim != 2 or route.shape[0] < 2 or route.shape[1] < 2:
            raise ValueError("路由至少需要两个二维点")
        route = route[:, :2]
        segment = np.diff(route, axis=0)
        length = np.linalg.norm(segment, axis=1)
        keep = np.concatenate(([True], length > 1e-6))
        route = route[keep]
        if len(route) < 2:
            raise ValueError("路由点全部重合")
        segment = np.diff(route, axis=0)
        length = np.linalg.norm(segment, axis=1)
        arc = np.concatenate(([0.0], np.cumsum(length)))
        tangent = segment / length[:, None]
        return route, arc, tangent

    @staticmethod
    def _sample_route(route, arc, tangent, s_value):
        """按弧长插值中心线点及当前方向。"""
        index = int(np.clip(np.searchsorted(arc, s_value, side="right") - 1, 0, len(tangent) - 1))
        ratio = (s_value - arc[index]) / max(arc[index + 1] - arc[index], 1e-6)
        center = route[index] + ratio * (route[index + 1] - route[index])
        return center, tangent[index]

    @staticmethod
    def _obb_overlap(center_a, yaw_a, length_a, width_a, center_b, yaw_b, length_b, width_b):
        """用分离轴定理检查两个二维 OBB。"""
        axes_a = np.asarray(
            [[math.cos(yaw_a), math.sin(yaw_a)], [-math.sin(yaw_a), math.cos(yaw_a)]],
            dtype=np.float64,
        )
        axes_b = np.asarray(
            [[math.cos(yaw_b), math.sin(yaw_b)], [-math.sin(yaw_b), math.cos(yaw_b)]],
            dtype=np.float64,
        )
        delta = np.asarray(center_b, dtype=np.float64) - np.asarray(center_a, dtype=np.float64)
        for normal in np.concatenate((axes_a, axes_b), axis=0):
            radius_a = 0.5 * length_a * abs(np.dot(axes_a[0], normal)) + 0.5 * width_a * abs(np.dot(axes_a[1], normal))
            radius_b = 0.5 * length_b * abs(np.dot(axes_b[0], normal)) + 0.5 * width_b * abs(np.dot(axes_b[1], normal))
            if abs(np.dot(delta, normal)) > radius_a + radius_b:
                return False
        return True

    def _dynamic_obstacles(self, env, time_seconds, use_joint_trajectory):
        """按相对时间取得动态障碍物：原始场景用 CV，危险场景优先用扩散联合轨迹。"""
        agents = np.asarray(env.data_dict["agent"][-1], dtype=np.float64)
        active = np.asarray(env.agent_active, dtype=bool)
        joint = env.pending_joint_trajectory if use_joint_trajectory else None
        joint_rows = {}
        joint_index = None
        if joint is not None and joint.positions_global.shape[1] > 0:
            joint_rows = {int(agent_id): row for row, agent_id in enumerate(joint.agent_ids)}
            joint_index = int(
                np.clip(round(time_seconds / float(env.dt)) - 1, 0, joint.positions_global.shape[1] - 1)
            )
        result = []
        for agent_id, state in enumerate(agents):
            if agent_id >= len(active) or not active[agent_id]:
                continue
            center = state[:2] + state[2:4] * time_seconds
            yaw = float(state[4])
            if agent_id in joint_rows:
                row = joint_rows[agent_id]
                if bool(joint.valid_mask[row, joint_index]):
                    center = np.asarray(joint.positions_global[row, joint_index], dtype=np.float64)
                    yaw = float(joint.yaws_global[row, joint_index, 0])
            result.append((center, yaw, float(state[5]), float(state[6])))
        return result

    def compute(self, env, use_joint_trajectory, static_obstacles=None):
        """计算终端可达位置面积和是否存在至少一条无碰路径。"""
        route, arc, tangent = self._route_geometry(env.scenario_dict["route"])
        ego = np.asarray(env.ego_state, dtype=np.float64)
        nearest = int(np.argmin(np.linalg.norm(route - ego[:2], axis=1)))
        s_initial = float(arc[min(nearest, len(arc) - 1)])
        tangent_initial = tangent[min(nearest, len(tangent) - 1)]
        normal_initial = np.asarray([-tangent_initial[1], tangent_initial[0]])
        d_initial = float(np.dot(ego[:2] - route[nearest], normal_initial))
        v_s_initial = float(np.dot(ego[2:4], tangent_initial))
        v_d_initial = float(np.dot(ego[2:4], normal_initial))
        ego_length = float(ego[5])
        ego_width = float(ego[6])
        static_obstacles = list(static_obstacles or [])

        def quantize(state):
            s_value, d_value, v_s, v_d = state
            return (
                round(s_value / self.s_resolution),
                round(d_value / self.d_resolution),
                round(v_s / self.speed_resolution),
                round(v_d / self.lateral_speed_resolution),
            )

        frontier = {quantize((s_initial, d_initial, v_s_initial, v_d_initial)): (
            s_initial, d_initial, v_s_initial, v_d_initial
        )}
        controls = tuple(product(self.longitudinal_accelerations, self.lateral_accelerations))
        num_steps = int(round(self.horizon / self.dt))
        for step_index in range(1, num_steps + 1):
            next_frontier = {}
            time_seconds = step_index * self.dt
            dynamic_obstacles = self._dynamic_obstacles(env, time_seconds, use_joint_trajectory)
            for state in frontier.values():
                s_value, d_value, v_s, v_d = state
                for acceleration_s, acceleration_d in controls:
                    next_v_s = float(np.clip(v_s + acceleration_s * self.dt, 0.0, self.max_speed))
                    next_v_d = float(np.clip(
                        v_d + acceleration_d * self.dt,
                        -self.max_lateral_speed,
                        self.max_lateral_speed,
                    ))
                    next_s = s_value + v_s * self.dt + 0.5 * acceleration_s * self.dt ** 2
                    next_d = d_value + v_d * self.dt + 0.5 * acceleration_d * self.dt ** 2
                    if next_s < 0.0 or next_s > arc[-1] or abs(next_d) > self.lane_half_width:
                        continue
                    centerline, direction = self._sample_route(route, arc, tangent, next_s)
                    normal = np.asarray([-direction[1], direction[0]])
                    center = centerline + normal * next_d
                    yaw = math.atan2(direction[1], direction[0])
                    blocked = any(
                        self._obb_overlap(
                            center, yaw, ego_length, ego_width,
                            obstacle_center, obstacle_yaw, obstacle_length, obstacle_width,
                        )
                        for obstacle_center, obstacle_yaw, obstacle_length, obstacle_width in dynamic_obstacles
                    )
                    if blocked:
                        continue
                    for obstacle in static_obstacles:
                        obstacle_center = np.asarray(obstacle["center"], dtype=np.float64)
                        if self._obb_overlap(
                            center, yaw, ego_length, ego_width,
                            obstacle_center, float(obstacle["yaw"]),
                            float(obstacle["length"]), float(obstacle["width"]),
                        ):
                            blocked = True
                            break
                    if blocked:
                        continue
                    next_state = (next_s, next_d, next_v_s, next_v_d)
                    next_frontier.setdefault(quantize(next_state), next_state)
            frontier = next_frontier
            if not frontier:
                break

        terminal_positions = {
            (round(state[0] / self.s_resolution), round(state[1] / self.d_resolution))
            for state in frontier.values()
        }
        area = len(terminal_positions) * self.s_resolution * self.d_resolution
        return {
            "area_m2": float(area),
            "solvable": bool(terminal_positions),
            "terminal_cell_count": int(len(terminal_positions)),
            "horizon_seconds": self.horizon,
        }


class AdversarialRiskMetrics:
    """统一管理每帧 EA 和每次攻击前后的可达集对比。"""

    def __init__(self, config=None):
        ea_config = None if config is None else _config_value(config, "ea", None)
        reach_config = None if config is None else _config_value(config, "reachability", None)
        self.ea_enabled = bool(_config_value(ea_config, "enabled", True))
        self.reachability_enabled = bool(_config_value(reach_config, "enabled", True))
        self.ea_solver = EvasiveAcceleration(ea_config)
        self.reachability = RouteReachability(reach_config)
        self.ea_values = []
        self.ea_undefined_frames = 0
        self.reachability_events = []
        self.reset_episode()

    def reset_episode(self):
        """清空当前场景统计，但不重建求解器。"""
        self._previous_headings = {}
        self._pending_baseline = None

    def begin_attack(self, env):
        """在新的攻击轨迹或障碍物进入环境前冻结原始可达集。"""
        if not self.reachability_enabled:
            return
        self._pending_baseline = self.reachability.compute(
            env,
            use_joint_trajectory=False,
            static_obstacles=env.get_static_obstacles(),
        )

    def evaluate_dangerous_reachability(self, env):
        """在扩散联合轨迹就绪后，于同一源时间步计算危险场景可达集。"""
        if not self.reachability_enabled or self._pending_baseline is None:
            return None
        dangerous = self.reachability.compute(
            env,
            use_joint_trajectory=True,
            static_obstacles=env.get_static_obstacles(),
        )
        original_area = float(self._pending_baseline["area_m2"])
        dangerous_area = float(dangerous["area_m2"])
        difficulty = float("nan") if original_area <= 0.0 else 1.0 - dangerous_area / original_area
        event = {
            "source_step": int(env.current_step),
            "original_area_m2": original_area,
            "dangerous_area_m2": dangerous_area,
            "original_solvable": bool(self._pending_baseline["solvable"]),
            "dangerous_solvable": bool(dangerous["solvable"]),
            "difficulty": float(difficulty),
        }
        self.reachability_events.append(event)
        self._pending_baseline = None
        return event

    @staticmethod
    def _road_user_from_array(state, yaw_rate=0.0):
        """将 Simulator 的 [x,y,vx,vy,yaw,length,width,...] 转换为 EA 输入。"""
        state = np.asarray(state, dtype=np.float64)
        return RoadUserState(
            x=float(state[0]),
            y=float(state[1]),
            speed=float(np.linalg.norm(state[2:4])),
            heading=float(state[4]),
            length=float(state[5]),
            width=float(state[6]),
            yaw_rate=float(yaw_rate),
        )

    def evaluate_ea(self, env):
        """计算当前帧自车与所有近场动态/静态对象的最大成对 EA。"""
        if not self.ea_enabled:
            return None
        dt = max(float(env.dt), 1e-6)
        ego = np.asarray(env.ego_state, dtype=np.float64)
        ego_previous = self._previous_headings.get("ego", float(ego[4]))
        ego_yaw_rate = _wrap_angle(float(ego[4]) - ego_previous) / dt
        ego_user = self._road_user_from_array(ego, ego_yaw_rate)
        values = []
        agents = np.asarray(env.data_dict["agent"][-1], dtype=np.float64)
        for agent_id, state in enumerate(agents):
            if agent_id >= len(env.agent_active) or not env.agent_active[agent_id]:
                continue
            if np.linalg.norm(state[:2] - ego[:2]) > 50.0:
                continue
            key = f"agent:{agent_id}"
            previous = self._previous_headings.get(key, float(state[4]))
            yaw_rate = _wrap_angle(float(state[4]) - previous) / dt
            values.append(self.ea_solver.compute(ego_user, self._road_user_from_array(state, yaw_rate)))
            self._previous_headings[key] = float(state[4])
        for obstacle in env.get_static_obstacles():
            if np.linalg.norm(np.asarray(obstacle["center"], dtype=float) - ego[:2]) > 50.0:
                continue
            obstacle_user = RoadUserState(
                x=float(obstacle["center"][0]),
                y=float(obstacle["center"][1]),
                speed=0.0,
                heading=float(obstacle["yaw"]),
                length=float(obstacle["length"]),
                width=float(obstacle["width"]),
                yaw_rate=0.0,
            )
            values.append(self.ea_solver.compute(ego_user, obstacle_user))
        self._previous_headings["ego"] = float(ego[4])
        finite_values = [value for value in values if np.isfinite(value)]
        if not finite_values:
            self.ea_undefined_frames += 1
            return float("nan")
        frame_ea = float(max(finite_values))
        self.ea_values.append(frame_ea)
        return frame_ea

    def compute_metrics(self):
        """返回可直接合并到现有 TTC/碰撞指标字典的新指标。"""
        metrics = {
            "max_evasive_acceleration_mps2": float(max(self.ea_values)) if self.ea_values else 0.0,
            "mean_evasive_acceleration_mps2": float(np.mean(self.ea_values)) if self.ea_values else 0.0,
            "ea_undefined_frames": int(self.ea_undefined_frames),
            "reachability_event_count": int(len(self.reachability_events)),
        }
        finite_events = [
            event for event in self.reachability_events if np.isfinite(event["difficulty"])
        ]
        if finite_events:
            hardest = max(finite_events, key=lambda event: event["difficulty"])
            metrics.update({
                "original_reachable_area_m2": hardest["original_area_m2"],
                "dangerous_reachable_area_m2": hardest["dangerous_area_m2"],
                "reachability_difficulty": hardest["difficulty"],
                "dangerous_scene_solvable": float(hardest["dangerous_solvable"]),
            })
        else:
            metrics.update({
                "original_reachable_area_m2": 0.0,
                "dangerous_reachable_area_m2": 0.0,
                "reachability_difficulty": float("nan"),
                "dangerous_scene_solvable": 0.0,
            })
        return metrics
