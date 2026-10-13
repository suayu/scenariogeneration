"""E1 五组共用的实际执行指标；预测可达性单独保留来源和覆盖率。"""
import math
from collections import Counter
import numpy as np

METRIC_PROTOCOL = "e1_obb_ttc_ea_v2_attempts_v3"


def box_axes(yaw):
    """矩形的纵轴和横轴，供连续平移 SAT 检查。"""
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    return np.array([[c, s], [-s, c]])


def obb_ttc(a, b, horizon=7.0):
    """恒速、固定航向矩形首次接触时刻；无相交返回 inf，重叠返回 0。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if (len(a) < 7 or len(b) < 7 or not np.isfinite(a[:7]).all()
            or not np.isfinite(b[:7]).all() or min(*a[5:7], *b[5:7]) <= 0
            or not np.isfinite(horizon) or horizon <= 0):
        raise ValueError("invalid OBB TTC inputs")
    aa, bb = box_axes(a[4]), box_axes(b[4])
    entry, leave = 0.0, float(horizon)
    delta, velocity = b[:2] - a[:2], b[2:4] - a[2:4]
    for axis in np.concatenate((aa, bb)):
        radius = np.abs(aa @ axis) @ (a[5:7] / 2) + np.abs(bb @ axis) @ (b[5:7] / 2)
        p, v = float(delta @ axis), float(velocity @ axis)
        if abs(v) < 1e-10:
            if abs(p) > radius:
                return float("inf")
            continue
        t1, t2 = sorted(((-radius - p) / v, (radius - p) / v))
        entry, leave = max(entry, t1), min(leave, t2)
        if entry > leave + 1e-10:
            return float("inf")
    return max(0.0, entry) if leave >= 0 else float("inf")


def center_tca(a, b):
    """旧指标保留为 TCA，不冒称几何 TTC。"""
    r, v = b[:2] - a[:2], b[2:4] - a[2:4]
    vv = float(v @ v)
    t = -float(r @ v) / vv if vv > 1e-6 else float("inf")
    return t if t > 0 and np.isfinite(t) else float("inf")


def polyline_distance(points, polylines):
    """到线段的精确投影距离，不把离散车道采样点当道路边界。"""
    points = np.asarray(points, float).reshape(-1, 2)
    lines = np.asarray(polylines, float)
    if lines.ndim == 2:
        lines = lines[None]
    starts, ends = lines[:, :-1].reshape(-1, 2), lines[:, 1:].reshape(-1, 2)
    vectors = ends - starts
    valid = np.sum(vectors * vectors, axis=1) > 1e-10
    starts, vectors = starts[valid], vectors[valid]
    if not len(starts):
        raise ValueError("degenerate lane/route geometry")
    result = []
    for chunk in np.array_split(points, max(1, int(np.ceil(len(points) / 1024)))):
        delta = chunk[:, None] - starts
        t = np.clip(np.sum(delta * vectors, axis=-1) / np.sum(vectors * vectors, axis=-1), 0, 1)
        result.extend(np.linalg.norm(delta - t[..., None] * vectors, axis=-1).min(axis=1))
    return np.asarray(result)


def geometric_lane_area(env, resolution=.5, half_width=1.75):
    """车道走廊栅格面积代理；非真实道路多边形，也非动力学可达面积。"""
    lanes = np.asarray(env.scenario_dict['lanes'], float)
    points = lanes.reshape(-1, 2)
    lo, hi = points.min(axis=0) - half_width, points.max(axis=0) + half_width
    xs = np.arange(lo[0] + resolution / 2, hi[0], resolution)
    ys = np.arange(lo[1] + resolution / 2, hi[1], resolution)
    if len(xs) * len(ys) > 2_000_000:
        return {"geometric_drivable_area_m2": None, "geometric_area_status": "grid_budget_exceeded"}
    import cv2
    grid = np.zeros((len(ys),len(xs)),dtype=np.uint8)
    # 全图固定栅格道路走廊并集；不把交通车辆占用误称静态道路面积损失。
    for lane in lanes:
        pixels = np.rint((lane-lo)/resolution).astype(np.int32)
        cv2.polylines(grid,[pixels],False,1,max(1,int(round(2*half_width/resolution))))
    return {"geometric_drivable_area_m2": float(grid.sum() * resolution ** 2),
            "geometric_area_status": "lane_corridor_proxy",
            "geometric_area_resolution_m": resolution, "geometric_area_lane_half_width_m": half_width,
            "geometric_area_basis": "union_of_lane_centerline_corridors_no_road_polygon"}


def trajectory_diagnostics(current, joint, dt):
    """审计插值/重规划导致的尖峰；诊断不偷偷改变基线轨迹或优化预算。"""
    if not len(joint.agent_ids):
        return {"actor_count": 0, "max_speed_mps": 0., "max_acceleration_mps2": 0., "max_jerk_mps3": 0.}
    states = np.asarray(current, float)[joint.agent_ids]
    pos = np.concatenate((states[:, None, :2], joint.positions_global), axis=1)
    vel = np.diff(pos, axis=1) / dt
    acc = np.diff(np.concatenate((states[:, None, 2:4], vel), axis=1), axis=1) / dt
    jerk = np.diff(acc, axis=1) / dt
    yaw = np.unwrap(np.concatenate((states[:, None, 4:5], joint.yaws_global), axis=1), axis=1)
    maximum = lambda values: float(np.linalg.norm(values, axis=-1).max()) if values.size else 0.
    return {"actor_count": len(states), "max_speed_mps": maximum(vel),
            "max_acceleration_mps2": maximum(acc), "max_jerk_mps3": maximum(jerk),
            "max_yaw_rate_radps": float(np.abs(np.diff(yaw, axis=1) / dt).max()),
            "basis": "sampled_future_and_real_current_state_no_posthoc_repair"}


class ExecutionMetrics:
    """初始/转移后状态审计，碰撞相对速度取接触前速度而非碰撞后清零值。"""
    def __init__(self, env):
        self.min_ttc = self.min_tca = self.min_distance = float("inf")
        self.collisions = []
        self.background_offroad_ids = set()
        self.previous_contacts = set()
        self.previous_ego = np.asarray(env.ego_state, float).copy()
        self.previous_agents = np.asarray(env.data_dict['agent'][-1], float).copy()
        self.previous_active = np.asarray(env.agent_active, bool).copy()
        self.ttc_valid_frames = 0
        self.frames = 0
        self.last_ttc = float("inf")
        self.last_tca = float("inf")

    def observe(self, env, info):
        ego = np.asarray(env.ego_state, float)
        states = np.asarray(env.data_dict['agent'][-1], float)
        active = np.asarray(env.agent_active, bool)
        # 帧内接触后被仿真器停用的车辆仍需审计，防止漏掉碰撞对象。
        ids = np.flatnonzero(active | self.previous_active)
        ttc, tca, contacts = float("inf"), float("inf"), set()
        for aid in ids:
            b = states[aid]
            candidate = obb_ttc(ego, b)
            # 停用对象仅保留当帧接触审计，不继续作为未来移动威胁。
            if active[aid] or candidate == 0:
                ttc, tca = min(ttc, candidate), min(tca, center_tca(ego, b))
            self.min_distance = min(self.min_distance, float(np.linalg.norm(ego[:2] - b[:2])))
            if candidate == 0:
                contacts.add(int(aid))
                if int(aid) not in self.previous_contacts:
                    ev, bv = self.previous_ego[2:4], self.previous_agents[aid, 2:4]
                    self.collisions.append({"step": int(env.current_step), "partner_id": int(aid),
                        "ego_speed_mps": float(np.linalg.norm(ev)), "partner_speed_mps": float(np.linalg.norm(bv)),
                        "relative_collision_speed_mps": float(np.linalg.norm(ev - bv)),
                        "velocity_basis": "last_pre_contact_observed_velocity"})
        if len(ids):
            distances = polyline_distance(states[ids, :2], env.scenario_dict['lanes'])
            self.background_offroad_ids.update(map(int, ids[distances > 3.0]))
        self.min_ttc, self.min_tca = min(self.min_ttc, ttc), min(self.min_tca, tca)
        self.last_ttc = ttc
        self.last_tca = tca
        self.ttc_valid_frames += 1
        self.frames += 1
        self.previous_contacts = contacts
        self.previous_ego, self.previous_agents, self.previous_active = ego.copy(), states.copy(), active.copy()
        return ttc

    def result(self):
        finite = lambda x: float(x) if np.isfinite(x) else None
        first = self.collisions[0] if self.collisions else {}
        return {"metric_protocol": METRIC_PROTOCOL, "min_ttc_s": finite(self.min_ttc),
                "min_ttc_capped_7s": min(self.min_ttc, 7.) if self.frames else None,
                "min_tca_s": finite(self.min_tca), "ttc_basis": "constant_velocity_fixed_yaw_continuous_OBB_contact",
                "ttc_horizon_s": 7.0, "ttc_valid_frame_count": self.ttc_valid_frames,
                "ttc_no_contact_in_horizon": self.frames > 0 and not np.isfinite(self.min_ttc),
                "min_center_distance_m": finite(self.min_distance), "ego_collision_events": self.collisions,
                "ego_collision_speed_mps": first.get("ego_speed_mps"),
                "partner_collision_speed_mps": first.get("partner_speed_mps"),
                "relative_collision_speed_mps": first.get("relative_collision_speed_mps"),
                "background_offroad_agent_ids": sorted(self.background_offroad_ids),
                "background_offroad": bool(self.background_offroad_ids),
                "background_offroad_basis": "center_to_lane_polyline_distance_gt_3m_diagnostic_only"}


def aggregate_attempts(records, planned_count=None):
    """失败不丢弃；明确预定尝试、实际执行与正常完成三类分母。"""
    planned = len(records) if planned_count is None else int(planned_count)
    if planned < len(records) or planned <= 0:
        raise ValueError("invalid planned denominator")
    executed = [r for r in records if r.get('executed_steps', 0) > 0]
    completed = [r for r in executed if not r.get('generation_failure') and not r.get('runtime_failure')
                 and not r.get('planner_service_failure_count')]
    collisions = sum(r.get('collision') is True for r in records)
    values = lambda key: [float(r[key]) for r in records if r.get(key) is not None and np.isfinite(r[key])]
    mean = lambda key: float(np.mean(values(key))) if values(key) else None
    result = {"planned_scenario_count": planned, "attempted_scenario_count": len(records),
        "unattempted_scenario_count": planned-len(records), "actual_execution_scenario_count": len(executed),
        "normally_completed_attempt_count": len(completed), "execution_coverage": len(executed)/planned,
        "collision_count": collisions, "collision_yield_per_planned_attempt": collisions/planned,
        "collision_rate": sum(r.get('collision') is True for r in completed)/len(completed) if completed else None,
        "collision_rate_among_actual_execution": sum(r.get('collision') is True for r in executed)/len(executed) if executed else None,
        "outcome_rate_denominator": "normally_completed_attempts; partial_execution_reported_separately",
        "generation_failure_count": sum(bool(r.get('generation_failure')) for r in records),
        "runtime_failure_count": sum(bool(r.get('runtime_failure')) for r in records),
        "planner_service_failure_count": sum(int(r.get('planner_service_failure_count',0)) for r in records),
        "prediction_service_failure_count": sum(int(r.get('prediction_service_failure_count',0)) for r in records),
        "attack_plan_count": sum(int(r.get('attack_plan_count',0)) for r in records),
        "attack_scene_count": sum(r.get('attack_executed_frames',0)>0 for r in records),
        "attack_executed_frames": sum(int(r.get('attack_executed_frames',0)) for r in records),
        "danger_valid_count": len(values('scenario_danger_score')),
        "danger_valid_frame_count": sum(int(r.get('danger_valid_frame_count',0)) for r in records),
        "danger_total_frame_count": sum(int(r.get('executed_steps',0)) for r in records),
        "background_collision_frames": sum(r.get('feasibility',{}).get('background_collision_frames',0) for r in records),
        "background_static_collision_frames": sum(r.get('feasibility',{}).get('background_static_collision_frames',0) for r in records)}
    for k in ['min_ttc_s','min_ttc_capped_7s','min_tca_s','mean_evasive_acceleration_mps2','max_evasive_acceleration_mps2',
              'scenario_danger_score','progress','relative_collision_speed_mps','mean_reachable_area_m2','runtime_seconds']:
        result['mean_'+k] = mean(k)
        result[k+'_observed_count'] = len(values(k))
    result['danger_frame_coverage'] = result['danger_valid_frame_count']/result['danger_total_frame_count'] if result['danger_total_frame_count'] else 0.
    result['ttc_no_contact_scene_count'] = sum(bool(r.get('ttc_no_contact_in_horizon')) for r in records)
    result['generation_failure_reasons'] = dict(Counter(r.get('generation_failure_reason') or 'unspecified' for r in records if r.get('generation_failure')))
    result['runtime_failure_reasons'] = dict(Counter(r.get('runtime_failure_reason') or 'unspecified' for r in records if r.get('runtime_failure')))
    for k in ['off_route','completed']:
        result[k+'_rate_among_actual_execution'] = sum(r.get(k) is True for r in executed)/len(executed) if executed else None
    return result
