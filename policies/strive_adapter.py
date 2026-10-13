"""STRIVE 官方模型的边界适配；不导入其全局 datasets/models 命名空间。"""
import copy
import functools
import json
import queue
import random
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from policies.traffic_types import JointTrajectory


OFFICIAL_COMMIT = "b708951f8665c97a1de9ed93b4ed3f58dd8cbf5d"


def resample_trajectory(current, future, model_dt, execution_dt):
    """保留真实起点，插值位置与展开后的航向；禁止超出模型时域外推。"""
    current, future = np.asarray(current), np.asarray(future)
    if current.ndim != 2 or current.shape[1] < 5:
        raise ValueError("current must be [A,>=5]")
    if future.ndim != 3 or future.shape[0] != len(current) or future.shape[2] != 4:
        raise ValueError("future must be [A,T,4] with x,y,heading_cos,heading_sin")
    if model_dt <= 0 or execution_dt <= 0 or future.shape[1] < 1:
        raise ValueError("invalid trajectory time scale")
    if not np.isfinite(current).all() or not np.isfinite(future).all():
        raise ValueError("nonfinite STRIVE trajectory")
    if not np.allclose(np.linalg.norm(future[..., 2:4], axis=-1), 1.0, atol=1e-3):
        raise ValueError("STRIVE heading must be a unit vector")
    knots = np.arange(future.shape[1] + 1) * model_dt
    times = np.arange(1, int(np.floor(knots[-1] / execution_dt + 1e-8)) + 1) * execution_dt
    if not len(times):
        raise ValueError("execution step exceeds model horizon")
    positions = np.empty((len(current), len(times), 2), dtype=np.float32)
    yaws = np.empty((len(current), len(times), 1), dtype=np.float32)
    for row in range(len(current)):
        for dim in range(2):
            positions[row, :, dim] = np.interp(times, knots, np.r_[current[row, dim], future[row, :, dim]])
        angles = np.unwrap(np.r_[current[row, 4], np.arctan2(future[row, :, 3], future[row, :, 2])])
        yaws[row, :, 0] = np.interp(times, knots, angles)
    velocities = np.diff(np.concatenate((current[:, None, :2], positions), axis=1), axis=1) / execution_dt
    return positions, yaws, velocities.astype(np.float32)


def make_request(frame, ego_history, ego_mask, lane_width=3.5):
    """按0.5秒采样真实历史；缺失历史标为NaN，未来仅用CV初始化并明确记录。"""
    if lane_width <= 0:
        raise ValueError("lane_width must be positive")
    ids = frame.agent_ids[frame.active_mask]
    if len(ids) == 0 or frame.ego_state_global is None:
        raise ValueError("STRIVE requires ego and interacting vehicles")
    if not (frame.agent_types[frame.active_mask, 1] == 1).all():
        raise ValueError("official checkpoint supports vehicles only")
    all_current = np.vstack((frame.ego_state_global, frame.states_global[frame.active_mask]))
    history = np.concatenate((np.asarray(ego_history)[None], frame.history_global[frame.active_mask]), axis=0)
    visible = np.concatenate((np.asarray(ego_mask)[None], frame.history_mask[frame.active_mask]), axis=0)
    if history.shape[:2] != visible.shape or history.shape[1] != frame.history_global.shape[1]:
        raise ValueError("ego/background histories must be aligned")
    # 正平移避免官方地图索引将负坐标错误映射到原点，不改变相对几何。
    points = np.concatenate((frame.lanes_global.reshape(-1, 2), all_current[:, :2]), axis=0)
    if not np.isfinite(points).all():
        raise ValueError("nonfinite scene geometry")
    origin = points.min(axis=0) - 100.0
    past = np.full((len(all_current), 4, 6), np.nan, dtype=np.float32)
    past_vis = np.zeros((len(all_current), 4), dtype=np.float32)
    for k in range(4):
        offset = (k - 3) * .5 / frame.dt
        idx = history.shape[1] - 1 + int(round(offset))
        if idx < 0 or not np.isclose(offset, round(offset), atol=1e-5):
            continue
        states = history[:, idx]
        valid = visible[:, idx]
        past[valid, k, :2] = states[valid, :2] - origin
        past[valid, k, 2:4] = np.column_stack((np.cos(states[valid, 4]), np.sin(states[valid, 4])))
        past[valid, k, 4] = np.linalg.norm(states[valid, 2:4], axis=-1)
        past[valid, k, 5] = 0.0
        if idx > 0:
            both = valid & visible[:, idx - 1]
            delta = states[both, 4] - history[both, idx - 1, 4]
            past[both, k, 5] = np.arctan2(np.sin(delta), np.cos(delta)) / frame.dt
        past_vis[valid, k] = 1.0
    if not past_vis[:, -1].all():
        raise ValueError("every controlled vehicle needs a real current state")
    future = np.empty((len(all_current), 12, 6), dtype=np.float32)
    for k in range(12):
        future[:, k, :2] = all_current[:, :2] - origin + all_current[:, 2:4] * ((k + 1) * .5)
        future[:, k, 2:4] = np.column_stack((np.cos(all_current[:, 4]), np.sin(all_current[:, 4])))
        future[:, k, 4] = np.linalg.norm(all_current[:, 2:4], axis=-1)
        future[:, k, 5] = 0.0
    shape_xy = np.ceil(points.max(axis=0) - origin + 100).astype(int) * 4 + 1
    if shape_xy.max() > 8000 or np.prod(shape_xy) > 32_000_000:
        raise ValueError("map raster exceeds fixed memory budget")
    raster = np.zeros((4, shape_xy[1], shape_xy[0]), dtype=np.uint8)
    for lane in frame.lanes_global:
        pixels = np.rint((lane - origin) * 4).astype(np.int32)
        cv2.polylines(raster[0], [pixels], False, 1, max(1, int(round(lane_width * 4))))
    return {
        "request_id": f"{frame.scene_id}:{frame.step}", "source_step": int(frame.step),
        "agent_ids": ids.tolist(), "origin": origin.tolist(), "past": past,
        "past_vis": past_vis, "future": future, "lw": all_current[:, 5:7],
        "raster": raster, "map_semantics": "lane_corridor_proxy; carpark/road_divider/lane_divider unavailable",
        "future_initialization": "constant_velocity_no_ground_truth_future",
        "lane_width_m": float(lane_width),
    }


def isolated_random_state(function):
    """反事实调用不得消耗真实实验的随机数流。"""
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        import torch
        python_state, numpy_state = random.getstate(), np.random.get_state()
        devices = [torch.cuda.current_device()] if torch.cuda.is_initialized() else []
        try:
            with torch.random.fork_rng(devices=devices):
                return function(*args, **kwargs)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
    return wrapped


@isolated_random_state
def counterfactual_rollout(env, policy, joint):
    """在独立状态和独立策略副本上执行候选背景轨迹，保留真实策略反馈。"""
    clone = copy.copy(env)
    for name in ("data_dict", "ego_state", "agent_active", "left_scene", "last_active_agent_position",
                 "local_frame", "rl_kinematics_model", "current_state", "viz_state"):
        setattr(clone, name, copy.deepcopy(getattr(env, name)))
    clone.cfg = copy.deepcopy(env.cfg)
    clone.cfg.sim.steps = env.t + joint.positions_global.shape[1] + 1
    clone.behaviour_model = None
    clone.traffic_backend = "external_joint"
    clone.save_step_data = lambda: None
    clone._update_viz_state = lambda *a, **kw: None
    # 策略副本保留在线状态；新策略可实现clone_for_rollout以避免复制大型网络。
    if hasattr(policy, "clone_for_rollout"):
        probe_policy = policy.clone_for_rollout(clone)
    else:
        probe_policy = copy.deepcopy(policy, {id(env): clone})
    observations = clone._get_observation()
    trajectory = []
    for k in range(joint.positions_global.shape[1]):
        # 新进入视野的车辆保持CV，且记录于最终方法保真性说明。
        ids = np.flatnonzero(clone.agent_active)
        states = clone.data_dict['agent'][-1][ids]
        positions = states[:, None, :2] + states[:, None, 2:4] * env.dt
        yaws = states[:, None, 4:5].copy()
        velocities = states[:, None, 2:4].copy()
        source_rows = {int(a): r for r, a in enumerate(joint.agent_ids)}
        for row, agent_id in enumerate(ids):
            if int(agent_id) in source_rows:
                r = source_rows[int(agent_id)]
                positions[row, 0] = joint.positions_global[r, k]
                yaws[row, 0] = joint.yaws_global[r, k]
                velocities[row, 0] = joint.velocities_global[r, k]
        clone.inject_joint_trajectory(JointTrajectory(clone.current_step, ids, positions, yaws, velocities, np.ones((len(ids), 1), bool)))
        observations, _, _ = clone.step(probe_policy.act(observations), None, None)
        state = clone.ego_state
        trajectory.append([state[0], state[1], np.cos(state[4]), np.sin(state[4])])
    return np.asarray(trajectory)


class StriveController:
    """子进程隔离官方包，带请求ID、超时和真实自车策略的优化回调。"""
    def __init__(self, root, python, checkpoint, directory, init_iters=75, adv_iters=300, timeout=1800):
        self.root, self.python, self.checkpoint = Path(root), str(python), Path(checkpoint)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.init_iters, self.adv_iters, self.timeout = int(init_iters), int(adv_iters), float(timeout)
        if self.init_iters < 1 or self.adv_iters < 1 or self.timeout <= 0:
            raise ValueError("STRIVE budgets must be positive")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Official STRIVE checkpoint missing: {self.checkpoint}")
        actual = subprocess.check_output(['git', '-C', str(self.root), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != OFFICIAL_COMMIT:
            raise ValueError("STRIVE official commit mismatch")

    def predict(self, env, policy):
        frame = env.get_scenario_frame(history_frames=16)
        ego_history = np.zeros((16, 8), np.float32)
        ego_mask = np.zeros(16, bool)
        observed = np.asarray(env.data_dict['ego'][-16:]).reshape(-1, 8)
        ego_history[-len(observed):], ego_mask[-len(observed):] = observed, True
        request = make_request(frame, ego_history, ego_mask)
        folder = self.directory / f"step_{frame.step:04d}"
        folder.mkdir(exist_ok=False)
        np.savez_compressed(folder / 'input.npz', **{k: request[k] for k in ('past','past_vis','future','lw','raster')})
        metadata = {k: v for k, v in request.items() if k not in ('past','past_vis','future','lw','raster')}
        (folder / 'input.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        worker = Path(__file__).resolve().parents[1] / 'experiments/comparison/strive_official_worker.py'
        cmd = [self.python, str(worker), '--root', str(self.root), '--checkpoint', str(self.checkpoint),
               '--input', str(folder), '--init-iters', str(self.init_iters), '--adv-iters', str(self.adv_iters),
               '--solution-iters', str(self.adv_iters)]
        started = time.monotonic()
        messages = queue.Queue()
        with (folder / 'worker.log').open('w', encoding='utf-8') as log:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
            def read_messages():
                for line in proc.stdout:
                    messages.put(line)
                messages.put(None)
            threading.Thread(target=read_messages, daemon=True).start()
            calls = 0
            try:
                while True:
                    remaining = self.timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError('STRIVE official optimizer timeout')
                    line = messages.get(timeout=remaining)
                    if line is None:
                        raise RuntimeError(f'STRIVE worker terminated before result; code={proc.wait()}')
                    message = json.loads(line)
                    if message.get('request_id') != request['request_id']:
                        raise RuntimeError('STRIVE response request_id mismatch')
                    if message['kind'] == 'rollout':
                        if message.get('callback_id') != calls:
                            raise RuntimeError('duplicate or unordered STRIVE callback')
                        calls += 1
                        future = np.asarray(message['future'], np.float32)
                        future[..., :2] += np.asarray(request['origin'])
                        p, y, v = resample_trajectory(frame.states_global[frame.active_mask], future, .5, frame.dt)
                        candidate = JointTrajectory(frame.step, frame.agent_ids[frame.active_mask], p, y, v, np.ones(p.shape[:2], bool))
                        ego = counterfactual_rollout(env, policy, candidate)
                        indices = np.rint(np.arange(1, 13) * .5 / frame.dt).astype(int) - 1
                        ego = ego[indices]
                        ego[:, :2] -= request['origin']
                        response = {"request_id": request['request_id'], "callback_id": calls - 1, "ego": ego.tolist()}
                        proc.stdin.write(json.dumps(response) + '\n')
                        proc.stdin.flush()
                    elif message['kind'] == 'result':
                        if message.get('model_loaded_strict') is not True:
                            raise RuntimeError('STRIVE model not strictly loaded')
                        code = proc.wait(timeout=min(30, remaining))
                        if code != 0:
                            raise RuntimeError(f'STRIVE worker failed: {code}')
                        if message.get('status') == 'generation_failure':
                            from policies.joint_safety import NoSafeJointCandidate
                            raise NoSafeJointCandidate(message.get('reason', 'STRIVE generation failure'))
                        future = np.asarray(message['future'], np.float32)[1:]
                        future[..., :2] += np.asarray(request['origin'])
                        p,y,v = resample_trajectory(frame.states_global[frame.active_mask], future, .5, frame.dt)
                        audit = {**metadata, **{k: val for k,val in message.items() if k != 'future'},
                                 'planner_rollout_calls': calls, 'elapsed_seconds': time.monotonic()-started,
                                 'fidelity': 'official_STRIVE_optimizer_adapted_Waymo_map_and_CV_initialization',
                                 'formal_ready': False}
                        (folder / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
                        np.savez_compressed(folder / 'output.npz', positions=p, yaws=y, velocities=v)
                        return JointTrajectory(frame.step, frame.agent_ids[frame.active_mask], p,y,v,np.ones(p.shape[:2],bool), audit)
                    else:
                        raise RuntimeError('invalid STRIVE message kind')
            finally:
                if proc.poll() is None:
                    proc.kill()
                proc.wait()
                proc.stdin.close()
                proc.stdout.close()
