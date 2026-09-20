"""保存逐尝试闭环证据，并按真实活跃车辆统计背景占用重叠。"""
import json
import time
from pathlib import Path

import numpy as np

from policies.difficulty_audit import json_safe
from policies.joint_safety import obb_clearance


def plain_data(value):
    # 轨迹元数据包含 NumPy 数组，先转为普通容器再处理非有限数值。
    if isinstance(value, dict):
        return {key: plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_data(item) for item in value]
    if isinstance(value, (np.ndarray, np.generic)):
        return plain_data(value.tolist())
    return value


def collision_pairs(states, active, obstacles):
    """返回背景车辆对和车辆—静态物体对，排除未激活车辆。"""
    ids = np.flatnonzero(active)
    boxes = np.asarray(states)[ids][:, [0, 1, 4, 5, 6]]
    pairs, static_pairs = set(), set()
    for row, agent_id in enumerate(ids):
        for other in range(row):
            if obb_clearance(boxes[row], boxes[other]) < 0:
                pairs.add((int(ids[other]), int(agent_id)))
        for obstacle in obstacles:
            box = np.array([*obstacle['center'], obstacle['yaw'], obstacle['length'], obstacle['width']])
            if obb_clearance(boxes[row], box) < 0:
                static_pairs.add((int(agent_id), str(obstacle['id'])))
    return pairs, static_pairs


def actual_guidance(env):
    """直接读取已加载扩散器的参数，而非复制生成器期望值。"""
    controller = env.diffusion_controller
    net = controller.policy.nets['policy']
    params = net.guide_config.params
    return {
        'params': {key: params[key] for key in (
            'inner_lr', 'inner_beta', 'n_guide_steps', 'scale_grad_by_std', 'grad_wrt')},
        'weights': net.Loss_Calculater.weights.detach().cpu().tolist(),
        'sampling': dict(net.diffuse_args),
    }


class AttemptTrace:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'execution_trace.jsonl'
        # 每次尝试仅创建一次，意外复用目录时拒绝覆盖。
        self.path.touch(exist_ok=False)
        self.start = time.monotonic()
        self.frames = 0
        self.attack_executed_frames = 0
        self.background_frames = 0
        self.static_frames = 0
        self.background_pairs = set()
        self.static_pairs = set()
        self.background_events = 0
        self.static_events = 0
        self.previous_background = set()
        self.previous_static = set()
        self.initial_background = []
        self.initial_static = []

    def append(self, record):
        # 只写抽象状态、几何和指标，不保存环境变量或提供方凭据。
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(json_safe(plain_data(record)), ensure_ascii=False, allow_nan=False) + '\n')

    def state(self, env, info=None, initial=False):
        states = np.asarray(env.data_dict['agent'][-1])
        active = np.asarray(env.agent_active, dtype=bool)
        obstacles = env.get_static_obstacles()
        pairs, static_pairs = collision_pairs(states, active, obstacles)
        if initial:
            self.initial_background = sorted(pairs)
            self.initial_static = sorted(static_pairs)
        else:
            self.frames += 1
            self.background_frames += bool(pairs)
            self.static_frames += bool(static_pairs)
            self.background_events += len(pairs - self.previous_background)
            self.static_events += len(static_pairs - self.previous_static)
            self.background_pairs.update(pairs)
            self.static_pairs.update(static_pairs)
        self.previous_background, self.previous_static = pairs, static_pairs
        self.append({'kind': 'state', 'step': int(env.current_step),
                     'ego': np.asarray(env.ego_state).tolist(), 'agents': states.tolist(),
                     'active': active.tolist(), 'static_obstacles': obstacles,
                     'background_pairs': sorted(pairs), 'background_static_pairs': sorted(static_pairs),
                     'info': info or {}, 'initial': initial})

    def prediction(self, env, generator, elapsed):
        joint = env.pending_joint_trajectory
        controls = actual_guidance(env)
        if generator.difficulty_mode != 'off':
            expected = generator.difficulty_controller.parameters(generator._iterative_stage)
            for key in ('inner_lr', 'inner_beta', 'n_guide_steps'):
                if not np.isclose(controls['params'][key], expected[key]):
                    raise RuntimeError(f'实际扩散控制与记录不一致：{key}')
            if not np.allclose(controls['weights'], [1.5, 1.0, expected['scenario_ttc'], expected['llm_anchor']]):
                raise RuntimeError('实际扩散权重与连续强度不一致')
        self.append({'kind': 'prediction', 'step': int(env.current_step),
                     'planning_and_generation_seconds': elapsed,
                     'intensity': generator._iterative_stage, 'controls': controls,
                     'attack_intent': env.attack_intent,
                     'joint': {'agent_ids': joint.agent_ids.tolist(),
                               'positions': joint.positions_global.tolist(),
                               'yaws': joint.yaws_global.tolist(),
                               'valid': joint.valid_mask.tolist(),
                               'metadata': joint.metadata}})

    def attack_execution(self, env, joint):
        """只有含有效锚点的目标预测被真实执行才记攻击帧。"""
        if joint is None:
            return
        meta = joint.metadata
        target = meta.get('anchor_guidance_target_id')
        if not meta.get('anchor_guidance_active') or target is None:
            return
        rows = np.flatnonzero(joint.agent_ids == target)
        if len(rows) != 1:
            return
        row = int(rows[0])
        actual = np.asarray(env.data_dict['agent'][-1])[target, :2]
        executed = bool(joint.valid_mask[row, 0] and np.allclose(actual, joint.positions_global[row, 0], atol=1e-5))
        self.attack_executed_frames += int(executed)
        record = {'kind': 'attack_execution', 'step': int(env.current_step),
                     'request_id': meta.get('planner_request_id'), 'target_id': int(target),
                     'diffusion_source_step': int(joint.source_step), 'executed': executed}
        self.append(record)
        return record

    def summary(self):
        return {'evaluated_frames': self.frames,
                'background_collision_frames': self.background_frames,
                'background_collision_events': self.background_events,
                'background_collision_pairs': sorted(self.background_pairs),
                'background_static_collision_frames': self.static_frames,
                'background_static_collision_events': self.static_events,
                'background_static_collision_pairs': sorted(self.static_pairs),
                'initial_background_pairs': self.initial_background,
                'initial_background_static_pairs': self.initial_static,
                'wall_seconds': time.monotonic() - self.start,
                'execution_trace_path': str(self.path),
                'collision_basis': 'active_vehicle_OBB_at_executed_frames; events_are_contiguous_pair_contacts'}
