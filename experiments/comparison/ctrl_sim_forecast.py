"""仅用于公共风险评估的 CtRL-Sim 隔离联合未来，不改变真实交通执行。"""

import contextlib
import copy
import io
import math
import time

import numpy as np
import torch

from policies.strive_adapter import isolated_random_state
from policies.traffic_types import JointTrajectory


def clone_model_state(model):
    """共享只读网络参数，复制模块、缓冲区及运行状态，避免每帧复制权重。"""
    return copy.deepcopy(model, {id(p): p for p in model.parameters()})


def validate_forecast(joint, env, horizon_seconds):
    """拒绝过期、缺车、短时域或无效预测，不以恒速填补缺失测量。"""
    if joint.source_step != env.current_step:
        raise ValueError("forecast_source_step_mismatch")
    if set(joint.agent_ids.tolist()) != set(np.flatnonzero(env.agent_active).tolist()):
        raise ValueError("forecast_active_agent_mismatch")
    required = int(math.ceil(horizon_seconds / env.dt - 1e-9))
    if joint.positions_global.shape[1] < required:
        raise ValueError("forecast_horizon_too_short")
    if not joint.valid_mask[:, :required].all():
        raise ValueError("forecast_incomplete_agent_coverage")
    for values in (joint.positions_global, joint.yaws_global, joint.velocities_global):
        if not np.isfinite(values[:, :required]).all():
            raise ValueError("forecast_nonfinite")


class PersistentForecastDataset:
    """副本中区分观测视野和物理存在；仅持续跟踪源帧已有车辆，不填充未来位置。"""

    def __init__(self, original, tracked, existence=None):
        self.original = original
        self.tracked = np.asarray(tracked, dtype=bool).copy()
        self.existence = existence

    def __getattr__(self, name):
        return getattr(self.original, name)

    def get_agent_mask(self, states, normalize_dict, fov=None):
        base = self.original.get_agent_mask(states, normalize_dict, fov=fov)
        finite = np.isfinite(np.asarray(states)[..., :5]).all(axis=-1)
        if len(states) != len(self.tracked):
            raise ValueError("tracked forecast stable-ID dimension mismatch")
        tracked = np.broadcast_to(self.tracked[:, None], base.shape) & finite
        if self.existence is not None:
            exists = self.existence()
            if exists.shape != base.shape:
                raise ValueError("tracked forecast history mask mismatch")
            # 几何可见也不能把不存在的历史/未来填充格误当真实观测。
            return (base | tracked) & exists & finite
        return (base | tracked) & finite


class CtrlSimJointForecaster:
    """从当前真实历史重建模型上下文，在独立世界滚动生成同一联合样本。"""

    @isolated_random_state
    def __init__(self, env, horizon_seconds=3.0, seed=42, tracking="observation"):
        from simulator import CtRLSim, CtRLSimBehaviourModel
        self.horizon_seconds, self.seed = float(horizon_seconds), int(seed)
        if tracking not in ("observation", "persistent"):
            raise ValueError("unsupported forecast tracking mode")
        self.tracking = tracking
        if not np.isfinite(self.horizon_seconds) or self.horizon_seconds <= 0:
            raise ValueError("forecast horizon must be positive")
        if env.behaviour_model is not None:
            self.template = env.behaviour_model
        else:
            cfg = env.cfg.sim.behaviour_model
            # SAFE-SIM 本身仍使用 diffusion；该网络仅在显式选择评估支路时加载。
            self.template = CtRLSimBehaviourModel(
                mode=env.mode, model_path=cfg.model_path,
                model=CtRLSim.load_from_checkpoint(cfg.model_path).to('cuda'),
                dset=env.ctrl_sim_dset, use_rtg=cfg.use_rtg,
                predict_rtgs=cfg.predict_rtgs, action_temperature=cfg.action_temperature,
                tilt=cfg.tilt, steps=env.steps + int(math.ceil(self.horizon_seconds / env.dt)) + 1,
            )

    def _behaviour_copy(self, env, horizon):
        from simulator import MAX_RTG_VAL, inverse_k_disks
        original = env.behaviour_model or self.template
        # 模型/数据集单独处理，避免复制静态数据集和冻结网络参数。
        model = clone_model_state(original.model)
        result = copy.deepcopy(original, {id(original.model): model, id(original.dset): original.dset})
        capacity = max(env.t + horizon + 1, int(original.cfg_dataset.train_context_length))
        if env.behaviour_model is not None:
            for name in ('states', 'actions', 'rtgs'):
                values = getattr(result, name)
                if values.shape[1] < capacity:
                    padding = [(0, 0)] * values.ndim
                    padding[1] = (0, capacity - values.shape[1])
                    setattr(result, name, np.pad(values, padding))
            return result
        # diffusion 没有 CtRL 动作 token；只用已发生的真实状态反解，不读取未来真值。
        result.steps = capacity
        count = len(env.agent_active)
        result.reset(count + 1)
        ego = np.asarray(env.data_dict['ego']).reshape(-1, 8)
        agents = np.asarray(env.data_dict['agent'])
        result.states[0, :env.t + 1] = ego
        result.states[1:, :env.t + 1] = agents.transpose(1, 0, 2)
        result.types[0] = env.data_dict['ego_type'][0][0]
        result.types[1:] = env.data_dict['agent_type'][0]
        result.rtgs[:] = MAX_RTG_VAL
        for k in range(env.t):
            result.actions[0, k] = np.asarray(inverse_k_disks(ego[k], ego[k + 1], env.ctrl_sim_dset.V)).item()
            for row in range(count):
                result.actions[row + 1, k] = np.asarray(inverse_k_disks(agents[k, row], agents[k + 1, row], env.ctrl_sim_dset.V)).item()
        result.t = env.t
        # SAFE-SIM 的历史 RTG 不可观测，显式记录最大 RTG 占位初始化，不能冒称原始 token。
        for name in ('has_collided', 'has_offroad', 'has_activated', 'has_activated_vehicle'):
            setattr(result, name, np.zeros(count, dtype=bool))
        return result

    @isolated_random_state
    def predict(self, env, policy):
        """复制环境、模型状态及自车策略，预测后恢复全部随机数流。"""
        start = time.monotonic()
        source_step = int(env.current_step)
        horizon = int(math.ceil(self.horizon_seconds / env.dt - 1e-9))
        sample_seed = (self.seed + 1000003 * source_step) % (2**31 - 1)
        torch.random.default_generator.manual_seed(sample_seed)
        if torch.cuda.is_initialized():
            torch.cuda.manual_seed(sample_seed)
        np.random.seed(sample_seed)
        clone = copy.copy(env)
        for name in ('data_dict', 'ego_state', 'agent_active', 'left_scene',
                     'last_active_agent_position', 'local_frame', 'rl_kinematics_model',
                     'current_state', 'viz_state', 'activate_agent_ids'):
            setattr(clone, name, copy.deepcopy(getattr(env, name)))
        clone.cfg = copy.deepcopy(env.cfg)
        clone.cfg.sim.steps = source_step + horizon + 1
        clone.steps = source_step + horizon + 1
        clone.traffic_backend = 'ctrl_sim'
        clone.pending_joint_trajectory = None
        clone.diffusion_controller = None
        clone.behaviour_model = self._behaviour_copy(env, horizon)
        if env.behaviour_model is None:
            # update_state 会读取最后一个已发生的动作/RTG，必须同步重建后的 token。
            clone.data_dict['agent_action'] = [
                clone.behaviour_model.actions[1:, k].copy() for k in range(env.t)]
            clone.data_dict['agent_rtg'] = [
                clone.behaviour_model.rtgs[1:, k, 0].copy() for k in range(env.t)]
        if self.tracking == "persistent":
            tracked = np.asarray(env.agent_active, dtype=bool)
            clone.ctrl_sim_dset = PersistentForecastDataset(env.ctrl_sim_dset, tracked)
            behaviour = clone.behaviour_model
            def existence():
                width = int(behaviour.cfg_dataset.train_context_length)
                start = max(0, behaviour.t - width + 1)
                return behaviour.states[:, start:start + width, -1].astype(bool)
            # 历史存在掩码保持真实；不能把未来缓冲区零值误当观测历史。
            behaviour.dset = PersistentForecastDataset(env.ctrl_sim_dset,
                                                      np.r_[True, tracked], existence)
        clone.save_step_data = lambda: None
        clone._update_viz_state = lambda *a, **kw: None
        probe_policy = (policy.clone_for_rollout(clone) if hasattr(policy, 'clone_for_rollout')
                        else copy.deepcopy(policy, {id(env): clone}))
        ids = np.flatnonzero(env.agent_active)
        observations = clone._get_observation()
        future, masks, exits = [], [], []
        # 模拟终止后的预测用于理论可达性时域；不得把它记为真实执行帧。
        with torch.inference_mode(), contextlib.redirect_stdout(io.StringIO()):
            for _ in range(horizon):
                observations, _, _ = clone.step(probe_policy.act(observations), None, None)
                future.append(clone.data_dict['agent'][-1][ids].copy())
                masks.append(clone.agent_active[ids].copy())
                for row, agent_id in enumerate(ids):
                    if not clone.agent_active[agent_id] and (len(masks) == 1 or masks[-2][row]):
                        state = clone.data_dict['agent'][-1][agent_id]
                        in_fov = bool(env.ctrl_sim_dset.get_agent_mask(state[None, None, :5], clone.local_frame)[0, 0])
                        exits.append({'agent_id': int(agent_id), 'future_index': len(masks) - 1,
                                      'outside_observation_fov': not in_fov,
                                      'offroad': bool(clone.behaviour_model.has_offroad[agent_id]),
                                      'collision': bool(clone.behaviour_model.has_collided[agent_id])})
        states = np.stack(future, axis=1)
        metadata = {
            'prediction_source': 'ctrl_sim_rollout', 'source_step': source_step,
            'tracking_mode': self.tracking, 'tracking_exits': exits,
            'sample_id': f'{env.current_scene_id}:{source_step}:{sample_seed}',
            'seed': sample_seed, 'horizon_seconds': horizon * env.dt,
            'elapsed_seconds': time.monotonic() - start,
            'ego_conditioning_policy': type(policy).__name__,
            'execution_backend': env.traffic_backend,
            'traffic_forecast_is_surrogate': env.traffic_backend != 'ctrl_sim',
            'history_actions': ('original_ctrl_sim_tokens' if env.behaviour_model is not None
                                else 'inverse_kinematics_from_observed_states'),
            'history_rtg': ('original_ctrl_sim_tokens' if env.behaviour_model is not None
                            else 'max_rtg_placeholder_unobserved'),
            'counterfactual_after_termination': True,
        }
        return JointTrajectory(source_step, ids, states[..., :2], states[..., 4:5],
                               states[..., 2:4], np.stack(masks, axis=1), metadata)
