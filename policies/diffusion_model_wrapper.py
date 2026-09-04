"""在 Scenario Dreamer 仿真器中使用 Safe-Sim 的生命周期封装。"""

import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
from pathlib import Path

import torch
import numpy as np

from policies.safe_sim_adapter import SafeSimBatchAdapter
from policies.traffic_types import JointTrajectory, ScenarioFrame


def _move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _slice_batch(value, indices, batch_size):
    """仅沿参与者批次维切分推理输入，保留场景级标量不变。"""
    if torch.is_tensor(value):
        if value.ndim > 0 and value.shape[0] == batch_size:
            return value.index_select(0, indices)
        return value
    if isinstance(value, dict):
        return {key: _slice_batch(item, indices, batch_size) for key, item in value.items()}
    if isinstance(value, list) and len(value) == batch_size:
        return [value[index] for index in indices.tolist()]
    if isinstance(value, tuple) and len(value) == batch_size:
        return tuple(value[index] for index in indices.tolist())
    return value


def _to_plain_value(value):
    """将 Hydra/OmegaConf 容器递归转换为普通 Python 容器。"""
    if isinstance(value, Mapping):
        return {str(key): _to_plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_value(item) for item in value]
    return value


class DiffusionModelWrapper:
    """一次加载 Safe-Sim、逐场景重置，并在每帧预测联合轨迹。"""

    def __init__(
        self,
        safe_sim_root,
        config_path,
        checkpoint_path,
        device="cuda:0",
        history_frames=11,
        prediction_horizon=32,
        max_neighbors=20,
        num_samples=20,
        sample_step=1,
        diffusion_agent_limit=0,
        far_agent_mode="guided",
        mixed_precision=False,
        multi_gpu_devices=None,
        guidance_config=None,
        anchor_guidance_enabled=False,
        anchor_guidance_strength=0.2,
        anchor_interval_seconds=1.0,
        anchor_robust_delta=1.0,
        anchor_inner_lr=0.2,
        anchor_max_update=0.5,
        anchor_guide_steps=1,
        anchor_scale_grad_by_std=True,
        capture_candidate_trajectories=False,
    ):
        self.safe_sim_root = Path(safe_sim_root).expanduser().resolve()
        self.config_path = Path(config_path).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self._validate_paths()
        self._install_safe_sim_import_paths()

        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Safe-Sim device {device!r} requested, but CUDA is unavailable")
        self.device = torch.device(device)
        self.prediction_horizon = int(prediction_horizon)
        self.num_samples = int(num_samples)
        self.sample_step = int(sample_step)
        self.diffusion_agent_limit = int(diffusion_agent_limit)
        self.far_agent_mode = str(far_agent_mode).lower()
        self.mixed_precision = bool(mixed_precision)
        # 仅在可视化明确开启时保留候选样本，避免常规闭环仿真产生额外 CPU 解码开销。
        self.capture_candidate_trajectories = bool(capture_candidate_trajectories)
        if self.diffusion_agent_limit < 0:
            raise ValueError("diffusion_agent_limit must be non-negative")
        if self.far_agent_mode not in {"guided", "unguided_diffusion"}:
            raise ValueError("far_agent_mode must be guided or unguided_diffusion")
        self.parallel_devices = self._resolve_parallel_devices(multi_gpu_devices)
        self.guidance_config = _to_plain_value(guidance_config or {})
        legacy_anchor_enabled = bool(anchor_guidance_enabled)
        self.guidance_mode = str(
            self.guidance_config.get(
                "mode", "llm_joint" if legacy_anchor_enabled else "unguided"
            )
        ).lower()
        valid_modes = {"unguided", "default", "llm_joint", "manual"}
        if self.guidance_mode not in valid_modes:
            raise ValueError(
                f"guidance mode must be one of {sorted(valid_modes)}, got {self.guidance_mode!r}"
            )
        self.anchor_guidance_strength = float(anchor_guidance_strength)
        self.anchor_interval_seconds = float(anchor_interval_seconds)
        self.anchor_robust_delta = float(anchor_robust_delta)
        self.anchor_inner_lr = float(anchor_inner_lr)
        self.anchor_max_update = float(anchor_max_update)
        self.anchor_guide_steps = int(anchor_guide_steps)
        self.anchor_scale_grad_by_std = bool(anchor_scale_grad_by_std)
        if self.anchor_guidance_strength < 0:
            raise ValueError("anchor_guidance_strength must be non-negative")
        if self.anchor_interval_seconds <= 0:
            raise ValueError("anchor_interval_seconds must be positive")
        if self.anchor_robust_delta <= 0:
            raise ValueError("anchor_robust_delta must be positive")
        if self.anchor_inner_lr < 0:
            raise ValueError("anchor_inner_lr must be non-negative")
        if self.anchor_max_update <= 0:
            raise ValueError("anchor_max_update must be positive")
        if self.anchor_guide_steps < 1:
            raise ValueError("anchor_guide_steps must be at least one")
        self.scene_id = None
        self.active_guidance_functions = []
        self.guidance_enabled = False
        self.uses_anchor_guidance = False
        # 性能诊断默认关闭；开启时才同步 CUDA 并记录分段耗时，不影响常规接口。
        self.performance_diagnostics = os.environ.get("SAFE_SIM_PERF_DIAGNOSTICS") == "1"
        self.performance_records = []

        self.policy, self.exp_config = self._load_policy()
        # 多卡模式为每张卡加载持久副本；默认列表仅包含原有的单卡模型。
        self.policy_replicas = [(self.device, self.policy)]
        for replica_device in self.parallel_devices[1:]:
            replica_policy, _ = self._load_policy(device=replica_device)
            self.policy_replicas.append((replica_device, replica_policy))
        for _, replica_policy in self.policy_replicas:
            replica_policy.nets["policy"].diffusion.autocast_denoiser = self.mixed_precision
        # 远车无引导路径只在显式启用且确有裁剪时加载，默认不增加显存或加载时间。
        self.unguided_policy = None
        if self.far_agent_mode == "unguided_diffusion" and self.diffusion_agent_limit > 0:
            self.unguided_policy, _ = self._load_unguided_policy()
        checkpoint_history = int(self.exp_config.algo.history_num_frames) + 1
        checkpoint_horizon = int(self.exp_config.algo.horizon)
        self.step_time = float(self.exp_config.algo.step_time)
        if int(history_frames) != checkpoint_history:
            raise ValueError(
                f"history_frames={history_frames} does not match checkpoint requirement {checkpoint_history}"
            )
        if self.prediction_horizon != checkpoint_horizon:
            raise ValueError(
                f"prediction_horizon={self.prediction_horizon} does not match checkpoint horizon {checkpoint_horizon}"
            )

        raster_size = int(self.exp_config.env.rasterizer.raster_size)
        pixel_size = float(self.exp_config.env.rasterizer.pixel_size)
        self.adapter = SafeSimBatchAdapter(
            history_frames=checkpoint_history,
            max_neighbors=max_neighbors,
            raster_size=raster_size,
            pixel_size=pixel_size,
            raster_center=(0.25, 0.5),
        )

    def _validate_paths(self):
        required = {
            "safe_sim_root": self.safe_sim_root,
            "config_path": self.config_path,
            "checkpoint_path": self.checkpoint_path,
        }
        missing = {name: str(path) for name, path in required.items() if not path.exists()}
        if missing:
            raise FileNotFoundError(f"Safe-Sim integration paths are missing: {missing}")
        if not (self.safe_sim_root / "tbsim").is_dir():
            raise FileNotFoundError(f"tbsim package not found under {self.safe_sim_root}")

    def _install_safe_sim_import_paths(self):
        import_paths = [self.safe_sim_root, self.safe_sim_root / "trajdata" / "src"]
        for path in reversed(import_paths):
            path_string = str(path)
            if path.exists() and path_string not in sys.path:
                sys.path.insert(0, path_string)

    def _resolve_parallel_devices(self, configured_devices):
        """解析可选多卡设备，空配置时严格保留既有单卡路径。"""
        if configured_devices is None:
            return [self.device]
        requested = [str(item) for item in configured_devices]
        if not requested:
            return [self.device]
        devices = [torch.device(item) for item in requested]
        if any(device.type != "cuda" for device in devices):
            raise ValueError("multi_gpu_devices must contain CUDA devices only")
        if any(device.index is None or device.index >= torch.cuda.device_count() for device in devices):
            raise ValueError("multi_gpu_devices contains an unavailable CUDA device")
        if len(set(str(device) for device in devices)) != len(devices):
            raise ValueError("multi_gpu_devices must not contain duplicates")
        if self.device not in devices:
            devices.insert(0, self.device)
        return devices

    def _load_policy(self, device=None):
        from tbsim.algos.algos import DiffusionTrafficModel
        from tbsim.configs.config import Dict
        from tbsim.configs.guidance_config import GuidanceConfig
        from tbsim.utils.batch_utils import batch_utils, set_global_batch_type
        from tbsim.utils.config_utils import get_experiment_config_from_file
        from policies.llm_anchor_guidance import register_llm_anchor_guidance
        from policies.scenario_guidance import register_scenario_guidance

        set_global_batch_type("trajdata")
        register_llm_anchor_guidance()
        register_scenario_guidance()
        exp_config = get_experiment_config_from_file(str(self.config_path))
        exp_config.unlock()
        guidance = GuidanceConfig()
        guidance.update_params(
            {
                "num_samples": self.num_samples,
                "sample_step": self.sample_step,
                "sampling_mode": exp_config.algo.diffuse_config.sampling_mode,
            }
        )
        profile = self._resolve_guidance_profile()
        self.active_guidance_functions = list(profile["functions"])
        self.guidance_enabled = bool(self.active_guidance_functions)
        self.uses_anchor_guidance = "llm_anchor" in self.active_guidance_functions
        if self.guidance_enabled:
            guidance.set_guidance_fn(self.active_guidance_functions)
            for name, config in profile["configs"].items():
                if name in self.active_guidance_functions:
                    guidance.update_config(name, config)
            weights = [float(value) for value in profile["weights"]]
            guidance.update_combine_loss(
                {
                    "filter_criterion": "combined",
                    "ctrl_filter_criterion": "combined",
                    "weights": weights,
                    "ctrl_weights": weights,
                }
            )
            params = {
                "guidance_horizon": self.prediction_horizon,
                "inner_lr": self.anchor_inner_lr,
                "inner_beta": self.anchor_max_update,
                "n_guide_steps": self.anchor_guide_steps,
                "scale_grad_by_std": self.anchor_scale_grad_by_std,
                "grad_wrt": "clean_guide",
            }
            params.update(self.guidance_config.get("params", {}))
            guidance.update_params(params)
        guidance_config = guidance.to_dict()
        guidance_config["params"]["num_samples"] = self.num_samples
        guidance_config["params"]["sample_step"] = self.sample_step
        guidance_config["params"]["sampling_mode"] = exp_config.algo.diffuse_config.sampling_mode
        exp_config.algo.guide_config = Dict(guidance_config)
        exp_config.algo.do_guidance = self.guidance_enabled
        exp_config.lock()

        modality_shapes = batch_utils().get_modality_shapes(exp_config)
        self.modality_shapes = modality_shapes
        load_device = self.device if device is None else torch.device(device)
        policy = DiffusionTrafficModel.load_from_checkpoint(
            str(self.checkpoint_path),
            algo_config=exp_config.algo,
            modality_shapes=modality_shapes,
            map_location=load_device,
        ).to(load_device).eval()
        return policy, exp_config

    def _load_unguided_policy(self, device=None):
        """临时复用配置加载无引导副本，不改变主引导模型和默认路径。"""
        original_mode = self.guidance_mode
        original_functions = self.active_guidance_functions
        original_guidance_enabled = self.guidance_enabled
        original_anchor_enabled = self.uses_anchor_guidance
        self.guidance_mode = "unguided"
        try:
            return self._load_policy(device=device)
        finally:
            self.guidance_mode = original_mode
            self.active_guidance_functions = original_functions
            self.guidance_enabled = original_guidance_enabled
            self.uses_anchor_guidance = original_anchor_enabled

    def _resolve_guidance_profile(self):
        """将四种启动模式解析为 Safe-Sim 的损失列表、权重和参数。"""
        common_configs = _to_plain_value(self.guidance_config.get("loss_configs", {}))
        common_configs.setdefault(
            "llm_anchor",
            {
                "loss_timesteps": self.prediction_horizon,
                "filter_timesteps": self.prediction_horizon,
                "loss_scale": self.anchor_guidance_strength,
                "robust_delta": self.anchor_robust_delta,
            },
        )
        if self.guidance_mode == "unguided":
            return {"functions": [], "weights": [], "configs": common_configs}

        if self.guidance_mode == "manual":
            manual = _to_plain_value(self.guidance_config.get("manual", {}))
            functions = list(manual.get("functions", []))
            weights = list(manual.get("weights", [1.0] * len(functions)))
            manual_configs = manual.get("configs", {})
            common_configs.update(manual_configs)
        else:
            functions = ["scenario_collision", "route", "scenario_ttc"]
            if self.guidance_mode == "llm_joint":
                functions.append("llm_anchor")
            profile = _to_plain_value(self.guidance_config.get(self.guidance_mode, {}))
            weights = list(profile.get("weights", [1.0] * len(functions)))
            common_configs.update(profile.get("configs", {}))

        if not functions:
            raise ValueError("manual guidance mode requires at least one guidance function")
        if len(weights) != len(functions):
            raise ValueError(
                f"guidance weights count {len(weights)} does not match functions count {len(functions)}"
            )
        return {"functions": functions, "weights": weights, "configs": common_configs}

    def update_iterative_guidance(self, params, weights):
        """原地更新已加载的 llm_joint 引导配置，供闭环攻击阶段使用。"""
        if self.guidance_mode != "llm_joint" or len(weights) != len(self.active_guidance_functions):
            return False
        for _, policy in self.policy_replicas:
            net = policy.nets["policy"]
            net.guide_config.params.update(dict(params))
            device = next(net.parameters()).device
            net.Loss_Calculater.weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
            net.Loss_Calculater.ctrl_weights = net.Loss_Calculater.weights.clone()
        return True

    @staticmethod
    def _find_loss_calculator(root, name):
        """在单损失或组合损失中查找指定计算器。"""
        if getattr(root, "name", None) == name:
            return root
        calculators = getattr(root, "loss_calculator_dict", {})
        return calculators.get(name)

    def _select_adversarial_target_id(
        self, frame, safe_batch, anchor_metadata, attack_intent=None
    ):
        """选择 TTC 对抗目标；LLM 联合模式严格服从大模型的攻击决定。"""
        if self.guidance_mode == "llm_joint" and attack_intent is not None:
            target_id = int(attack_intent["target_id"])
            if target_id in safe_batch.row_to_agent_id:
                return target_id
        if anchor_metadata.get("active"):
            return int(anchor_metadata["target_id"])

        # LLM 未攻击、锚点过期或目标不再活跃时，不得用最近车辆补造攻击目标。
        if self.guidance_mode == "llm_joint":
            return None
        if "scenario_ttc" not in self.active_guidance_functions:
            return None
        if frame.ego_state_global is None or len(safe_batch.row_to_agent_id) == 0:
            return None

        positions = frame.states_global[safe_batch.row_to_agent_id, :2]
        distances = torch.from_numpy(
            ((positions - frame.ego_state_global[None, :2]) ** 2).sum(axis=-1)
        )
        return int(safe_batch.row_to_agent_id[int(torch.argmin(distances))])

    def reset(self, scene_id):
        """重置场景相关状态，并清空本场景的可选性能诊断记录。"""
        self.scene_id = str(scene_id)
        self.performance_records.clear()

    def _perf_start(self):
        for device, _ in self.policy_replicas:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        return time.perf_counter()

    def _perf_stop(self, start_time):
        for device, _ in self.policy_replicas:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        return time.perf_counter() - start_time

    def _select_diffusion_agent_ids(self, frame, attack_intent):
        """优先保留攻击车和自车邻近车辆，其余车辆采用恒速回退轨迹。"""
        active_ids = frame.agent_ids[frame.active_mask].astype("int64", copy=False)
        limit = self.diffusion_agent_limit
        if limit == 0 or limit >= len(active_ids):
            return active_ids
        if frame.ego_state_global is None:
            return active_ids[:limit]
        distances = ((
            frame.states_global[active_ids, :2] - frame.ego_state_global[None, :2]
        ) ** 2).sum(axis=-1)
        priority = []
        if attack_intent is not None:
            target_id = int(attack_intent["target_id"])
            if target_id in active_ids:
                priority.append(target_id)
        for index in distances.argsort().tolist():
            agent_id = int(active_ids[index])
            if agent_id not in priority:
                priority.append(agent_id)
            if len(priority) >= limit:
                break
        selected = set(priority[:limit])
        # 保持稳定编号顺序，确保解码、可视化与联合筛选的行含义不变。
        return active_ids[[agent_id in selected for agent_id in active_ids]]

    def _merge_constant_velocity_agents(self, frame, joint, active_ids):
        """补全未扩散车辆的恒速预测，维持仿真器的全参与者联合轨迹契约。"""
        if np.array_equal(joint.agent_ids, active_ids):
            joint.metadata["diffusion_agent_count"] = int(len(active_ids))
            joint.metadata["constant_velocity_agent_count"] = 0
            return joint
        horizon = joint.positions_global.shape[1]
        positions = np.zeros((len(active_ids), horizon, 2), dtype=np.float32)
        velocities = np.zeros_like(positions)
        yaws = np.zeros((len(active_ids), horizon, 1), dtype=np.float32)
        valid_mask = np.ones((len(active_ids), horizon), dtype=bool)
        selected_row = {int(agent_id): row for row, agent_id in enumerate(joint.agent_ids)}
        time_offsets = (
            np.arange(1, horizon + 1, dtype=np.float32) * float(frame.dt)
        )
        for row, agent_id in enumerate(active_ids):
            source_row = selected_row.get(int(agent_id))
            if source_row is not None:
                positions[row] = joint.positions_global[source_row]
                velocities[row] = joint.velocities_global[source_row]
                yaws[row] = joint.yaws_global[source_row]
                valid_mask[row] = joint.valid_mask[source_row]
                continue
            state = frame.states_global[agent_id]
            velocity = state[2:4].astype(np.float32, copy=False)
            positions[row] = state[None, :2] + time_offsets[:, None] * velocity[None, :]
            velocities[row] = velocity[None, :]
            yaws[row, :, 0] = float(state[4])
        metadata = dict(joint.metadata)
        metadata.update({
            "diffusion_agent_count": int(len(joint.agent_ids)),
            "constant_velocity_agent_count": int(len(active_ids) - len(joint.agent_ids)),
        })
        return JointTrajectory(
            source_step=joint.source_step,
            agent_ids=active_ids.copy(),
            positions_global=positions,
            yaws_global=yaws,
            velocities_global=velocities,
            valid_mask=valid_mask,
            metadata=metadata,
        )

    def _combine_joint_trajectories(self, frame, joints, active_ids):
        """按稳定车辆编号合并引导与无引导预测，保持原有全参与者轨迹接口。"""
        if not joints:
            raise ValueError("at least one joint trajectory is required")
        horizon = joints[0].positions_global.shape[1]
        positions = np.zeros((len(active_ids), horizon, 2), dtype=np.float32)
        velocities = np.zeros_like(positions)
        yaws = np.zeros((len(active_ids), horizon, 1), dtype=np.float32)
        valid_mask = np.zeros((len(active_ids), horizon), dtype=bool)
        rows = {int(agent_id): index for index, agent_id in enumerate(active_ids)}
        for joint in joints:
            if joint.positions_global.shape[1] != horizon:
                raise ValueError("joint trajectories must share a prediction horizon")
            for source_row, agent_id in enumerate(joint.agent_ids):
                target_row = rows.get(int(agent_id))
                if target_row is None:
                    raise ValueError("joint trajectory contains an inactive participant")
                positions[target_row] = joint.positions_global[source_row]
                velocities[target_row] = joint.velocities_global[source_row]
                yaws[target_row] = joint.yaws_global[source_row]
                valid_mask[target_row] = joint.valid_mask[source_row]
        if not valid_mask.all():
            raise RuntimeError("guided and unguided trajectories did not cover every active participant")
        metadata = dict(joints[0].metadata)
        metadata.update({
            "diffusion_agent_count": int(len(joints[0].agent_ids)),
            "unguided_diffusion_agent_count": int(sum(len(joint.agent_ids) for joint in joints[1:])),
            "constant_velocity_agent_count": 0,
        })
        return JointTrajectory(
            source_step=frame.step,
            agent_ids=active_ids.copy(),
            positions_global=positions,
            yaws_global=yaws,
            velocities_global=velocities,
            valid_mask=valid_mask,
            metadata=metadata,
        )

    def _configure_anchor_loss(self, policy_net, anchor_tensors, anchor_active, device):
        """在每个模型副本上设置本分片的 LLM 锚点，并返回待清理的损失对象。"""
        if not self.uses_anchor_guidance:
            return None
        anchor_loss = self._find_loss_calculator(policy_net.Loss_Calculater, "llm_anchor")
        if anchor_loss is None:
            raise RuntimeError("llm_anchor guidance is configured but its calculator is missing")
        if anchor_active:
            tensors = _move_to_device(anchor_tensors, device)
            anchor_loss.set_anchor_targets(
                tensors["llm_anchor_positions_local"],
                tensors["llm_anchor_mask"],
                self.num_samples,
            )
        else:
            anchor_loss.clear_anchor_targets()
        return anchor_loss

    def _sample_policy(self, policy, device, model_batch, anchor_tensors, anchor_active):
        """在指定 GPU 上执行一块车辆批次，允许 FP16 作为可选性能实验。"""
        local_batch = _move_to_device(model_batch, device)
        policy_net = policy.nets["policy"]
        anchor_loss = self._configure_anchor_loss(
            policy_net, anchor_tensors, anchor_active, device
        )
        inference_context = torch.enable_grad() if self.guidance_enabled else torch.inference_mode()
        try:
            with inference_context:
                return policy.get_action(local_batch, sample=True)
        finally:
            if anchor_loss is not None:
                anchor_loss.clear_anchor_targets()

    @staticmethod
    def _sample_unguided_policy(policy, device, model_batch):
        """远车只执行无梯度扩散采样，不计算任何对抗或锚点损失。"""
        local_batch = _move_to_device(model_batch, device)
        with torch.inference_mode():
            return policy.get_action(local_batch, sample=True)

    def _parallel_get_action(self, model_batch, anchor_tensors, anchor_active):
        """按车辆维分片到多张 GPU；最终仍由主卡作全局候选样本筛选。"""
        batch_size = int(model_batch["image"].shape[0])
        shard_count = min(len(self.policy_replicas), batch_size)
        indices = torch.arange(batch_size, dtype=torch.long)
        shards = [chunk for chunk in torch.tensor_split(indices, shard_count) if len(chunk)]

        def run_shard(replica, shard_indices):
            device, policy = replica
            shard_batch = _slice_batch(model_batch, shard_indices, batch_size)
            shard_anchors = (
                _slice_batch(anchor_tensors, shard_indices, batch_size)
                if anchor_tensors is not None
                else None
            )
            return self._sample_policy(
                policy, device, shard_batch, shard_anchors, anchor_active
            )

        with ThreadPoolExecutor(max_workers=shard_count) as executor:
            futures = [
                executor.submit(run_shard, replica, shard)
                for replica, shard in zip(self.policy_replicas, shards)
            ]
            results = [future.result() for future in futures]
        actions, infos = zip(*results)
        positions = torch.cat(
            [action.positions.to(self.device) for action in actions], dim=0
        )
        yaws = torch.cat([action.yaws.to(self.device) for action in actions], dim=0)
        merged_action = type(actions[0])(positions=positions, yaws=yaws)
        sample_positions = []
        sample_yaws = []
        for info in infos:
            samples = info.get("action_samples", {})
            if "positions" not in samples or "yaws" not in samples:
                return merged_action, {}
            sample_positions.append(samples["positions"].to(self.device))
            sample_yaws.append(samples["yaws"].to(self.device))
        return merged_action, {
            "action_samples": {
                "positions": torch.cat(sample_positions, dim=0),
                "yaws": torch.cat(sample_yaws, dim=0),
            }
        }

    def _select_joint_guided_action(self, policy_net, model_batch, action, info):
        """使用全场联合损失为所有交通参与者选择同一个扩散样本。"""
        samples = info.get("action_samples")
        if not self.guidance_enabled or not isinstance(samples, dict):
            return action, None
        positions = samples.get("positions")
        yaws = samples.get("yaws")
        if positions is None or yaws is None or positions.ndim != 4:
            return action, None
        batch_size, num_samples, horizon = positions.shape[:3]
        if batch_size < 2 or num_samples < 2:
            return action, None

        # Safe-Sim 的第 0 个样本是逐车独立过滤后的拼接结果，不具有联合一致性。
        state = torch.cat(
            [
                positions,
                torch.zeros_like(positions[..., :1]),
                yaws[..., :1],
            ],
            dim=-1,
        ).reshape(batch_size * num_samples, horizon, 4)
        dummy_action = torch.zeros_like(state[..., :2])
        with torch.no_grad():
            guidance_data = policy_net._prepare_guidance_data(model_batch)
            losses = policy_net.Loss_Calculater.calculate_loss(
                dummy_action,
                state,
                guidance_data,
            ).reshape(batch_size, num_samples, horizon)
            joint_scores = losses.sum(dim=(0, 2))
            # 全量迭代模式：在原联合损失之上叠加轻量画像匹配，不改变默认样本选择。
            intent = model_batch.get("full_adversarial_method")
            if intent is not None and bool(intent):
                profile = model_batch.get("adversarial_profile", {})
                target_rows = torch.nonzero(model_batch["guidance_target_mask"] > 0, as_tuple=False).flatten()
                if len(target_rows):
                    traj = positions[int(target_rows[0]), :, :, :2]
                    lateral = traj[:, :, 0].sub(traj[:, :1, 0]).abs().mean(dim=1)
                    forward = traj[:, :, 1].sub(traj[:, :1, 1]).abs().mean(dim=1)
                    exploit = profile.get("preferred_exploit") if isinstance(profile, dict) else None
                    match = -lateral if exploit == "forward_pressure" else (lateral if exploit == "cut_in_or_lateral_conflict" else forward)
                    joint_scores = joint_scores - 0.05 * match
            joint_scores[0] = torch.inf
            selected_index = int(torch.argmin(joint_scores).item())

        selected_action = type(action)(
            positions=positions[:, selected_index],
            yaws=yaws[:, selected_index],
        )
        return selected_action, selected_index

    def predict(self, frame, attack_intent=None):
        if not isinstance(frame, ScenarioFrame):
            raise TypeError("frame must be a ScenarioFrame")
        if self.scene_id is None:
            raise RuntimeError("DiffusionModelWrapper.reset(scene_id) must be called before predict")
        if str(frame.scene_id) != self.scene_id:
            raise RuntimeError(
                f"received frame for scene {frame.scene_id!r}, but controller is reset for {self.scene_id!r}"
            )
        if abs(float(frame.dt) - self.step_time) > 1e-6:
            raise ValueError(
                f"Scenario Dreamer dt={frame.dt} does not match Safe-Sim checkpoint step_time={self.step_time}"
            )

        perf = {} if self.performance_diagnostics else None
        if perf is not None:
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            input_start = self._perf_start()
        active_ids = frame.agent_ids[frame.active_mask].astype(np.int64, copy=False)
        controlled_ids = self._select_diffusion_agent_ids(frame, attack_intent)
        safe_batch = self.adapter.build(frame, controlled_agent_ids=controlled_ids)
        if len(safe_batch.row_to_agent_id) == 0:
            return JointTrajectory.empty(frame.step)
        far_ids = active_ids[~np.isin(active_ids, controlled_ids)]
        far_safe_batch = None
        if self.far_agent_mode == "unguided_diffusion" and len(far_ids):
            far_safe_batch = self.adapter.build(frame, controlled_agent_ids=far_ids)

        anchor_metadata = {"active": False, "target_id": None, "valid_steps": 0}
        if self.uses_anchor_guidance:
            anchor_tensors, anchor_metadata = self.adapter.build_anchor_guidance(
                frame=frame,
                safe_batch=safe_batch,
                attack_intent=attack_intent,
                prediction_horizon=self.prediction_horizon,
                anchor_interval_seconds=self.anchor_interval_seconds,
            )

        model_batch = dict(safe_batch.data)
        target_mask = torch.zeros((len(safe_batch.row_to_agent_id),), dtype=torch.float32)
        target_id = self._select_adversarial_target_id(
            frame, safe_batch, anchor_metadata, attack_intent=attack_intent
        )
        if target_id is not None:
            rows = torch.from_numpy(safe_batch.row_to_agent_id == target_id)
            target_mask[rows] = 1.0
        model_batch["guidance_target_mask"] = target_mask
        attack_active = torch.zeros_like(target_mask)
        attack_age_frames = 0
        if self.guidance_mode == "llm_joint" and attack_intent is not None:
            source_step = int(attack_intent["source_step"])
            if source_step > frame.step:
                raise ValueError("attack intent source_step cannot be later than the current frame")
            attack_active.copy_(target_mask)
            attack_age_frames = int(frame.step - source_step)
        # 运行时攻击阶段仅通过可选张量传入，不改变无攻击或无引导采样路径。
        model_batch["guidance_attack_active"] = attack_active
        # LLM 指定的攻击车允许离开道路；其他交通参与者仍受 route 损失约束。
        model_batch["guidance_route_exempt_mask"] = attack_active
        model_batch["guidance_attack_age_frames"] = torch.full(
            (len(safe_batch.row_to_agent_id),),
            attack_age_frames,
            dtype=torch.int64,
        )
        model_batch_cpu = model_batch
        model_batch = _move_to_device(model_batch_cpu, self.device)
        expected_image_shape = (len(safe_batch.row_to_agent_id), *self.modality_shapes["image"])
        if tuple(model_batch["image"].shape) != expected_image_shape:
            raise ValueError(
                f"Safe-Sim image batch must be {expected_image_shape}, got {tuple(model_batch['image'].shape)}"
            )
        if perf is not None:
            perf["input_preparation_s"] = self._perf_stop(input_start)
        policy_net = self.policy.nets["policy"]
        if perf is not None:
            sampling_start = self._perf_start()
        if len(self.policy_replicas) > 1 and len(safe_batch.row_to_agent_id) > 1:
            action, info = self._parallel_get_action(
                model_batch_cpu,
                anchor_tensors if self.uses_anchor_guidance else None,
                anchor_metadata["active"],
            )
        else:
            action, info = self._sample_policy(
                self.policy,
                self.device,
                model_batch_cpu,
                anchor_tensors if self.uses_anchor_guidance else None,
                anchor_metadata["active"],
            )
        far_action = None
        if far_safe_batch is not None:
            far_action, _ = self._sample_unguided_policy(
                self.unguided_policy,
                self.device,
                dict(far_safe_batch.data),
            )
        if perf is not None:
            perf["model_sampling_s"] = self._perf_stop(sampling_start)
            selection_start = self._perf_start()
        action, joint_sample_index = self._select_joint_guided_action(
            policy_net,
            model_batch,
            action,
            info,
        )
        if perf is not None:
            perf["joint_selection_s"] = self._perf_stop(selection_start)
        if perf is not None:
            decode_start = self._perf_start()
        positions_local = action.positions.detach().cpu().numpy()
        yaws_local = action.yaws.detach().cpu().numpy()
        joint = self.adapter.decode(frame, safe_batch, positions_local, yaws_local)
        candidate_trajectories_global = []
        # 将攻击目标的每个扩散候选样本解码到全局坐标，供可视化显示；不参与控制决策。
        samples = info.get("action_samples") if isinstance(info, dict) else None
        if (
            self.capture_candidate_trajectories
            and target_id is not None
            and isinstance(samples, dict)
            and samples.get("positions") is not None
            and samples.get("yaws") is not None
        ):
            sample_positions = samples["positions"]
            sample_yaws = samples["yaws"]
            if sample_positions.ndim == 4 and sample_positions.shape[0] == len(safe_batch.row_to_agent_id):
                for sample_index in range(sample_positions.shape[1]):
                    candidate_joint = self.adapter.decode(
                        frame,
                        safe_batch,
                        sample_positions[:, sample_index].detach().cpu().numpy(),
                        sample_yaws[:, sample_index].detach().cpu().numpy(),
                    )
                    candidate = candidate_joint.trajectory_for(target_id)
                    if candidate is not None:
                        candidate_trajectories_global.append(candidate[:, :2])
        if far_safe_batch is not None:
            far_joint = self.adapter.decode(
                frame,
                far_safe_batch,
                far_action.positions.detach().cpu().numpy(),
                far_action.yaws.detach().cpu().numpy(),
            )
            joint = self._combine_joint_trajectories(
                frame, [joint, far_joint], active_ids
            )
        else:
            joint = self._merge_constant_velocity_agents(frame, joint, active_ids)
        if perf is not None:
            perf["decode_s"] = self._perf_stop(decode_start)
        joint.metadata.update(
            {
                "checkpoint": self.checkpoint_path.name,
                "num_samples": self.num_samples,
                "sample_step": self.sample_step,
                "all_samples_available": "action_samples" in info,
                "joint_sample_index": joint_sample_index,
                "candidate_trajectories_global": candidate_trajectories_global,
                "guidance_mode": self.guidance_mode,
                "guidance_enabled": self.guidance_enabled,
                "guidance_functions": list(self.active_guidance_functions),
                "adversarial_guidance_active": target_id is not None,
                "adversarial_guidance_target_id": target_id,
                "attack_age_frames": attack_age_frames,
                "anchor_guidance_enabled": self.uses_anchor_guidance,
                "anchor_guidance_active": anchor_metadata["active"],
                "anchor_guidance_target_id": anchor_metadata["target_id"],
                "anchor_guidance_valid_steps": anchor_metadata["valid_steps"],
                "anchor_guidance_strength": self.anchor_guidance_strength,
                "mixed_precision": self.mixed_precision,
                "multi_gpu_devices": [str(device) for device, _ in self.policy_replicas],
                "far_agent_mode": self.far_agent_mode,
            }
        )
        if perf is not None:
            perf["peak_memory_bytes"] = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0
            self.performance_records.append(perf)
            print("[safe-sim-perf] " + " ".join(f"{key}={value:.6f}" for key, value in perf.items()))
        return joint


class SafeSimDiffusionController(DiffusionModelWrapper):
    """供 Simulator 配置和诊断使用的领域专用名称。"""
    pass
