"""在 Scenario Dreamer 仿真器中使用 Safe-Sim 的生命周期封装。"""

import sys
from collections.abc import Mapping
from pathlib import Path

import torch

from policies.safe_sim_adapter import SafeSimBatchAdapter
from policies.traffic_types import JointTrajectory, ScenarioFrame


def _move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
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
        guidance_config=None,
        anchor_guidance_enabled=False,
        anchor_guidance_strength=0.2,
        anchor_interval_seconds=1.0,
        anchor_robust_delta=1.0,
        anchor_inner_lr=0.2,
        anchor_max_update=0.5,
        anchor_guide_steps=1,
        anchor_scale_grad_by_std=True,
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

        self.policy, self.exp_config = self._load_policy()
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

    def _load_policy(self):
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
        policy = DiffusionTrafficModel.load_from_checkpoint(
            str(self.checkpoint_path),
            algo_config=exp_config.algo,
            modality_shapes=modality_shapes,
            map_location=self.device,
        ).to(self.device).eval()
        return policy, exp_config

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

    @staticmethod
    def _find_loss_calculator(root, name):
        """在单损失或组合损失中查找指定计算器。"""
        if getattr(root, "name", None) == name:
            return root
        calculators = getattr(root, "loss_calculator_dict", {})
        return calculators.get(name)

    def _select_adversarial_target_id(self, frame, safe_batch, anchor_metadata):
        """选择 TTC 对抗目标；LLM 联合模式严格服从大模型的攻击决定。"""
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
        self.scene_id = str(scene_id)

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

        safe_batch = self.adapter.build(frame)
        if len(safe_batch.row_to_agent_id) == 0:
            return JointTrajectory.empty(frame.step)

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
            frame, safe_batch, anchor_metadata
        )
        if target_id is not None:
            rows = torch.from_numpy(safe_batch.row_to_agent_id == target_id)
            target_mask[rows] = 1.0
        model_batch["guidance_target_mask"] = target_mask
        model_batch = _move_to_device(model_batch, self.device)
        expected_image_shape = (len(safe_batch.row_to_agent_id), *self.modality_shapes["image"])
        if tuple(model_batch["image"].shape) != expected_image_shape:
            raise ValueError(
                f"Safe-Sim image batch must be {expected_image_shape}, got {tuple(model_batch['image'].shape)}"
            )
        inference_context = torch.enable_grad() if self.guidance_enabled else torch.inference_mode()
        policy_net = self.policy.nets["policy"]
        anchor_loss = None
        if self.uses_anchor_guidance:
            anchor_loss = self._find_loss_calculator(policy_net.Loss_Calculater, "llm_anchor")
            if anchor_loss is None:
                raise RuntimeError("llm_anchor guidance is configured but its calculator is missing")
            if anchor_metadata["active"]:
                anchor_tensors = _move_to_device(anchor_tensors, self.device)
                anchor_loss.set_anchor_targets(
                    anchor_tensors["llm_anchor_positions_local"],
                    anchor_tensors["llm_anchor_mask"],
                    self.num_samples,
                )
            else:
                anchor_loss.clear_anchor_targets()
        try:
            with inference_context:
                action, info = self.policy.get_action(model_batch, sample=True)
        finally:
            if anchor_loss is not None:
                anchor_loss.clear_anchor_targets()
        positions_local = action.positions.detach().cpu().numpy()
        yaws_local = action.yaws.detach().cpu().numpy()
        joint = self.adapter.decode(frame, safe_batch, positions_local, yaws_local)
        joint.metadata.update(
            {
                "checkpoint": self.checkpoint_path.name,
                "num_samples": self.num_samples,
                "sample_step": self.sample_step,
                "all_samples_available": "action_samples" in info,
                "guidance_mode": self.guidance_mode,
                "guidance_enabled": self.guidance_enabled,
                "guidance_functions": list(self.active_guidance_functions),
                "adversarial_guidance_active": target_id is not None,
                "adversarial_guidance_target_id": target_id,
                "anchor_guidance_enabled": self.uses_anchor_guidance,
                "anchor_guidance_active": anchor_metadata["active"],
                "anchor_guidance_target_id": anchor_metadata["target_id"],
                "anchor_guidance_valid_steps": anchor_metadata["valid_steps"],
                "anchor_guidance_strength": self.anchor_guidance_strength,
            }
        )
        return joint


class SafeSimDiffusionController(DiffusionModelWrapper):
    """供 Simulator 配置和诊断使用的领域专用名称。"""
    pass
