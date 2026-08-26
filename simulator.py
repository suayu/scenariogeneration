import os
import pickle
import json
import copy
import numpy as np
import torch
import torch.nn.functional as F

from datasets.waymo.dataset_ctrl_sim import CtRLSimDataset
from utils.gpudrive_helpers import (
    get_action_value_tensor,
    get_ego_state,
    get_partner_obs,
    get_map_obs,
    get_route_obs,
    from_json_Map,
    ForwardKinematics
)
from utils.sim_helpers import (
    ego_completed_route,
    ego_collided,
    ego_off_route,
    ego_progress,
    normalize_route
)
from utils.geometry import normalize_agents
from utils.lane_graph_helpers import resample_lanes_with_mask
from utils.k_disks_helpers import inverse_k_disks, forward_k_disks
from utils.collision_helpers import compute_collision_states_one_scene
from utils.metrics_helpers import compute_sim_agent_jsd_metrics
from utils.torch_helpers import from_numpy
from utils.data_container import CtRLSimData
from utils.data_helpers import add_batch_dim, modify_agent_states
from utils.data_analyse import analyse_json, analyse_file
from utils.viz import render_state
from utils.llm_scene_viz import render_llm_scene_png
from models.ctrl_sim import CtRLSim
from policies.diffusion_model_wrapper import SafeSimDiffusionController
from policies.llm_adversarial_planner import LLMAdversarialPlanner
from policies.obstacle_wrapper import ObstacleWrapper, ScenarioDreamerObstacleBackend
from policies.traffic_types import JointTrajectory, ScenarioFrame

MAX_RTG_VAL = 349


class Simulator:
    """ We implement our own simple simulator for testing planners.
        
    This makes it easier to integrate with the CtRL-Sim behaviour model.
    Three modes are supported:
    - scenario_dreamer: Scenario Dreamer simulation environments with reactive CtRL-Sim agents
    - waymo_ctrl_sim: Waymo Open Dataset simulation environments with reactive CtRL-Sim agents
    - waymo_log_replay: Waymo Open Dataset simulation environments with log-replay agents.
    """
    def __init__(self, cfg):
        """ Initialize simulator."""
        self.cfg = cfg
        self.mode = self.cfg.sim.mode
        self.steps = self.cfg.sim.steps 
        self.dt = self.cfg.sim.dt 
        self.dataset_path = self.cfg.sim.dataset_path
        self.json_path = self.cfg.sim.json_path
        self.test_files = [os.path.join(self.dataset_path, file) 
                           for file in os.listdir(self.dataset_path)]
        self.num_test_scenarios = len(self.test_files)

        self.ctrl_sim_dset = CtRLSimDataset(self.cfg.ctrl_sim.dataset, split_name='val')
        traffic_cfg = self.cfg.sim.get('traffic_model')
        if traffic_cfg is None:
            raise ValueError("cfg.sim.traffic_model is required; choose an explicit traffic backend")
        self.traffic_backend = str(traffic_cfg.backend)
        self.behaviour_model = None
        self.diffusion_controller = None
        if self.traffic_backend == 'safe_sim_diffusion':
            anchor_cfg = traffic_cfg.get('anchor_guidance')
            self.diffusion_controller = SafeSimDiffusionController(
                safe_sim_root=traffic_cfg.safe_sim_root,
                config_path=traffic_cfg.config_path,
                checkpoint_path=traffic_cfg.checkpoint_path,
                device=traffic_cfg.device,
                history_frames=traffic_cfg.history_frames,
                prediction_horizon=traffic_cfg.prediction_horizon,
                max_neighbors=traffic_cfg.max_neighbors,
                num_samples=traffic_cfg.num_samples,
                sample_step=traffic_cfg.sample_step,
                diffusion_agent_limit=traffic_cfg.get('diffusion_agent_limit', 0),
                far_agent_mode=traffic_cfg.get('far_agent_mode', 'guided'),
                mixed_precision=traffic_cfg.get('mixed_precision', False),
                multi_gpu_devices=traffic_cfg.get('multi_gpu_devices', []),
                guidance_config=traffic_cfg.get('guidance'),
                anchor_guidance_enabled=False if anchor_cfg is None else anchor_cfg.enabled,
                anchor_guidance_strength=0.2 if anchor_cfg is None else anchor_cfg.strength,
                anchor_interval_seconds=1.0 if anchor_cfg is None else anchor_cfg.anchor_interval_seconds,
                anchor_robust_delta=1.0 if anchor_cfg is None else anchor_cfg.robust_delta,
                anchor_inner_lr=0.2 if anchor_cfg is None else anchor_cfg.inner_lr,
                anchor_max_update=0.5 if anchor_cfg is None else anchor_cfg.max_update,
                anchor_guide_steps=1 if anchor_cfg is None else anchor_cfg.guide_steps,
                anchor_scale_grad_by_std=True if anchor_cfg is None else anchor_cfg.scale_grad_by_std,
            )
        elif self.traffic_backend == 'ctrl_sim':
            # 已废弃：CtRL-Sim 模型加载代码仅保留供历史参考，不再执行。
            # self.behaviour_model = CtRLSimBehaviourModel(
            #     mode=self.mode,
            #     model_path=self.cfg.sim.behaviour_model.model_path,
            #     model=CtRLSim.load_from_checkpoint(self.cfg.sim.behaviour_model.model_path).to('cuda'),
            #     dset=self.ctrl_sim_dset,
            #     use_rtg=self.cfg.sim.behaviour_model.use_rtg,
            #     predict_rtgs=self.cfg.sim.behaviour_model.predict_rtgs,
            #     action_temperature=self.cfg.sim.behaviour_model.action_temperature,
            #     tilt=self.cfg.sim.behaviour_model.tilt,
            #     steps=self.steps,
            # )
            raise RuntimeError("CtRL-Sim 后端已废弃，请使用 safe_sim_diffusion 后端")
        elif self.traffic_backend != 'log_replay':
            raise ValueError(f"Unsupported traffic backend: {self.traffic_backend}")

        # 两个开销较大的控制器均随 Simulator 初始化一次，场景切换时只重置内部状态。
        llm_cfg = self.cfg.sim.get('llm')
        llm_model_name = "qwen3-32b" if llm_cfg is None else str(llm_cfg.model_name)
        llm_model_names = None if llm_cfg is None else list(
            llm_cfg.get('model_names', [llm_model_name])
        )
        use_multimodal = False if llm_cfg is None else bool(llm_cfg.multimodal)
        attack_mode = "trajectory_only" if llm_cfg is None else str(
            llm_cfg.get('attack_mode', 'trajectory_only')
        )
        self.llm_planner = LLMAdversarialPlanner(
            model_name=llm_model_name,
            model_names=llm_model_names,
            use_multimodal=use_multimodal,
            attack_mode=attack_mode,
        )
        self.action_map = get_action_value_tensor()
        # tracks state of all objects during simulation
        self.data_dict = {}
        self.step_data = []

        # === 对抗轨迹控制属性 ===
        self.adversarial_agent_id = None
        self.adversarial_traj = None
        self.adversarial_step_idx = 0
        self.nearby_distance = 35 # 距离阈值：用于判断哪些路段/对象适合进行对抗性攻击

        self.activate_agent_ids = []  # 用于存储当前场景中活跃的交通参与者索引
        self.current_scene_id = None
        self.pending_joint_trajectory = None
        self.attack_intent = None
        # 静态障碍物只通过统一包装器访问，避免上层规划器依赖具体仿真器 API。
        self.static_obstacle_elements = {}
        self.obstacle_wrapper = ObstacleWrapper(ScenarioDreamerObstacleBackend(self))

    @property
    def current_step(self):
        """返回 Simulator 唯一可信的当前时间步。"""
        return self.t


    def load_initial_scene(self, i):
        """ Load initial configurations of scenario (map and initial state) given index."""
        # scenario in scenario dreamer format
        # 从二进制文件 (.pkl) 加载场景数据
        with open(os.path.join(self.dataset_path, self.test_files[i]), 'rb') as f:
            scenario_dict = pickle.load(f)

        # 如果使用 RL 策略，加载额外的地图信息
        if self.cfg.sim.policy == 'rl':
            # load additional map info from gpudrive json
            if self.cfg.sim.mode == 'scenario_dreamer':
                json_filename = f"{self.test_files[i].split('/')[-1][:-4]}.json"
            else:
                json_filename = f"{self.test_files[i].split('/')[-1][11:-4]}.json"
            json_path = os.path.join(self.json_path, json_filename)
            with open(json_path, 'r') as f:
                gpudrive_dict = json.load(f)
            
            # convert map to GPUDrive format for compatibility 
            # with RL planners trained in GPUDrive
            # 将地图转换为 GPUDrive 格式
            gpudrive_dict = from_json_Map(
                gpudrive_dict, 
                polylineReductionThreshold=self.cfg.sim.polyline_reduction_threshold
            )

            # MAP
            scenario_dict['lanes_compressed'] = gpudrive_dict['lanes_compressed']
            scenario_dict['world_mean'] = gpudrive_dict['world_mean']
        return scenario_dict
    

    def _find_invalid_new_agents(
            self, 
            next_states, 
            newly_added_agent_mask, 
            still_existing_agent_mask,
            dist_gap_s=5.0,
            heading_threshold=np.pi/6,
            dist_threshold=2.0):
        """ Find newly added agents that are invalid due to 
        being at edge of FOV and heading outwards. Such agents
        would immediately leave the scene again, so we remove them.
        Also remove newly added agents that violate time gap."""
        normalized_next_states = normalize_agents(
            next_states[:, None], 
            self.local_frame
        )
        lanes, lanes_mask = self.ctrl_sim_dset.get_normalized_lanes_in_fov(
            self.data_dict['lanes'], 
            self.local_frame
        )
        lanes_resampled = resample_lanes_with_mask(
            lanes, 
            lanes_mask, 
            num_points=100
        )
        dist_to_lanes = np.linalg.norm(
            normalized_next_states[:, None, :, :2] 
            - lanes_resampled[None], axis=-1
        ).min(2)
        closest_lane_idxs = np.argmin(dist_to_lanes, axis=-1)

        new_agent_idxs_to_remove = []
        newly_added_agent_idxs = np.where(newly_added_agent_mask)[0]
        for new_agent_idx in newly_added_agent_idxs:
            heading = normalized_next_states[new_agent_idx, 0, 4]
            if (np.abs(heading - np.pi/2) < heading_threshold
                and (normalized_next_states[new_agent_idx, 0, 1]
                     - self.cfg.ctrl_sim.dataset.fov) < dist_threshold):
                new_agent_idxs_to_remove.append(new_agent_idx)
                continue
            
            closest_lane = closest_lane_idxs[new_agent_idx]
            closest_lane_mask = closest_lane_idxs == closest_lane
            agent_in_same_lane_mask = np.logical_and(
                closest_lane_mask,
                still_existing_agent_mask
            )
            if not agent_in_same_lane_mask.sum():
                continue

            dist_to_agent_in_same_lane = np.linalg.norm(
                normalized_next_states[new_agent_idx, :, :2] 
                - normalized_next_states[agent_in_same_lane_mask][:, 0, :2], 
                axis=-1)
            closest_agent_idx = np.where(
                agent_in_same_lane_mask
            )[0][np.argmin(dist_to_agent_in_same_lane)]
            dist_gap = np.linalg.norm(
                normalized_next_states[closest_agent_idx, 0, 2:4]) * dist_gap_s
            dist_to_closest_agent = np.linalg.norm(
                normalized_next_states[new_agent_idx, 0, :2] 
                - normalized_next_states[closest_agent_idx, 0, :2])

            if dist_to_closest_agent < dist_gap:
                new_agent_idxs_to_remove.append(new_agent_idx)
        return new_agent_idxs_to_remove

    def get_scenario_frame(self, history_frames=None):
        """返回供 Safe-Sim 适配器使用的强类型全局状态快照。"""
        if history_frames is None:
            history_frames = int(self.cfg.sim.traffic_model.history_frames)
        current_states = np.asarray(self.data_dict['agent'][-1], dtype=np.float32)
        agent_count = current_states.shape[0]
        history_global = np.zeros((agent_count, history_frames, 8), dtype=np.float32)
        history_mask = np.zeros((agent_count, history_frames), dtype=bool)
        state_history = self.data_dict['agent'][-history_frames:]
        active_history = self.data_dict['agent_active_history'][-history_frames:]
        offset = history_frames - len(state_history)
        for history_index, (states, active) in enumerate(zip(state_history, active_history)):
            target_index = offset + history_index
            history_global[:, target_index] = np.asarray(states, dtype=np.float32)
            history_mask[:, target_index] = np.asarray(active, dtype=bool)

        obstacle_rows = [
            [
                float(obstacle["center"][0]),
                float(obstacle["center"][1]),
                float(obstacle["yaw"]),
                float(obstacle["length"]),
                float(obstacle["width"]),
            ]
            for obstacle in self.get_static_obstacles()
        ]
        # 静态障碍物沿用全局坐标，避免在仿真器与扩散适配器间重复转换。
        static_obstacles_global = (
            np.asarray(obstacle_rows, dtype=np.float32)
            if obstacle_rows
            else np.empty((0, 5), dtype=np.float32)
        )

        return ScenarioFrame(
            scene_id=str(self.current_scene_id),
            step=self.current_step,
            dt=float(self.dt),
            agent_ids=np.arange(agent_count, dtype=np.int64),
            states_global=current_states.copy(),
            agent_types=np.asarray(self.scenario_dict['agent_types'], dtype=np.float32).copy(),
            active_mask=np.asarray(self.agent_active, dtype=bool).copy(),
            history_global=history_global,
            history_mask=history_mask,
            lanes_global=np.asarray(self.scenario_dict['lanes'], dtype=np.float32).copy(),
            route_global=np.asarray(self.scenario_dict['route'], dtype=np.float32).copy(),
            ego_state_global=np.asarray(self.ego_state, dtype=np.float32).copy(),
            static_obstacles_global=static_obstacles_global,
        )

    def prepare_background_traffic(self):
        """在自车策略求值前生成并暂存所有非自车参与者的联合轨迹。"""
        if self.traffic_backend != 'safe_sim_diffusion':
            return None
        frame = self.get_scenario_frame(self.diffusion_controller.adapter.history_frames)
        interval = int(getattr(self.cfg.sim.traffic_model, 'replan_interval_steps', 1))
        if interval < 1:
            raise ValueError('replan_interval_steps must be at least one')
        attack_key = None
        if self.attack_intent is not None:
            attack_key = (
                int(self.attack_intent['target_id']),
                int(self.attack_intent['source_step']),
                str(self.attack_intent['strategy']),
            )
        cached = self._cached_joint_trajectory
        reuse_offset = 0 if cached is None else int(frame.step - cached.source_step)
        active_ids = np.flatnonzero(self.agent_active)
        if (
            cached is not None
            and 0 < reuse_offset < interval
            and reuse_offset < cached.positions_global.shape[1]
            and np.array_equal(cached.agent_ids, active_ids)
            and attack_key == self._cached_attack_key
        ):
            # 为当前仿真步创建视图，使消费逻辑仍然只执行轨迹的第一帧。
            metadata = dict(cached.metadata)
            metadata.update({
                'replan_reused': True,
                'replan_cache_offset': reuse_offset,
                'replan_interval_steps': interval,
            })
            joint_trajectory = JointTrajectory(
                source_step=frame.step,
                agent_ids=cached.agent_ids.copy(),
                positions_global=cached.positions_global[:, reuse_offset:].copy(),
                yaws_global=cached.yaws_global[:, reuse_offset:].copy(),
                velocities_global=cached.velocities_global[:, reuse_offset:].copy(),
                valid_mask=cached.valid_mask[:, reuse_offset:].copy(),
                metadata=metadata,
            )
        else:
            joint_trajectory = self.diffusion_controller.predict(
                frame,
                attack_intent=self.attack_intent,
            )
            joint_trajectory.metadata.update({
                'replan_reused': False,
                'replan_cache_offset': 0,
                'replan_interval_steps': interval,
            })
            self._cached_joint_trajectory = joint_trajectory
            self._cached_attack_key = attack_key
        self.inject_joint_trajectory(joint_trajectory)
        return joint_trajectory

    def inject_joint_trajectory(self, joint_trajectory):
        """暂存一次 Safe-Sim 联合预测，供紧接着的仿真步执行。"""
        if not isinstance(joint_trajectory, JointTrajectory):
            raise TypeError("joint_trajectory must be a JointTrajectory")
        if joint_trajectory.source_step != self.current_step:
            raise ValueError(
                f"joint trajectory source step {joint_trajectory.source_step} does not match {self.current_step}"
            )
        known_ids = set(range(len(self.data_dict['agent'][-1])))
        unknown_ids = set(joint_trajectory.agent_ids.tolist()) - known_ids
        if unknown_ids:
            raise ValueError(f"joint trajectory contains unknown stable agent IDs: {sorted(unknown_ids)}")
        expected_ids = set(np.flatnonzero(self.agent_active).tolist())
        actual_ids = set(joint_trajectory.agent_ids.tolist())
        if actual_ids != expected_ids:
            raise ValueError(
                "joint trajectory must contain every active non-ego participant exactly once; "
                f"missing={sorted(expected_ids - actual_ids)}, extra={sorted(actual_ids - expected_ids)}"
            )
        self.pending_joint_trajectory = joint_trajectory
        self.data_dict['joint_trajectory'] = joint_trajectory

    def _consume_joint_trajectory(self, source_step):
        joint = self.pending_joint_trajectory
        if joint is None:
            raise RuntimeError(
                "Safe-Sim diffusion backend requires prepare_background_traffic() before Simulator.step()"
            )
        if joint.source_step != source_step:
            raise RuntimeError(
                f"stale Safe-Sim prediction from step {joint.source_step}; expected {source_step}"
            )
        next_states = copy.deepcopy(self.data_dict['agent'][-1])
        for row, agent_id in enumerate(joint.agent_ids):
            if joint.positions_global.shape[1] == 0 or not joint.valid_mask[row, 0]:
                raise RuntimeError(f"Safe-Sim returned no valid first step for agent {agent_id}")
            next_states[agent_id, 0:2] = joint.positions_global[row, 0]
            next_states[agent_id, 2:4] = joint.velocities_global[row, 0]
            next_states[agent_id, 4] = joint.yaws_global[row, 0, 0]
        self.pending_joint_trajectory = None
        return next_states

    def set_attack_intent(self, target_id, anchors, strategy):
        """保存大模型攻击计划，供 Safe-Sim 锚点 guidance 在后续帧使用。"""
        self.attack_intent = {
            'target_id': int(target_id),
            'anchors': copy.deepcopy(anchors),
            'strategy': strategy,
            'source_step': self.current_step,
        }

    def clear_attack_intent(self):
        """清除当前攻击计划，使后续扩散推理恢复无锚点引导。"""
        self.attack_intent = None

    def apply_obstacle_plan(self, placements, max_groups=2):
        """将已转换为全局坐标的抽象障碍物计划交给统一包装器执行。"""
        # Simulator 只调用包装器，不能直接操作静态障碍物存储或具体后端。
        return self.obstacle_wrapper.create_plan(placements, max_groups=max_groups)

    def get_static_obstacles(self):
        """返回不包含具体后端句柄的静态障碍物描述。"""
        return self.obstacle_wrapper.public_state()

    def get_attack_target_prediction(self):
        if self.attack_intent is None or self.pending_joint_trajectory is None:
            return None
        return self.pending_joint_trajectory.trajectory_for(self.attack_intent['target_id'])
    

    def step(self, action, anchors, refined_traj):
        """ Step function for scenario dreamer environment."""
        source_step = self.t
        self.t += 1
        self.activate_agent_ids = []  # Reset the list of active agent IDs for this step
        
        old_ego_state = copy.deepcopy(self.ego_state)
        # if action not supplied, default to log-replay
        if action is not None:
            if self.cfg.sim.policy == 'rl':
                action = (
                        torch.nan_to_num(action, nan=0).long()
                    ).cpu()
                action = self.action_map[action].numpy()
                if len(action.shape) > 1:
                    action = action[0]
                
                self.ego_state = self.rl_kinematics_model.forward_kinematics(action)
            else:
                (next_x, 
                 next_y, 
                 next_theta, 
                 next_speed) = (action[0], 
                                action[1], 
                                action[2], 
                                action[3])
                agent_next_state = np.array(
                    [next_x, 
                     next_y, 
                     next_speed * np.cos(next_theta), 
                     next_speed * np.sin(next_theta), 
                     next_theta, 
                     self.ego_state[5], 
                     self.ego_state[6], 
                     self.ego_state[7]]
                )
                # 更新自车状态
                self.ego_state = agent_next_state
        else:
            self.ego_state = self.ego_trajectory[self.t]

        self.local_frame = {
            'center': self.ego_state[:2].copy(),
            'yaw': self.ego_state[4].copy()
        }

        if self.traffic_backend == 'ctrl_sim':
            # 仅旧版 CtRL-Sim 后端需要把自车连续状态反解为离散动作。
            inverse_ego_action = inverse_k_disks(old_ego_state, self.ego_state, self.ctrl_sim_dset.V)
        else:
            # Safe-Sim 不使用 CtRL-Sim 离散动作，保留占位值以兼容既有日志结构。
            inverse_ego_action = np.array(-1, dtype=np.int64)

        self.data_dict['ego_action'].append(inverse_ego_action)
        # 为兼容既有数据结构，继续保留自车 RTG 占位值。
        self.data_dict['ego_rtg'].append(np.array([MAX_RTG_VAL])[None, :])
        
        if self.traffic_backend == 'log_replay' or self.mode == 'waymo_log_replay':
            self.data_dict['agent_next_action'] = self.scenario_dict['actions'][:, self.t - 1]
            self.data_dict['agent_next_rtg'] = np.zeros(len(self.scenario_dict['agents']))
            next_states = forward_k_disks(
                states=self.data_dict['agent'][-1],
                actions=self.data_dict['agent_next_action'],
                vocab=self.ctrl_sim_dset.V,
                delta_t=self.dt,
                exists=self.agent_active,
            )
        elif self.traffic_backend == 'ctrl_sim':
            # 已废弃：以下 CtRL-Sim 背景交通推理与单车轨迹覆盖逻辑仅保留供历史参考，不再执行。
            # self.data_dict = self.behaviour_model.step(self.data_dict)
            # next_states = forward_k_disks(
            #     states=self.data_dict['agent'][-1],
            #     actions=self.data_dict['agent_next_action'],
            #     vocab=self.ctrl_sim_dset.V,
            #     delta_t=self.dt,
            #     exists=self.agent_active,
            # )
            # if self.adversarial_agent_id is not None and self.adversarial_traj is not None:
            #     if self.adversarial_step_idx < len(self.adversarial_traj):
            #         target_state = self.adversarial_traj[self.adversarial_step_idx]
            #         next_states[self.adversarial_agent_id, 0:5] = target_state[0:5]
            #         self.adversarial_step_idx += 1
            raise RuntimeError("CtRL-Sim 背景交通推理已废弃，请使用 safe_sim_diffusion 后端")
        else:
            next_states = self._consume_joint_trajectory(source_step)
            # 扩散模型没有 CtRL-Sim 离散动作，使用形状兼容的占位数组供既有日志读取。
            self.data_dict['agent_next_action'] = np.full(
                len(self.scenario_dict['agents']), -1, dtype=np.int64
            )
            self.data_dict['agent_next_rtg'] = np.zeros(len(self.scenario_dict['agents']))
        
        # update last active positions for active agents
        # TODO: is this really necessary? If an agent leaves, we never use its position again, right?
        self.last_active_agent_position[self.agent_active] = next_states[self.agent_active]
        # for the non-active agents, next state is set to most recent active state
        next_states[~self.agent_active] = self.last_active_agent_position[~self.agent_active]
        
        agent_mask = self.ctrl_sim_dset.get_agent_mask(
            copy.deepcopy(next_states[:, None, :self.ctrl_sim_dset.HEAD_IDX+1]), 
            self.local_frame)[:, 0]
        # print("agent_mask:", agent_mask)
        # assert False, "agent_mask check"
        
        # newly added agents:
        # not active previously (self.agent_active set to 0) 
        # in the simulation radius (agent_mask set to 1)
        # have not yet previously left scene (once left, cannot re-enter)
        newly_added_agent_mask = np.logical_and(
            np.logical_and(
                ~self.agent_active, 
                agent_mask
            ),
            ~self.left_scene
        )
        still_existing_agent_mask = np.logical_and(
            self.agent_active,
            agent_mask
        )

        if newly_added_agent_mask.sum():
            new_agent_idxs_to_remove = self._find_invalid_new_agents(
                next_states, 
                newly_added_agent_mask, 
                still_existing_agent_mask,
            )
            # remove new vehicle from scene if it doesn't respect time gap
            for agent_idx in new_agent_idxs_to_remove:
                self.left_scene[agent_idx] = True
        
        self.left_scene = np.logical_or(
            self.left_scene,
            (self.agent_active.astype(int) - agent_mask.astype(int)) == 1
        )
        
        # activated agents are those 
        # - in the FOV 
        # - have not previously left the scene
        self.agent_active = agent_mask * ~self.left_scene

        # update the data dictionary agent information
        self.data_dict['agent_active'] = copy.deepcopy(self.agent_active)
        self.data_dict['agent_active_history'].append(copy.deepcopy(self.agent_active))
        self.data_dict['agent'].append(next_states)
        self.data_dict['agent_action'].append(self.data_dict['agent_next_action'])
        self.data_dict['agent_rtg'].append(self.data_dict['agent_next_rtg'])
        
        # update the data dictionary ego information
        self.data_dict['ego'].append(self.ego_state[None, :])
        
        # 保存当前时间步数据
        self.save_step_data()

        # 检测终止条件
        terminated = False
        completed_route = ego_completed_route(
            self.local_frame['center'], 
            self.scenario_dict['route']
        )
        collided_with_agents = ego_collided(
            self.ego_state, 
            self.data_dict['agent'][-1][self.agent_active],
            agent_scale=self.cfg.sim.agent_scale
        )
        # 障碍物碰撞的具体几何判定由 Scenario Dreamer 后端封装，不泄露给高级规划器。
        obstacle_collision_ids = self.obstacle_wrapper.colliding_ids(self.ego_state)
        collided = bool(collided_with_agents or obstacle_collision_ids)
        off_route = ego_off_route(
            self.local_frame['center'], 
            self.scenario_dict['route'],
        )
        
        # 处理仿真终止情况：碰撞、偏离路线、完成路线或达到最大步数
        if (collided or off_route or completed_route 
            or self.t == self.cfg.sim.steps):
            # handle case where off route simply 
            # because you went past the endpoint of the route
            if completed_route:
                off_route = False 
                collided = bool(obstacle_collision_ids)
            
            progress = ego_progress(
                self.local_frame['center'], 
                self.scenario_dict['route']
            )
            terminated = True
            print(f"[Waarning] Terminated: {terminated}, Collided: {collided}, Off Route: {off_route}, Completed Route: {completed_route}, Progress: {progress:.2f}")
            info = {
                'collision': collided,
                'obstacle_collision_ids': obstacle_collision_ids,
                'off_route': off_route,
                'completed': completed_route,
                'progress': progress
            }
        else:
            info = {}

        # remove offroad / collided agents from scene
        if self.behaviour_model is not None:
            invalid_agents = self.behaviour_model.update_running_statistics(
                self.data_dict,
                self.scenario_dict,
                terminated,
            )
            invalid_agent_idxs = np.where(invalid_agents)[0]
            if len(invalid_agent_idxs):
                for idx in invalid_agent_idxs:
                    self.left_scene[idx] = True
                    self.agent_active[idx] = True
                self.data_dict['agent_active'] = copy.deepcopy(self.agent_active)

        self.current_state = self._get_observation()
        print(" Ego State:", self.ego_state)
        # print("anchors:", anchors)
        self._update_viz_state(anchors, refined_traj)
        
        return self.current_state, terminated, info
    

    def _get_observation(self):
        """ Get agent observation tensor for current time step."""
        if self.cfg.sim.policy == 'rl':
            ego_obs = get_ego_state(self.ego_state)
            # there is a one-step delay in gpudrive partner observations
            if self.t == 0:
                partner_idx = -1
            else:
                partner_idx = -2
            partner_obs = get_partner_obs(
                self.data_dict['agent'][partner_idx], 
                self.ego_state, 
                self.agent_active
            )
            map_obs = get_map_obs(
                self.data_dict['lanes_compressed'].copy(),
                self.ego_state
            )
            # Get route observations - route points should be centered on world_mean
            route_points = np.array(self.scenario_dict['route'], dtype=np.float32)
            route_obs = get_route_obs(
                route_points,
                self.ego_state
            )
            full_tensor = np.concatenate([ego_obs, partner_obs, map_obs, route_obs], axis=-1, dtype=np.float32)
            obs =  torch.from_numpy(full_tensor).to('cuda:0')
        else:
            # append active mask
            current_agent_states = np.concatenate(
                [self.data_dict['agent'][-1],
                np.expand_dims(copy.deepcopy(
                    self.agent_active
                ), axis=1)
                ], axis=1
            )
            ego_state = np.concatenate(
                [self.ego_state,
                 np.ones(1)])
            
            obs = np.concatenate([
                current_agent_states, 
                np.expand_dims(
                    ego_state, 
                    axis=0)
            ])
        
        return obs


    def _update_viz_state(self, anchors=None, refined_traj=None, num_route_points=30):
        """ Update visualization state for current time step."""
        current_agent_states = self.data_dict['agent'][-1]
        # print("current_agent_states:", current_agent_states)
        current_agent_types = self.data_dict['agent_type'][0]
        agent_active_mask = self.agent_active
        current_agent_states_rel = normalize_agents(
            current_agent_states[:, None], 
            normalize_dict=self.local_frame
        )[:, 0]
        
        lanes, lanes_mask = self.ctrl_sim_dset.get_normalized_lanes_in_fov(
            self.scenario_dict['lanes'], 
            normalize_dict=self.local_frame
        )
        lanes[~lanes_mask] = 0.0

        route = normalize_route(
            self.scenario_dict['route'], 
            normalize_dict=self.local_frame
        )
        dist_to_route = np.linalg.norm(route, axis=-1)
        route_start = np.argmin(dist_to_route)
        route = route[route_start:route_start+num_route_points]

        # 对 anchors 和 diffusion_trajectory 进行坐标转换
        if anchors is not None:
            anchors = np.array(anchors)  # 確保 anchors 是 NumPy 數組
            anchors = normalize_route(anchors, normalize_dict=self.local_frame)
        if refined_traj is not None:
            refined_traj = np.array(refined_traj)  # 確保 refined_traj 是 NumPy 數組
            refined_traj = refined_traj[:, :2]  # 提取 [x, y] 坐标
            refined_traj = normalize_route(refined_traj, normalize_dict=self.local_frame)

        # 将静态障碍物由全局坐标转换到当前自车局部坐标，仅供可视化使用。
        static_obstacles = []
        visualization_rotation = np.pi / 2.0 - float(self.local_frame['yaw'])
        for obstacle in self.get_static_obstacles():
            center_local = normalize_route(
                np.asarray([obstacle['center']], dtype=np.float32),
                normalize_dict=self.local_frame,
            )[0]
            local_yaw = float(obstacle['yaw']) + visualization_rotation
            static_obstacles.append({
                **obstacle,
                'center': center_local,
                # 用正弦和余弦将角度稳定映射到 [-pi, pi]。
                'yaw': float(np.arctan2(np.sin(local_yaw), np.cos(local_yaw))),
            })

        # 整理数据，供render_state函数读取，并传递给viz模块。
        self.viz_state = {
            'route': route,
            'agent_states': current_agent_states_rel,
            'agent_types': current_agent_types,
            'agent_active': agent_active_mask,
            'raw_agent_states': current_agent_states,
            'lanes': lanes,
            'lanes_mask': lanes_mask,
            'anchors': anchors,
            'diffusion_trajectory': refined_traj,
            'static_obstacles': static_obstacles,
        }
        # assert False

    def initialize_data_dict(self):
        """ Initialize data dictionary for simulation."""
        data_dict = {}

        ego = self.ego_state[None, :]
        ego_type = np.zeros((1,5))
        ego_type[0, 1] = 1

        agents = self.scenario_dict['agents'][:, 0]
        agent_types = self.scenario_dict['agent_types']

        data_dict['agent'] = [agents]
        data_dict['agent_type'] = [agent_types]
        data_dict['agent_action'] = []
        data_dict['agent_rtg'] = []
        data_dict['agent_next_action'] = []
        data_dict['agent_next_rtg'] = []

        data_dict['ego'] = [ego]
        data_dict['ego_type'] = [ego_type]
        data_dict['ego_action'] = []
        data_dict['ego_rtg'] = []
        # no ego next action because behaviour model does not predict that
        data_dict['ego_next_rtg'] = []

        # as ctrl-sim needs to process the lanes
        data_dict['lanes'] = self.scenario_dict['lanes']
        if self.cfg.sim.policy == 'rl':
            data_dict['lanes_compressed'] = self.scenario_dict['lanes_compressed']
        # which agents are actively being simulated at the current timestep
        data_dict['agent_active'] = copy.deepcopy(self.agent_active)
        data_dict['agent_active_history'] = [copy.deepcopy(self.agent_active)]

        self.data_dict = data_dict

        if self.behaviour_model is not None:
            invalid_agents = self.behaviour_model.update_running_statistics(self.data_dict, self.scenario_dict)
            invalid_agent_idxs = np.where(invalid_agents)[0]
            if len(invalid_agent_idxs):
                for idx in invalid_agent_idxs:
                    self.left_scene[idx] = True
                    self.agent_active[idx] = True
                self.data_dict['agent_active'] = copy.deepcopy(self.agent_active)
                self.data_dict['agent_active_history'][0] = copy.deepcopy(self.agent_active)


    def reset(self, i):
        """ Reset the environment for a new scenario given index."""
        self.t = 0
        self.current_scene_id = os.path.basename(self.test_files[i])
        self.pending_joint_trajectory = None
        # 缓存仅服务于显式启用的低频重规划，场景切换时必须清空。
        self._cached_joint_trajectory = None
        self._cached_attack_key = None
        self.attack_intent = None
        # 障碍物不跨场景保留；clear 会同时释放对应的后端对象。
        self.obstacle_wrapper.clear()
        # === 重置对抗轨迹状态 ===
        self.adversarial_agent_id = None
        self.adversarial_traj = None
        self.adversarial_step_idx = 0
        self.scenario_dict = self.load_initial_scene(i)

        self.ego_trajectory = self.scenario_dict['agents'][-1]
        # current state of the ego
        self.ego_state = self.ego_trajectory[0]

        self.rl_kinematics_model = ForwardKinematics(
            self.ego_state[:2], 
            self.ego_state[2:4], 
            self.ego_state[4],
            self.ego_state[5], 
            self.ego_state[6]
        )

        # 初始化其他交通参与者
        if self.cfg.sim.simulate_vehicles_only:
            vehicle_mask = self.scenario_dict['agent_types'][:-1, 1] == 1
        else:
            vehicle_mask = np.ones(self.scenario_dict['agent_types'][:-1].shape[0], dtype=bool)
        self.scenario_dict['agents'] = self.scenario_dict['agents'][:-1][vehicle_mask]
        self.scenario_dict['agent_types'] = self.scenario_dict['agent_types'][:-1][vehicle_mask]
        if self.traffic_backend == 'safe_sim_diffusion':
            non_vehicle_ids = np.flatnonzero(self.scenario_dict['agent_types'][:, 1] != 1)
            if len(non_vehicle_ids):
                raise ValueError(
                    "The configured Safe-Sim checkpoint is vehicle-only, but the scene contains "
                    f"non-vehicle agent IDs {non_vehicle_ids.tolist()}; set simulate_vehicles_only=True"
                )
        if self.mode == 'waymo_log_replay':
            self.scenario_dict['actions'] = self.scenario_dict['actions'][:-1][vehicle_mask]

        # 初始化行为模型
        if self.behaviour_model is not None:
            self.behaviour_model.reset(
                len(self.scenario_dict['agents']) + 1) # 加一是因为还需计入自车
        if self.diffusion_controller is not None:
            self.diffusion_controller.reset(self.current_scene_id)

        # 初始化数据字典
        self.local_frame = {
            'center': self.ego_trajectory[0, :2].copy(),
            'yaw': self.ego_trajectory[0, 4].copy()
        }

        # Find agents in FOV
        agent_mask = self.ctrl_sim_dset.get_agent_mask(
            copy.deepcopy(self.scenario_dict['agents'][:, :, :self.ctrl_sim_dset.HEAD_IDX+1]), 
            self.local_frame
        )
        # tells which of the non-ego agents are active
        # and thus get added to context + rendered in visualization
        self.agent_active = agent_mask[:, 0]
        self.left_scene = np.zeros_like(self.agent_active).astype(bool)

        # we initialize all agents to be most "recently activated" at the first timestep
        # TODO: This is really just a cache of the initial states, right? Why such a confusing variable name?
        self.last_active_agent_position = self.scenario_dict['agents'][:, 0]

        # Initialize data dictionary to track simulation state
        self.initialize_data_dict()

        # 保存初始数据
        self.save_initial_data(i)

        self.current_state = self._get_observation()
        self._update_viz_state()

        return self.current_state
    

    def render_llm_scene_image(self, agent_ids=None):
        """生成供多模态大模型读取的当前帧简化鸟瞰图。"""
        active_mask = np.asarray(self.viz_state['agent_active'], dtype=bool).copy()
        if agent_ids is not None:
            requested_mask = np.zeros_like(active_mask)
            requested_mask[np.asarray(agent_ids, dtype=np.int64)] = True
            active_mask &= requested_mask
        agent_states = self.viz_state['agent_states'][active_mask]
        agent_ids = np.flatnonzero(active_mask)
        ego_state = normalize_agents(
            self.ego_state[None, None, :],
            normalize_dict=self.local_frame,
        )[0, 0]
        return render_llm_scene_png(
            ego_state=ego_state,
            agent_states=agent_states,
            agent_ids=agent_ids,
            lanes=self.viz_state['lanes'],
            lanes_mask=self.viz_state['lanes_mask'],
        )

    def render_state(self, name, movie_path):
        """ Render the current state of the simulation."""
        agent_states = (
            self.viz_state['agent_states']
            [self.viz_state['agent_active']])
        # print("self.viz_state['agent_active']:", self.viz_state['agent_active'])

        # 提取 agent_active 所有 True 值的下标，从 0 开始计数
        agent_active_indices = [i for i, active in enumerate(self.viz_state['agent_active']) if active]
        # print("Active agent indices (1-based):", agent_active_indices)

        raw_agent_states = (
            self.viz_state['raw_agent_states']
            [self.viz_state['agent_active']])
        
        ego_state = normalize_agents(
            self.ego_state[None, None, :], 
            normalize_dict=self.local_frame
        )[:, 0]
        states = np.concatenate(
            [agent_states, ego_state]
            , axis=0)

        agent_types = (
            self.viz_state['agent_types']
            [self.viz_state['agent_active']])
        agent_types = np.concatenate(
            [agent_types, 
             np.array(
                 [0,1,0,0,0], dtype=int
             )[None, :]
            ], axis=0)

        route = self.viz_state['route']
        lanes = self.viz_state['lanes']
        lanes_mask = self.viz_state['lanes_mask']
        anchors = self.viz_state['anchors']
        diffusion_trajectory = self.viz_state['diffusion_trajectory']
        static_obstacles = self.viz_state['static_obstacles']
        visualization_cfg = self.cfg.sim.get('visualization', {})
        # 可视化开关只控制绘图，不修改 LLM 意图或 Safe-Sim 闭环推理数据。
        show_llm_anchors = bool(visualization_cfg.get('show_llm_anchors', True))
        show_diffusion_trajectory = bool(
            visualization_cfg.get('show_diffusion_trajectory', True)
        )
        
        render_state(
            states, 
            raw_agent_states,
            agent_types, 
            route,  # 简单巡线生成的自车未来轨迹
            lanes,  # 道路中心线列表
            lanes_mask, 
            anchors,
            diffusion_trajectory,
            self.t, 
            name, 
            movie_path, 
            lightweight=self.cfg.sim.lightweight,
            # active_agent_ids=self.activate_agent_ids  # 传递活跃交通参与者的全局编号
            active_agent_ids=agent_active_indices,  # 传递活跃交通参与者的全局编号
            static_obstacles=static_obstacles,
            show_llm_anchors=show_llm_anchors,
            show_diffusion_trajectory=show_diffusion_trajectory,
        )

    def save_initial_data(self, scenario_idx):
        """
        保存初始数据
        {
            "road_network": [...],  // 道路拓扑
            "road_network_compressed": {...},  // 压缩道路拓扑
            "ego_vehicle": {
                "initial_state": [x, y, vx, vy, heading, length, width]
            },
            "agents": [
                {
                    "initial_state": [x, y, vx, vy, heading, length, width]
                },
                ...
            ]
        }
        """
        # scenario_dict KEYS: dict_keys(['route', 'agents', 'agent_types', 'num_agents',
        #                                'route_lane_indices', 'lanes', 'lane_graph', 'num_lanes'])
        # data_dict KEYS: dict_keys(['agent', 'agent_type', 'agent_action', 'agent_rtg', 'agent_next_action',
        #                            'agent_next_rtg', 'ego', 'ego_type', 'ego_action', 'ego_rtg', 'ego_next_rtg', 'lanes', 'agent_active'])

        initial_data = {
            "road_network": self.scenario_dict['lanes'].tolist(),
            #"road_network_compressed": self.scenario_dict.get('lanes_compressed', None).tolist(),
            "ego_vehicle": {
                "initial_state": self.ego_state.tolist()
            },
            # "agents": {
            #     f"agent_{idx}": {
            #         "agent_type": int(np.argmax(self.scenario_dict['agent_types'][idx])), # 獨熱編碼 [is_unset, is_vehicle, is_pedestrian, is_cyclist, is_other] 轉爲int
            #         "initial_state": agent.tolist()
            #     }
            #     for idx, agent in enumerate(self.scenario_dict['agents'])
            # }
            "agents": {
                f"agent_{idx}": {
                    "agent_type": ["unset", "vehicle", "pedestrian", "cyclist", "other"][np.argmax(self.scenario_dict['agent_types'][idx])],
                    "initial_state": agent.tolist()
                }
                for idx, agent in enumerate(self.scenario_dict['agents'])
            }
        }
        self.step_data = {
            "ego_vehicle": [],
            **{f"agent_{idx}": [] for idx in range(len(self.data_dict['agent'][0]))}
        }
        output_dir = os.path.join(self.cfg.sim.scenario_data_output_path, "initial_data")
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"scenario_{scenario_idx:04d}_initial.json")
        with open(file_path, 'w') as f:
            json.dump(initial_data, f, indent=4)
        print(f"Initial data saved to {file_path}")

    def save_step_data(self):
        """
        保存当前时间步数据
        "ego_vehicle": [
            [x, y, vx, vy, heading, length, width],
            [x, y, vx, vy, heading, length, width],
            ...
        ],
        "agent_0": [
            [x, y, vx, vy, heading, length, width],
            [x, y, vx, vy, heading, length, width],
            ...
        ],
        """
        self.step_data["ego_vehicle"].append(self.ego_state.tolist())
        for idx, agent in enumerate(self.data_dict['agent'][-1]): # type(agent) = numpy.ndarray
            self.step_data[f"agent_{idx}"].append(agent.tolist())

    def dump_step_data(self, cur_scenario_idx):
        # 如果仿真结束，保存所有时间步数据到单个 JSON 文件
        output_dir = os.path.join(self.cfg.sim.scenario_data_output_path, "step_data")
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"scenario_{cur_scenario_idx:04d}_all_steps.json")
        with open(file_path, 'w') as f:
            json.dump(self.step_data, f, indent=4)
        print(f"All step data saved to {file_path}")

    def get_state_for_planning(self):
        """提取当前环境状态供大模型分析,生成env_state_json字典"""
        agents_info = []
        history_info = {}
        # active_agent_ids = []  # 用于存储活跃交通参与者的全局编号
        ego_x, ego_y = self.ego_state[:2]
        frame = self.get_scenario_frame(
            history_frames=int(self.cfg.sim.traffic_model.history_frames)
        )

        # 筛选距离自车附近的交通参与者
        # 或许可以采用KD-Tree等空间索引方法加速查找
        for idx, agent in enumerate(self.data_dict['agent'][-1]):
            if self.agent_active[idx]:
                agent_x, agent_y = agent[:2]
                distance = np.sqrt((agent_x - ego_x)**2 + (agent_y - ego_y)**2)
                if distance <= self.nearby_distance:
                    agent_history = []
                    for history_step in range(frame.history_global.shape[1]):
                        valid = bool(frame.history_mask[idx, history_step])
                        agent_history.append({
                            "state": [float(s) for s in frame.history_global[idx, history_step, :5]],
                            "valid": valid,
                        })
                    agents_info.append({
                        "id": idx,
                        "state": [float(s) for s in agent[:5]],  # x, y, vx, vy, heading
                        "type": ["unset", "vehicle", "pedestrian", "cyclist", "other"][np.argmax(self.scenario_dict['agent_types'][idx])],
                        "history": agent_history,
                    })
                    history_info[str(idx)] = agent_history
                    # self.activate_agent_ids.append(idx)  # 记录活跃交通参与者的全局编号

        # 筛选自车较近范围内的道路点
        route_info = []
        # route_info = [[float(p[0]), float(p[1])] for p in self.scenario_dict['route'][:5]]
        # print("route:", self.scenario_dict['route'])
        for point in self.scenario_dict['route']:
            route_x, route_y = point[:2]
            distance = np.sqrt((route_x - ego_x)**2 + (route_y - ego_y)**2)
            if distance <= self.nearby_distance:
                route_info.append([float(route_x), float(route_y)])

        return {
            "ego_state": [float(s) for s in self.ego_state[:5]],
            "route": route_info,
            "agents": agents_info,
            # 仅导出抽象字段，防止规划器获取 Scenario Dreamer 或 CARLA 的内部对象。
            "static_obstacles": self.get_static_obstacles(),
            "history": history_info,
            "history_order": "oldest_to_newest",
            "current_step": self.current_step,
        }

    def inject_adversarial_trajectory(self, agent_idx, trajectory):
        """仅供旧版 CtRL-Sim 使用的单车轨迹覆盖接口。

        Safe-Sim 使用 ``inject_joint_trajectory``，并要求一次预测包含攻击目标在内的
        所有活跃交通参与者。
        """
        if self.traffic_backend == 'safe_sim_diffusion':
            raise RuntimeError(
                "Safe-Sim 已禁用单车轨迹注入，请使用 inject_joint_trajectory"
            )
        self.adversarial_agent_id = agent_idx
        self.adversarial_traj = trajectory
        self.adversarial_step_idx = 0
        # 保存对抗轨迹
        self.data_dict['adversarial_trajectory'] = trajectory

class CtRLSimBehaviourModel:
    NUM_AGENT_STATES = 8  # [pos_x, pos_y, vel_x, vel_y, heading, length, width, existence]
    NUM_AGENT_TYPES = 5  # [is_unset, is_vehicle, is_pedestrian, is_cyclist, is_other]
    
    """ Behaviour model wrapper for Ctrl-Sim model used in simulation."""
    def __init__(self, 
                 mode,
                 model_path,
                 model,
                 dset,
                 use_rtg,
                 predict_rtgs,
                 action_temperature,
                 tilt,
                 steps):

        self.mode = mode
        self.model_path = model_path 
        self.model = model 
        self.model.eval()
        self.dset = dset
        self.cfg_model = model.cfg.model
        self.cfg_dataset = model.cfg.dataset
        
        self.steps = steps
        self.use_rtg = use_rtg 
        self.predict_rtgs = predict_rtgs
        self.action_temperature = action_temperature 
        self.tilt = tilt
        self.t = 0

        # for aggregating metrics
        self.agent_active_all = []
        self.sim_lin_speeds = []
        self.gt_lin_speeds = []
        self.sim_ang_speeds = []
        self.gt_ang_speeds = []
        self.sim_accels = []
        self.gt_accels = []
        self.sim_dist_near_veh = [] 
        self.gt_dist_near_veh = []
        self.collision_rate_scenario = []
        self.offroad_rate_scenario = []

        self.has_collided = None
        self.has_offroad = None
        # which agents (since beginning of trajectory) has been activated. Used for computing metrics.
        self.has_activated = None
        self.has_activated_vehicle = None
    
    def update_running_statistics(
            self, 
            data_dict,
            scenario_dict, 
            scene_complete=False,
            offroad_threshold=3.0
        ):
        """ Update running statistics for behaviour model metrics."""
        # scenario_dict: agents: [A, 91, 8] (no ego vehicle)
        # data_dict: agent: [self.t, A, 8]: [pos_x, pos_y, vel_x, vel_y, heading, length, width, existence]

        is_vehicle = data_dict['agent_type'][0][:, 1] == 1
        invalid_agents = np.zeros(
            data_dict['agent_active'].shape[0]
        ).astype(bool)
        
        if self.t == 0:
            self.has_collided = np.zeros(
                data_dict['agent_active'].shape[0]
            ).astype(bool)
            self.has_offroad = np.zeros(
                data_dict['agent_active'].shape[0]
            ).astype(bool)
            self.has_activated = data_dict['agent_active']
            self.has_activated_vehicle = np.logical_and(
                data_dict['agent_active'], is_vehicle)
        
        else:
            self.has_activated = np.logical_or(
                self.has_activated,
                data_dict['agent_active']
            )
            
            active_vehicles = np.logical_and(
                data_dict['agent_active'], 
                is_vehicle
            )
            self.has_activated_vehicle = np.logical_or(
                self.has_activated_vehicle,
                active_vehicles
            )
        
        agent_active = data_dict['agent_active']
        self.agent_active_all.append(agent_active)
        
        # compute simulated and ground-truth features for metrics
        if self.mode == 'waymo_ctrl_sim':
            sim_agents = np.array(
                data_dict['agent']
            )[self.t, agent_active]
            gt_agents = np.array(
                scenario_dict['agents']
                [agent_active, self.t])

            sim_vels = sim_agents[:, 2:4]
            gt_vels = gt_agents[:, 2:4]
            sim_lin_speeds = np.linalg.norm(sim_vels, axis=-1)
            gt_lin_speeds = np.linalg.norm(gt_vels, axis=-1)
            self.sim_lin_speeds.append(sim_lin_speeds)
            self.gt_lin_speeds.append(gt_lin_speeds)

            sim_ang_speeds = np.rad2deg(sim_agents[:, 4]) / 0.1
            gt_ang_speeds = np.rad2deg(gt_agents[:, 4]) / 0.1
            self.sim_ang_speeds.append(sim_ang_speeds)
            self.gt_ang_speeds.append(gt_ang_speeds)

            if self.t > 0:
                accel_mask = np.logical_and(
                    self.agent_active_all[self.t],
                    self.agent_active_all[self.t - 1]
                )
                
                sim_vels_all_t = np.array(
                    data_dict['agent'])[self.t, :, 2:4]
                gt_vels_all_t = np.array(
                    scenario_dict['agents'][:, self.t, 2:4])
                sim_vels_all_tminus1 = np.array(
                    data_dict['agent'])[self.t-1, :, 2:4]
                gt_vels_all_tminus1 = np.array(
                    scenario_dict['agents'][:, self.t-1, 2:4])

                sim_vels_t = sim_vels_all_t[accel_mask]
                gt_vels_t = gt_vels_all_t[accel_mask]
                sim_vels_tminus1 = sim_vels_all_tminus1[accel_mask]
                gt_vels_tminus1 = gt_vels_all_tminus1[accel_mask]

                sim_accels = np.linalg.norm(
                    (sim_vels_t - sim_vels_tminus1) / 0.1, axis=-1)
                gt_accels = np.linalg.norm(
                    (gt_vels_t - gt_vels_tminus1) / 0.1, axis=-1)

                self.gt_accels.append(gt_accels)
                self.sim_accels.append(sim_accels)
            
            if sim_agents.shape[0] > 1:
                sim_pos = sim_agents[:, :2]
                sim_pairwise_distances = np.linalg.norm(
                    sim_pos[:, np.newaxis, :] 
                    - sim_pos[np.newaxis, :, :], axis=-1)
                np.fill_diagonal(sim_pairwise_distances, np.inf)
                sim_dist_near_veh = np.min(sim_pairwise_distances, axis=1)

                gt_pos = gt_agents[:, :2]
                gt_pairwise_distances = np.linalg.norm(
                    gt_pos[:, np.newaxis, :] 
                    - gt_pos[np.newaxis, :, :], axis=-1)
                np.fill_diagonal(gt_pairwise_distances, np.inf)
                gt_dist_near_veh = np.min(gt_pairwise_distances, axis=1)

                self.sim_dist_near_veh.append(sim_dist_near_veh)
                self.gt_dist_near_veh.append(gt_dist_near_veh)

        # determine which agents (of those currently activated are colliding)
        sim_agents = np.array(data_dict['agent'])[self.t, agent_active]
        if sim_agents.shape[0] > 1:
            agents_colliding = compute_collision_states_one_scene(
                modify_agent_states(sim_agents)
            )
            
            active_agent_idxs = np.where(agent_active == 1)[0]
            colliding_all = np.zeros(len(agent_active)).astype(bool)
            for active_agent_idx, agent_colliding in zip(
                active_agent_idxs, agents_colliding):
                colliding_all[active_agent_idx] = agent_colliding

            # compute the offroad rate for vehicles
            normalize_dict = {  
                'center': data_dict['ego'][self.t][0, :2].copy(),
                'yaw': data_dict['ego'][self.t][0, 4].copy()
            }
            lanes, lanes_mask = self.dset.get_normalized_lanes_in_fov(
                data_dict['lanes'], 
                normalize_dict
            )
            lanes_resampled = resample_lanes_with_mask(
                lanes, 
                lanes_mask, 
                num_points=100
            )
            
            agents_normalized = normalize_agents(
                data_dict['agent'][self.t][:, None], 
                normalize_dict
            )
            min_dist_to_lane = np.linalg.norm(
                lanes_resampled.reshape(-1, 2)[None, :] - 
                agents_normalized[:, :, :2], axis=-1).min(1)
            agents_offroad = min_dist_to_lane > offroad_threshold
            agents_offroad[~agent_active] = False
            agents_offroad[~is_vehicle] = False
            offroad_all = agents_offroad

            # remove agents that are colliding
            invalid_agents = np.logical_or(
                invalid_agents,
                colliding_all
            )
            # remove agents that are offroad
            invalid_agents = np.logical_or(
                invalid_agents,
                offroad_all
            )
            
            self.has_collided = np.logical_or(
                self.has_collided,
                colliding_all
            )
            self.has_offroad = np.logical_or(
                self.has_offroad,
                offroad_all
            )

        if scene_complete:
            if np.sum(self.has_activated) > 0:
                collision_rate = (np.sum(self.has_collided) 
                                  / np.sum(self.has_activated))
            else:
                collision_rate = 0.

            if np.sum(self.has_activated_vehicle) > 0:
                offroad_rate = (np.sum(self.has_offroad) 
                                / np.sum(self.has_activated_vehicle))
            else:
                offroad_rate = 0.
            
            self.collision_rate_scenario.append(collision_rate)  
            self.offroad_rate_scenario.append(offroad_rate)
            self.agent_active_all = [] 

        return invalid_agents

    
    def compute_metrics(self):
        """ Compute behaviour model metrics after all scenarios have been run."""
        metrics_dict = {
            'collision_rate': np.array(
                self.collision_rate_scenario).mean(),
            'offroad_rate': np.array(
                self.offroad_rate_scenario).mean()
        }

        if self.mode == 'waymo_ctrl_sim':
            metrics_dict = compute_sim_agent_jsd_metrics(
                metrics_dict,
                self.gt_lin_speeds,
                self.sim_lin_speeds,
                self.gt_ang_speeds,
                self.sim_ang_speeds,
                self.gt_accels,
                self.sim_accels,
                self.gt_dist_near_veh,
                self.sim_dist_near_veh
            )
        
        return metrics_dict, ["{}: {:.6f}".format(k,v) for (k,v) in metrics_dict.items()]

    def reset(self, num_agents):
        """ Reset the behaviour model state for a new scenario."""
        self.t = 0
        self.states = np.zeros((num_agents, self.steps, self.NUM_AGENT_STATES))
        self.types = np.zeros((num_agents, self.NUM_AGENT_TYPES))
        self.actions = np.zeros((num_agents, self.steps))
        self.rtgs = np.ones((num_agents, self.steps, self.cfg_model.num_reward_components)) * MAX_RTG_VAL

    def update_state(self, data_dict):
        """ Update the internal state of the behaviour model with new data."""
        # now, EGO is the first index
        self.states[:1, self.t, :] = data_dict['ego'][self.t]
        self.states[1:, self.t, :] = data_dict['agent'][self.t]

        if self.t == 0:
            self.types[:1] = data_dict['ego_type'][0]
            self.types[1:] = data_dict['agent_type'][0]
        
        # for ego, we use the action from the RL policy
        # for the other agents, that is what ctrl-sim is for
        self.actions[:1, self.t] = data_dict['ego_action'][self.t] 
        self.rtgs[:1, self.t, :] = data_dict['ego_rtg'][self.t] 
        
        # Update previous timestep actions and rtgs for non-ego agents.
        if self.t > 0:
            self.actions[1:, self.t-1] = data_dict['agent_action'][self.t-1]
            if self.predict_rtgs:
                self.rtgs[1:, self.t-1, 0] = data_dict['agent_rtg'][self.t-1]

        # clear out cache for all non-existing agents
        self.states[1:][~data_dict['agent_active']] = 0
    
    def get_motion_data(self, data_dict):
        """ Prepare inputs to CtRL-Sim model for forward pass."""
        timesteps = np.arange(
            self.cfg_dataset.train_context_length
        ).astype(int)

        # retrieve relevant context
        if self.t < self.cfg_dataset.train_context_length:
            ag_states = self.states[:, :self.cfg_dataset.train_context_length].copy()
            ag_types = self.types.copy()
            actions = self.actions[:, :self.cfg_dataset.train_context_length].copy()
            rtgs = self.rtgs[:, :self.cfg_dataset.train_context_length, 0].copy()
            rtg_mask = ag_states[:, :, -1]
            timestep_buffer = np.repeat(
                timesteps[np.newaxis, :, np.newaxis], 
                self.cfg_dataset.max_num_agents, 
                0
            )
            normalize_timestep = self.t
        else:
            ag_states = self.states[:,self.t-(
                self.cfg_dataset.train_context_length - 1
                ):self.t+1].copy()
            ag_types = self.types.copy()
            actions = self.actions[:, self.t-(
                self.cfg_dataset.train_context_length - 1
                ):self.t+1].copy()
            rtgs = self.rtgs[:, self.t-(
                self.cfg_dataset.train_context_length - 1
                ):self.t+1, 0].copy()
            rtg_mask = ag_states[:, :, -1]
            timestep_buffer = np.repeat(
                timesteps[np.newaxis, :, np.newaxis], 
                self.cfg_dataset.max_num_agents, 
                0)
            normalize_timestep = self.cfg_dataset.train_context_length - 1

        # ego is index 0 now
        normalize_dict = {
            'center': ag_states[0, normalize_timestep, :2].copy(),
            'yaw': ag_states[0, normalize_timestep, 4].copy()
        }
        
        # filters out observations that are not within the FOV at the normalize_timestep
        agent_mask = self.dset.get_agent_mask(
            copy.deepcopy(ag_states[:, :, :self.dset.HEAD_IDX+1]
        ), normalize_dict)

        # we don't filter out non-moving agents
        moving_agent_mask = np.ones(
            ag_states.shape[0]
        ).astype(bool)
        
        motion_datas = {}
        correspondences = {}
        motion_data_id = 0 
        # vehicle ids in the FOV (ie, that need to be predicted)
        # that have not yet been added to a data buffer for prediction.
        unaccounted_veh_ids = np.where(data_dict['agent_active'] == 1)[0]
        
        while len(unaccounted_veh_ids) > 0:
            (state_buffer, 
             agent_type_buffer, 
             agent_mask_buffer, 
             action_buffer, 
             rtg_buffer, 
             rtg_mask_buffer, 
             _,
             new_origin_agent_idx, 
             correspondence
             ) = self.dset.select_closest_max_num_agents(
                 ag_states, 
                 ag_types, 
                 agent_mask, 
                 actions, 
                 rtgs, 
                 rtg_mask, 
                 moving_agent_mask,
                 origin_agent_idx=0, 
                 timestep=normalize_timestep, 
                 active_agents=unaccounted_veh_ids + 1) # +1 because ego is index 0
            
            # correspondence[i] is the index of the 
            # i'th element in state_buffer in ag_states
            # This is because the ego is always closest 
            # to the ego and we define ego as first position
            assert correspondence[0] == 0
            # This now tells us the mapping to data_dict['agents']
            # as data_dict['agents'] does not include ego
            correspondence -= 1

            assert np.all(
                np.isin(correspondence[1:], 
                np.where(data_dict['agent_active'] == 1
            )[0]))
            
            lanes, lanes_mask = self.dset.get_normalized_lanes_in_fov(
                data_dict['lanes'], 
                normalize_dict
            )
            state_buffer = normalize_agents(
                state_buffer, 
                normalize_dict
            )
            
            # add ego indicator
            is_ego = np.zeros(len(state_buffer))
            is_ego[new_origin_agent_idx] = 1
            is_ego = is_ego.astype(int)
            is_ego = np.tile(is_ego[:, None, None], 
                             (1, self.cfg_dataset.train_context_length, 1))

            # EXIST_IDX still last index
            state_buffer = np.concatenate(
                [state_buffer[:, :, :-1], 
                 is_ego, 
                 state_buffer[:, :, -1:]], axis=-1)

            # filter out agents / lane positions that are not in the FOV
            state_buffer[~agent_mask_buffer.astype(bool)] = 0
            rtg_mask_buffer[~agent_mask_buffer.astype(bool)] = 0
            lanes = np.concatenate(
                [lanes, lanes_mask[:, :, None]]
                , axis=-1)

            motion_data = dict()
            motion_data['idx'] = self.t
            motion_data['agent'] = from_numpy({
                'agent_states': add_batch_dim(state_buffer),
                'agent_types': add_batch_dim(agent_type_buffer), 
                'actions': add_batch_dim(action_buffer),
                'rtgs': add_batch_dim(rtg_buffer[:, :, None]),
                'rtg_mask': add_batch_dim(rtg_mask_buffer[:, :, None]),
                'timesteps': add_batch_dim(timestep_buffer),
                'moving_agent_mask': add_batch_dim(moving_agent_mask)
            })
            motion_data['map'] = from_numpy({
                'road_points': add_batch_dim(lanes),
            })
            motion_data = CtRLSimData(motion_data)
            
            unaccounted_veh_ids = np.setdiff1d(
                unaccounted_veh_ids, 
                correspondence[1:])

            motion_datas[motion_data_id] = motion_data 
            correspondences[motion_data_id] = correspondence 
            motion_data_id += 1

        return motion_datas, correspondences


    def get_tilt_logits(self, tilt):
        """ Get tilted logits for reward-to-go prediction."""
        rtg_bin_values = np.zeros((self.cfg_dataset.rtg_discretization, 1))
        rtg_bin_values[:, 0] = tilt * np.linspace(0, 1, self.cfg_dataset.rtg_discretization)
        # test print
        # print("tilt:",tilt, "rtg_bin_values:",rtg_bin_values)
        return rtg_bin_values
    
    def process_predicted_rtg(
            self, 
            rtg_logits, 
            token_index, 
            data_dict, 
            motion_data, 
            tensor_id, 
            veh_id, 
            is_tilted=False
        ):
        """ Process predicted reward-to-go for a single agent."""
        next_rtg_logits = rtg_logits[0, tensor_id, token_index].reshape(
            self.cfg_dataset.rtg_discretization, 
            self.cfg_model.num_reward_components
        )
        
        if is_tilted:
            tilt_logits = torch.from_numpy(
                self.get_tilt_logits(self.tilt)
            ).cuda()
        else:
            tilt_logits = torch.from_numpy(
                self.get_tilt_logits(0)
            ).cuda()
        
        next_rtg_dis = F.softmax(
            next_rtg_logits[:, 0] 
            + tilt_logits[:, 0], dim=0)
        next_rtg = torch.multinomial(
            next_rtg_dis, 
            1)
        motion_data['agent'].rtgs[0, tensor_id, token_index, 0] = next_rtg.item()
        data_dict['agent_next_rtg'][veh_id] = next_rtg.item()

        return data_dict, motion_data

    
    def predict(self, motion_datas, data_dict, correspondences):
        """ Predict next actions and rtgs for all agents given motion data."""
        if self.t < self.cfg_dataset.train_context_length:
            token_index = self.t 
        else:
            token_index = -1
        
        data_dict['agent_next_action'] = np.zeros(
            len(data_dict['agent'][0]))
        data_dict['agent_next_rtg'] = np.zeros(
            len(data_dict['agent'][0]))
        for motion_data_id in motion_datas:
            motion_data = motion_datas[motion_data_id]
            correspondence = correspondences[motion_data_id]

            motion_data = motion_data.cuda()
            # s --> R
            if self.predict_rtgs:
                preds = self.model(motion_data, eval=True)
                rtg_logits = preds['rtg_preds']
                
                # start from 1, as we don't predict the ego
                for tensor_id, veh_id in enumerate(correspondence):
                    if tensor_id == 0:
                        continue
                    data_dict, motion_data = self.process_predicted_rtg(
                        rtg_logits, 
                        token_index, 
                        data_dict, 
                        motion_data, 
                        tensor_id, 
                        veh_id, 
                        is_tilted=True
                    )
            
            # R --> a
            preds = self.model(motion_data, eval=True)
            # [batch_size=1, num_agents, timesteps, action_dim]
            logits = preds['action_preds']

            # temperature sampling for action prediction
            for tensor_id, veh_id in enumerate(correspondence):
                if tensor_id == 0:
                    continue
                next_action_logits = logits[0, tensor_id, token_index]
                next_action_dis = F.softmax(
                    next_action_logits 
                    / self.action_temperature, dim=0)
                next_action = torch.multinomial(
                    next_action_dis, 1)
                data_dict['agent_next_action'][veh_id] = next_action.item()
        
        return data_dict
    
    
    def step(self, data_dict):
        """ Step function for behaviour model to predict next actions and rtgs."""
        self.update_state(data_dict)
        motion_datas, correspondences = self.get_motion_data(data_dict)
        data_dict = self.predict(motion_datas, data_dict, correspondences)

        self.t += 1
        return data_dict
