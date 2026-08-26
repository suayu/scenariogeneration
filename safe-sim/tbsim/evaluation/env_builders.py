from copy import deepcopy
import numpy as np
from collections import defaultdict
import os
from functools import partial

# from l5kit.data import LocalDataManager, ChunkedDataset
# from l5kit.rasterization import build_rasterizer
from trajdata import AgentType, UnifiedDataset

# from tbsim.l5kit.vectorizer import build_vectorizer


from tbsim.configs.eval_config import EvaluationConfig
from tbsim.configs.base import ExperimentConfig
from tbsim.utils.metrics import OrnsteinUhlenbeckPerturbation
from tbsim.envs.env_trajdata import EnvUnifiedSimulation, EnvSplitUnifiedSimulation
from tbsim.utils.config_utils import  translate_trajdata_cfg
import tbsim.envs.env_metrics as EnvMetrics
from tbsim.evaluation.metric_composers import CVAEMetrics, OccupancyMetrics
from tbsim.utils.trajdata_utils import get_full_fut_traj,get_full_fut_valid, get_stationary_mask
# from tbsim.l5kit.l5_ego_dataset import EgoDatasetMixed

from trajdata.custom_func.get_lane_info import get_lane_info


class EnvironmentBuilder(object):
    """Builds an simulation environment for evaluation."""
    def __init__(self, eval_config: EvaluationConfig, exp_config: ExperimentConfig, device):
        self.eval_cfg = eval_config
        self.exp_cfg = exp_config
        self.device = device

    def _get_analytical_metrics(self):
        metrics = dict(
            ego_off_road_rate=EnvMetrics.OffRoadRateVec(),         # 自车偏离道路的比率（向量化）
            all_collision_rate=EnvMetrics.CollisionRate(),         # 所有智能体的碰撞率

            ego_failure=EnvMetrics.CriticalFailure(num_offroad_frames=2),  # 自车关键失败（偏离道路累积帧数）
            all_failure=EnvMetrics.CriticalFailure(num_offroad_frames=2),  # 所有智能体关键失败
        )
        return metrics
    
    def _get_agent2ego_metrics(self):
        # 衡量周围车辆与自车的相对运动风险。
        metrics = dict(
            agents2ego_ttc = EnvMetrics.TimeToCollisionMetrics(),   # TTC
            agents2ego_dist = EnvMetrics.DistanceMetrics(),         # 距离
        )
        return metrics
    
    def _get_learned_metrics(self):
        """
        构建基于学习模型的指标（如 CVAE 似然、占用栅格似然）。

        这些指标依赖预训练的神经网络模型（如条件变分自编码器 CVAE 或占用预测模型），
        用于评估预测轨迹的合理性或场景的拟人度。

        具体实现：
        1. 如果启用了扰动(perturb),则创建 Ornstein-Uhlenbeck 扰动过程，
           用于对输入轨迹添加随机噪声，以测试模型的鲁棒性。
        2. 加载 CVAE 模型（用于评估轨迹似然）和 Occupancy 模型（用于评估占用似然）。
           注意：此处硬编码了 ckpt_root_dir,可能用于调试或特定实验。
        3. 返回包含这些学习指标的字典。
        """
        perturbations = dict()
        if self.eval_cfg.perturb.enabled:
            # 根据配置的 sigma 列表创建多个 OU 扰动过程
            for sigma in self.eval_cfg.perturb.OU.sigma:
                perturbations["OU_sigma_{}".format(sigma)] = OrnsteinUhlenbeckPerturbation(
                    theta=self.eval_cfg.perturb.OU.theta * np.ones(3),   # 均值回归速度
                    sigma=sigma * np.array(self.eval_cfg.perturb.OU.scale)  # 波动幅度
                )
        # 注意：以下路径硬编码，可能用于特定测试，实际使用应来自配置。
        self.eval_cfg.ckpt_root_dir = "/net/acadia3a/data/wjchang/tbsim_output/0927_vae/test/run1"

        # 初始化 CVAE 指标计算器（用于评估轨迹的似然）
        cvae_metrics = CVAEMetrics(
            eval_config=self.eval_cfg,
            device=self.device,
            ckpt_root_dir=self.eval_cfg.ckpt_root_dir,
        )

        # 初始化占用栅格指标计算器（用于评估场景占用似然）
        learned_occu_metric = OccupancyMetrics(
            eval_config=self.eval_cfg,
            device=self.device,
            ckpt_root_dir=self.eval_cfg.ckpt_root_dir,
        )

        metrics = dict(
            all_cvae_metrics=cvae_metrics.get_metrics(
                self.eval_cfg,
                perturbations=perturbations,
                rolling=self.eval_cfg.cvae.rolling,          # 是否采用滚动窗口评估
                rolling_horizon=self.eval_cfg.cvae.rolling_horizon,  # 滚动窗口长度
                env=self.eval_cfg.env,
            ),
        )
        return metrics

    def get_env(self):
        # 抽象方法:子类必须实现,返回具体环境实例
        raise NotImplementedError


class EnvNuscBuilder(EnvironmentBuilder):
    def get_env(self, split_ego=False, parse_obs=True):
        """
        构建并返回 nuScenes 仿真环境实例。

        Args:
            split_ego (bool): 是否将 ego 和 agents 的控制分离（用于对抗场景，如引导策略）。
            parse_obs (bool): 是否解析观测（可能影响观测预处理方式）。

        Returns:
            env: 环境实例(EnvUnifiedSimulation 或 EnvSplitUnifiedSimulation)。
        """
        # 克隆实验配置并解锁以允许修改
        exp_cfg = self.exp_cfg.clone()
        exp_cfg.unlock()

        # 设置数据集路径和仿真参数
        exp_cfg.train.dataset_path = self.eval_cfg.dataset_path
        exp_cfg.env.simulation.num_simulation_steps = self.eval_cfg.num_simulation_steps
        # 起始帧索引：从历史帧数之后开始（保证有足够历史用于预测）
        exp_cfg.env.simulation.start_frame_index = exp_cfg.algo.history_num_frames + 1

        # 加载缓存和数据过滤配置（用于加速数据加载）
        exp_cfg.train.load_cache = self.eval_cfg.train.load_cache
        exp_cfg.train.data_filter = self.eval_cfg.train.data_filter
        exp_cfg.lock()   # 锁定配置，防止后续意外修改

        # 将实验配置转换为 trajdata 库可识别的配置（数据加载参数）
        data_cfg = translate_trajdata_cfg(exp_cfg)

        # 计算时间窗口参数（秒）
        future_sec = data_cfg.future_num_frames * data_cfg.step_time
        history_sec = data_cfg.history_num_frames * data_cfg.step_time
        neighbor_distance = data_cfg.max_agents_distance   # 智能体交互距离阈值

        # 构造 UnifiedDataset 的关键字参数
        kwargs = dict(
            desired_data=["val"],  # 仅使用验证集（避免训练集泄露）
            future_sec=(future_sec, future_sec),  # 固定未来窗口长度
            history_sec=(history_sec, history_sec),  # 固定历史窗口长度
            data_dirs={  # 数据目录映射（根据 trajdata 格式）
                "nusc_trainval": data_cfg.dataset_path,
                "nusc_mini": data_cfg.dataset_path,
            },
            only_types=[AgentType.VEHICLE],  # 只关心车辆（忽略行人、自行车等）
            agent_interaction_distances=defaultdict(lambda: neighbor_distance),  # 交互距离
            incl_raster_map=True,  # 包含栅格化地图（用于 CNN 输入）
            raster_map_params={  # 栅格地图参数
                "px_per_m": int(1 / data_cfg.pixel_size),  # 每米像素数
                "map_size_px": data_cfg.raster_size,       # 地图尺寸（像素）
                "return_rgb": False,                       # 不返回 RGB（单通道或语义通道）
                "offset_frac_xy": data_cfg.raster_center,  # 栅格中心偏移
                "original_format": True,                   # 使用原始数据集格式
            },
            incl_vector_map=True,  # 包含矢量地图（车道线等）
            vector_map_params={    # 矢量地图参数
                "incl_road_lanes": True,          # 包含车道线
                "incl_road_areas": False,         # 不包含道路区域
                "incl_ped_crosswalks": False,     # 不包含人行横道
                "incl_ped_walkways": False,       # 不包含步行道
                # 矢量地图的聚合可能较慢，如果不需要则关闭 collate
                "collate": False,
            },
            num_workers=os.cpu_count(),  # 数据加载并行线程数（使用全部 CPU 核心）
            # augmentations = [noise_hists],  # 可选的增强（此处注释掉）
            desired_dt=data_cfg.step_time,     # 期望的时间步长（0.1s）
            standardize_data=data_cfg.standardize_data,  # 是否标准化数据
            extras={  # 额外字段提取函数
                "closest_lane_point": partial(get_lane_info, VEC_MAP_PARAMS=self.eval_cfg.vec_map_params),
                "full_fut_valid": get_full_fut_valid,   # 获取未来轨迹有效性
                "full_fut_traj": get_full_fut_traj,     # 获取完整未来轨迹
            },
            obs_format="x,y,z,xd,yd,xdd,ydd,s,c",  # 观测特征格式（位置、速度、加速度、航向等）
            # max_neighbor_num = data_cfg.other_agents_num,  # 可选的最大邻居数
        )
        # print(os.cpu_count())  # 打印 CPU 核心数（调试信息）

        # 如果配置了矢量化车道（未用），可取消注释
        # if data_cfg.vectorize_lane!="none":
        #     kwargs["vectorize_lane"] = data_cfg.vectorize_lane

        # 创建统一数据集实例
        env_dataset = UnifiedDataset(**kwargs)

        # 初始化指标字典（当前为空，因为上述指标创建代码被注释）
        metrics = dict()
        # 以下注释了通过配置开关来添加指标，可根据需要启用
        # if self.eval_cfg.metrics.compute_analytical_metrics:
        #     metrics.update(self._get_analytical_metrics())

        # 根据 split_ego 标志选择环境类型
        if split_ego:
            env = EnvSplitUnifiedSimulation(
                exp_cfg.env,
                dataset=env_dataset,
                seed=self.eval_cfg.seed,
                num_scenes=self.eval_cfg.num_scenes_per_batch,  # 批处理场景数
                prediction_only=False,      # 不仅预测，还要执行仿真（控制）
                metrics=metrics,
                split_ego=split_ego,
                parse_obs=parse_obs,
            )
        else:
            env = EnvUnifiedSimulation(
                exp_cfg.env,
                dataset=env_dataset,
                seed=self.eval_cfg.seed,
                num_scenes=self.eval_cfg.num_scenes_per_batch,
                prediction_only=False,
                metrics=metrics,
            )

        return env
    
class EnvNuplanBuilder(EnvironmentBuilder):
    
    def get_env(self,split_ego=False,parse_obs=True):
        exp_cfg = self.exp_cfg.clone()
        exp_cfg.unlock()
        exp_cfg.train.dataset_path = self.eval_cfg.dataset_path
        exp_cfg.env.simulation.num_simulation_steps = self.eval_cfg.num_simulation_steps
        exp_cfg.env.simulation.start_frame_index = exp_cfg.algo.history_num_frames + 1
        # add for cache
        exp_cfg.train.load_cache = self.eval_cfg.train.load_cache
        exp_cfg.train.data_filter = self.eval_cfg.train.data_filter
        exp_cfg.lock()

        data_cfg = translate_trajdata_cfg(exp_cfg)

        future_sec = data_cfg.future_num_frames * data_cfg.step_time
        history_sec = data_cfg.history_num_frames * data_cfg.step_time
        neighbor_distance = data_cfg.max_agents_distance

        kwargs = dict(
            desired_data=["nuplan_mini-mini_val"], #["val"]
            future_sec=(future_sec, future_sec),
            history_sec=(history_sec, history_sec),
            # history_sec=(10.0, 10.0),
            ego_only=True,
            data_dirs={
                "nuplan_mini": "/net/acadia3a/data/datasets/nuplan/dataset/nuplan-v1.1",
            },
            only_types=[AgentType.VEHICLE],
            agent_interaction_distances=defaultdict(lambda: neighbor_distance),
            incl_raster_map=True,
            raster_map_params={
                "px_per_m": int(1 / data_cfg.pixel_size),
                "map_size_px": data_cfg.raster_size,
                "return_rgb": False,
                "offset_frac_xy": data_cfg.raster_center,
                "original_format": True,
            },
            incl_vector_map = True,
            vector_map_params = {
                "incl_road_lanes": True,
                "incl_road_areas": False,
                "incl_ped_crosswalks": False,
                "incl_ped_walkways": False,
                # Collation can be quite slow if vector maps are included,
                # so we do not unless the user requests it.
                "no_collate": True,
            },
            num_workers=os.cpu_count(),
            desired_dt=data_cfg.step_time,
            standardize_data=data_cfg.standardize_data,
            extras={
            "closest_lane_point": get_lane_info,
            # "all_possible_lane_pts": get_refs
            },
            obs_format="x,y,z,xd,yd,xdd,ydd,s,c",
            
            # max_neighbor_num = data_cfg.other_agents_num,
            # rebuild_cache=True,
            # rebuild_maps=True
        )
        print(os.cpu_count())
        # if data_cfg.vectorize_lane!="none":
        #     kwargs["vectorize_lane"] = data_cfg.vectorize_lane
        env_dataset = UnifiedDataset(**kwargs)

        metrics = dict()
        if self.eval_cfg.metrics.compute_analytical_metrics:
            metrics.update(self._get_analytical_metrics())
        # metrics = {}
        if split_ego:
            env = EnvSplitUnifiedSimulation(
                exp_cfg.env,
                dataset=env_dataset,
                seed=self.eval_cfg.seed,
                num_scenes=self.eval_cfg.num_scenes_per_batch,
                prediction_only=False,
                metrics=metrics,
                split_ego=split_ego,
                parse_obs = parse_obs,
            )
        else:
            env = EnvUnifiedSimulation(
                exp_cfg.env,
                dataset=env_dataset,
                seed=self.eval_cfg.seed,
                num_scenes=self.eval_cfg.num_scenes_per_batch,
                prediction_only=False,
                metrics=metrics,
            )

        return env

class EnvDrivesimBuilder(EnvironmentBuilder):
    def get_env(self,split_ego=False,parse_obs=True):
        exp_cfg = self.exp_cfg.clone()
        exp_cfg.unlock()
        exp_cfg.train.dataset_path = self.eval_cfg.dataset_path
        exp_cfg.env.simulation.num_simulation_steps = self.eval_cfg.num_simulation_steps
        exp_cfg.env.simulation.start_frame_index = exp_cfg.algo.history_num_frames + 1
        exp_cfg.lock()

        data_cfg = translate_trajdata_cfg(exp_cfg)

        future_sec = data_cfg.future_num_frames * data_cfg.step_time
        history_sec = data_cfg.history_num_frames * data_cfg.step_time
        neighbor_distance = data_cfg.max_agents_distance

        kwargs = dict(
            desired_data=["main"],
            future_sec=(0.1, future_sec),
            history_sec=(history_sec, history_sec),
            data_dirs={"drivesim":"home"},
            only_types=[AgentType.VEHICLE],
            agent_interaction_distances=defaultdict(lambda: neighbor_distance),
            incl_raster_map=True,
            raster_map_params={
                "px_per_m": int(1 / data_cfg.pixel_size),
                "map_size_px": data_cfg.raster_size,
                "return_rgb": False,
                "offset_frac_xy": data_cfg.raster_center,
                "original_format": True,
            },
            incl_vector_map = True,
            vector_map_params = {
                "incl_road_lanes": True,
                "incl_road_areas": False,
                "incl_ped_crosswalks": False,
                "incl_ped_walkways": False,
                # Collation can be quite slow if vector maps are included,
                # so we do not unless the user requests it.
                "no_collate": False,
            },
            # num_workers=os.cpu_count(),
            num_workers = 0,
            desired_dt=data_cfg.step_time,
            standardize_data=data_cfg.standardize_data,
            # max_neighbor_num = data_cfg.other_agents_num,
        )
        # if data_cfg.vectorize_lane!="none":
        #     kwargs["vectorize_lane"] = data_cfg.vectorize_lane
        env_dataset = UnifiedDataset(**kwargs)

        metrics = dict()
        if self.eval_cfg.metrics.compute_analytical_metrics:
            metrics.update(self._get_analytical_metrics())
        if self.eval_cfg.metrics.compute_learned_metrics:
            metrics.update(self._get_learned_metrics())
        if split_ego:
            env = EnvSplitUnifiedSimulation(
                exp_cfg.env,
                dataset=env_dataset,
                seed=self.eval_cfg.seed,
                num_scenes=self.eval_cfg.num_scenes_per_batch,
                prediction_only=False,
                metrics=metrics,
                split_ego=split_ego,
                parse_obs = parse_obs,
            )
        else:
            env = EnvUnifiedSimulation(
                exp_cfg.env,
                dataset=env_dataset,
                seed=self.eval_cfg.seed,
                num_scenes=self.eval_cfg.num_scenes_per_batch,
                prediction_only=False,
                metrics=metrics,
            )

        return env
