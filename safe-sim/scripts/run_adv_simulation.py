"""
A script for evaluating closed-loop simulation, merging policy/env initialization
with guidance configuration, parameter parsing, and result directory naming.
"""

import argparse
import importlib
import json
import os
import sys
import pickle
import random
from collections import Counter
from pprint import pprint

import numpy as np
import torch
import yaml
from imageio import get_writer

from tbsim.configs.eval_config import EvaluationConfig

from tbsim.policies.wrappers import  Pos2YawWrapper, RolloutWrapper
from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.utils.env_utils import rollout_episodes, build_environment_and_scenes
from tbsim.utils.tensor_utils import map_ndarray

from tbsim.configs.guidance_config import GuidanceConfig
from tbsim.configs.base import Dict

torch.set_float32_matmul_precision("medium")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas.io.feather_format")

# ------------------------------------------------------------------
# Utility functions for parsing incoming guidance parameters
# ------------------------------------------------------------------

def parse_value(value_str):
    """Parse string value into appropriate Python type."""
    value_str = value_str.strip()
    
    if value_str.lower() == 'none':
        return None
    if value_str.lower() == 'true':
        return True
    if value_str.lower() == 'false':
        return False
        
    if value_str.startswith('[') and value_str.endswith(']'):
        try:
            elements = [parse_value(e.strip()) for e in value_str[1:-1].split(',')]
            return elements
        except:
            pass
            
    if value_str.startswith('{') and value_str.endswith('}'):
        try:
            import ast
            return ast.literal_eval(value_str)
        except:
            pass
    
    try:
        return float(value_str) if '.' in value_str else int(value_str)
    except:
        return value_str

def parse_guidance_params(param_strings):
    """
    Parse guidance parameters from a single string of the form:
      guidance_name,param_name,value;guidance_name2,param_name2,value2
    """
    if not param_strings:
        return {}
        
    params_dict = {}
    param_groups = param_strings.split(';')
    
    for param_str in param_groups:
        try:
            parts = param_str.split(',', 2)
            if len(parts) != 3:
                raise ValueError(f"Invalid parameter format: {param_str}")
                
            guidance_name, param_name, value_str = parts
            value = parse_value(value_str)
            
            if guidance_name not in params_dict:
                params_dict[guidance_name] = {}
            params_dict[guidance_name][param_name] = value
            
        except Exception as e:
            raise ValueError(
                f"Error parsing parameter: {param_str}\n"
                f"Error: {str(e)}\n"
                "Expected format: guidance_name,param_name,value"
            )
    
    return params_dict

def build_single_policy(eval_cfg, device, exp_config=None):
    """
    构建单一策略（用于所有智能体，即 ego 和 agents 共享同一策略）。

    Args:
        eval_cfg (EvaluationConfig): 评估配置对象，包含 eval_class、ckpt_root_dir、policy.pos_to_yaw 等。
        device (torch.device): 运行设备(CPU/GPU)。
        exp_config (ExperimentConfig, optional): 实验配置。

    Returns:
        policy: 构建好的策略实例。
        exp_config: 从策略组合器中返回的实验配置,包含算法超参数。
    """
    # 动态导入策略组合器模块（tbsim.evaluation.policy_composers）
    # 该模块包含多个策略组合器类（如 BC、CVAE、BITS 等），用于加载不同模型
    policy_composers = importlib.import_module("tbsim.evaluation.policy_composers")
    # 根据 eval_cfg.eval_class 字符串获取对应的组合器类（例如 'BC' -> BC 类）
    composer_class = getattr(policy_composers, eval_cfg.eval_class)
    # 实例化组合器，传入配置、设备和检查点根目录
    composer = composer_class(eval_cfg, device, ckpt_root_dir=eval_cfg.ckpt_root_dir)
    # 调用组合器的 get_policy 方法，加载模型权重并返回策略实例及实验配置
    policy, exp_config = composer.get_policy()
    
    # 若配置要求将位置输出转换为偏航角（即策略输出为位置偏移，需转化为绝对偏航）
    # Pos2YawWrapper 是一个装饰器，将策略输出的位置变化转换为航向角变化
    if eval_cfg.policy.pos_to_yaw:
        policy = Pos2YawWrapper(
            policy,
            dt=exp_config.algo.step_time if exp_config is not None else 0.1,    
            yaw_correction_speed=eval_cfg.policy.yaw_correction_speed           # 偏航修正速度
        )
    return policy, exp_config

def build_dual_policies(eval_cfg, device, modify_cfg):
    """
    构建双策略：分别用于 ego 车辆和其他智能体。

    此函数用于需要将 ego 和 agents 分开控制的场景（如对抗性仿真），
    其中 ego 使用一个策略（由 eval_cfg.eval_class 指定），
    agents 使用另一个策略（由 eval_cfg.agent_eval_class 指定）。

    Args:
        eval_cfg (EvaluationConfig): 评估配置对象，包含两个策略类名及检查点路径。
        device (torch.device): 运行设备。
        modify_cfg (Dict): 用于修改 agents 策略的配置（例如引导参数），
            通常由 run_adv_simulation 中的引导配置生成。

    Returns:
        policy: ego 策略实例。
        agent_policy: agents 策略实例。
        exp_config: 从 agents 策略组合器返回的实验配置。
    """
    policy_composers = importlib.import_module("tbsim.evaluation.policy_composers")

    # Build ego policy
    policy, _ = build_single_policy(eval_cfg, device)
    
    # Build agent policy
    composer_class = getattr(policy_composers, eval_cfg.agent_eval_class)
    composer = composer_class(eval_cfg, device, ckpt_root_dir=eval_cfg.ckpt_root_dir)
    agent_policy, exp_config = composer.get_policy(modify_config=modify_cfg)
    
    if eval_cfg.policy.pos_to_yaw:
        agent_policy = Pos2YawWrapper(
            agent_policy,
            dt=exp_config.algo.step_time if exp_config is not None else 0.1,
            yaw_correction_speed=eval_cfg.policy.yaw_correction_speed
        )
    
    return policy, agent_policy, exp_config
# ------------------------------------------------------------------
# The primary function that 1)intilize the scene and 2)runs simulation
# ------------------------------------------------------------------
def run_adv_simulation(eval_cfg,
                       data_to_disk,
                       render_to_video,
                       import_scene_list_mode="None",
                       visualize_diffusion=False,
                       guidance_params=None):
    """
    使用可配置的参数运行对抗模拟，同时以您之前使用的相同方式构建环境和策略。
    运行对抗性仿真,整合了环境初始化、策略加载、引导配置、场景批次执行与结果保存等全流程。

    主要用途：在闭环仿真中评估轨迹预测模型在对抗场景下的表现，支持多种环境
    (nuScenes, DriveSim, nuPlan, L5Kit)及可选的引导策略(如碰撞避免、偏离车道惩罚等)。

    参数说明：
        eval_cfg (EvaluationConfig): 包含所有仿真、策略、环境相关配置的配置对象。
        data_to_disk (bool): 是否将每个episode的原始观测/动作数据保存为HDF5文件。
        render_to_video (bool): 是否将仿真过程渲染为MP4视频并保存。
        import_scene_list_mode (str): 选择场景的策略，如 "intersection"（交叉口）、
            "human"、 "ttc"（碰撞时间）等，用于从数据集中筛选特定类型的场景。
        visualize_diffusion (bool): 当前未使用（可能为后续扩散模型可视化预留）。
        guidance_params (str): 一个特定格式的字符串，用于覆盖引导配置中的参数，
            格式示例："params,inner_lr,0.3;collision,radius,2.0"。

    返回值：
        result_stats (dict): 汇总所有仿真episode的统计指标,如碰撞率、偏离距离等。
        total_info (dict): 包含场景索引、场景ID等辅助信息。
        None: 不返回renderings。
        total_adjust_plan (dict): 每个episode的adjustment plan数据。
        total_trace (dict): 每个episode的轨迹追踪数据(如状态序列）。

    default eval_cfg: {
        "name": "exp1StrivePolicy_trajdata",
        "env": "nusc",
        "dataset_path": "/hqlab/dataset_nas1/nuscenes",
        "eval_class": "StrivePolicy_trajdata",
        "seed": 0,
        "num_scenes_per_batch": 1,
        "num_scenes_to_evaluate": 100,
        "num_episode_repeats": 1,
        "start_frame_index_each_episode": null,
        "seed_each_episode": null,
        "ego_only": false,
        "agent_eval_class": "Diffusion",
        "ckpt_root_dir": "checkpoints/",
        "experience_hdf5_path": null,
        "results_dir": "./output/exp1StrivePolicy_trajdata",
        "ckpt": {
            "policy": {
                "ckpt_dir": null,
                "ckpt_key": null
            },
            "planner": {
                "ckpt_dir": "safesim_checkpoints/safesim_fut32/test/run0",
                "ckpt_key": 70000
            },
            "predictor": {
                "ckpt_dir": "safesim_checkpoints/nusc_dynUnicycle_gl0_yrl0_tfTrue_4130553/run0/checkpoints",
                "ckpt_key": 94000
            },
            "cvae_metric": {
                "ckpt_dir": null,
                "ckpt_key": null
            },
            "occupancy_metric": {
                "ckpt_dir": null,
                "ckpt_key": null
            }
        },
        "policy": {
            "mask_drivable": true,
            "num_plan_samples": 50,
            "num_action_samples": 10,
            "pos_to_yaw": true,
            "yaw_correction_speed": 1.0,
            "diversification_clearance": null,
            "sample": true,
            "cost_weights": {
                "collision_weight": 15.0,
                "lane_weight": 1.0,
                "lane_dir_weight": 1.0,
                "likelihood_weight": 0.0,
                "progress_weight": 0.01
            }
        },
        "metrics": {
            "compute_analytical_metrics": true,
            "compute_agent2ego_metrics": false,
            "compute_learned_metrics": false
        },
        "perturb": {
            "enabled": false,
            "OU": {
                "theta": 0.8,
                "sigma": [
                    0.0,0.1,0.2,0.5,1.0,2.0,4.0
                ],
                "scale": [
                    1.0,1.0,0.2
                ]
            }
        },
        "rolling_perturb": {
            "enabled": false,
            "OU": {
                "theta": 0.8,
                "sigma": 0.5,
                "scale": [
                    1.0,1.0,0.2
                ]
            }
        },
        "occupancy": {
            "rolling": true,
            "rolling_horizon": [
                5,10,20
            ]
        },
        "cvae": {
            "rolling": true,
            "rolling_horizon": [
                5,10,20
            ]
        },
        "adjustment": {
            "enabled": false,
            "random_init_plan": false,
            "remove_existing_neighbors": false
        },
        "init_recipe": {
            "num_target_agents": 1,
            "predefined_scene_init": null
        },
        "vec_map_params": {
            "NUM_FUTURE_LANES": 6,
            "FIND_ALL_POS_REFS": true,
            "CENTERLINE_LENGTH": 150,
            "CENTERLINE_LOOKAHEAD": 150,
            "CENTERLINE_LOOKBACK": 50,
            "FIND_CLOSEST_DIST_THRESHOLD": 2.5,
            "FIND_CLOSEST_MAX_THRESHOLD": 5.0,
            "FIND_ALL_DIST_THRESHOLD": 10.0,
            "FIND_ALL_MAX_THRESHOLD": 20.0,
            "DEFAULT_MAX_HEADING_ERROR": 0.7853981633974483,
            "DFS_THRESHOLD_FUTURE": 150,
            "DFS_THRESHOLD_PAST": 50,
            "DFS_DEFAULT_THRESHOLD": 30,
            "EXTEND_DISTANCE": 100.0
        },
        "guidance": true,
        "guidance_fn": [
            "route",
            "collision",
            "ttc",
            "causecollision"
        ],
        "train": {
            "load_cache": false,
            "data_filter": ""
        },
        "eval_scenes": [
            0,1,2,3,4,5,6,……,147,148,149
        ],
        "n_step_action": 5,
        "num_simulation_steps": 100,
        "skip_first_n": 0
    }

    """
    ## set env config 

    # ------------------ Reproducibility settings ------------------
    # 固定所有随机种子（Python内置random、NumPy、PyTorch CPU和CUDA），
    # 并禁用cuDNN的自动优化（benchmark=False）以及开启确定性算法，
    # 保证相同配置下多次运行结果一致。
    np.random.seed(eval_cfg.seed)
    random.seed(eval_cfg.seed)
    torch.manual_seed(eval_cfg.seed)
    torch.cuda.manual_seed(eval_cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ------------------ Choose batch type based on environment ------------------
    # 根据仿真环境选择对应的全局批量数据处理类型。
    # trajdata 适用于 nuScenes/DriveSim/nuPlan，使用 trajdata 库的数据结构；
    # l5kit 适用于 Lyft Level 5 数据集。
    if eval_cfg.env in ["nusc", "drivesim", "nuplan"]:
        set_global_batch_type("trajdata")
    elif eval_cfg.env == 'l5kit':
        set_global_batch_type("l5kit")

    # print("eval_cfg:",eval_cfg)

    # ------------------ Guidance configuration ------------------
    # Guidance用于在仿真中施加外部约束或目标，如避免碰撞、保持车道等。
    # 创建 GuidanceConfig 实例，并设置引导函数,由 eval_cfg.guidance_fn 指定。
    guide_config = GuidanceConfig()
    guide_config.set_guidance_fn(eval_cfg.guidance_fn)

    # 引导算法的默认超参数，这些参数控制引导梯度更新的强度、采样策略等。
    guide_config.update_params({
        'inner_lr': 0.2,                                # 内循环优化学习率
        'scale_grad_by_std': False,                     # 是否按标准差缩放梯度
        'inner_beta': 0.5,                              # 内循环优化的动量系数
        'multiple_guidance_strategy': "weight_guide",   # 多引导目标融合策略
        'grad_wrt': "clean_guide",
        'sample_mode': "ddpm",
        'sample_step': 1                                # 扩散采样步数
    })

    # 如果用户通过 guidance_params 传入了自定义参数覆盖，则解析并应用。
    if guidance_params:
        parsed_params = parse_guidance_params(guidance_params)
        for guidance_name, params in parsed_params.items():
            if guidance_name == 'params':
                guide_config.update_params(params)                  # 更新通用参数
            elif guidance_name == 'combine_loss':
                guide_config.update_combine_loss(params)            # 更新多损失组合权重
            else:
                guide_config.update_config(guidance_name, params)   # 更新特定引导函数的配置

    # Wrap into `modify_cfg`
    modify_cfg = Dict()
    modify_cfg.guide_config = Dict(guide_config.to_dict())

    # ------------------ Build the result directory name ------------------
    # result_dir_name = create_result_dir_name(guidance_params)
    # 创建结果保存目录
    eval_cfg.results_dir = os.path.join(eval_cfg.results_dir)
    os.makedirs(eval_cfg.results_dir, exist_ok=True)

    # Optionally store the final guidance config & params
    # 在结果目录下创建 configs 子目录，用于存储本次运行的引导配置快照
    config_dir = os.path.join(eval_cfg.results_dir, "configs")
    os.makedirs(config_dir, exist_ok=True)
    if guidance_params:
        # 保存用户传入的原始参数解析结果
        with open(os.path.join(config_dir, "guidance_params.json"), "w") as f:
            json.dump(parse_guidance_params(guidance_params), f, indent=2)
    # 保存最终的完整引导配置字典
    with open(os.path.join(config_dir, "full_guidance_config.json"), "w") as f:
        json.dump(guide_config.to_dict(), f, indent=2)

    # ------------------ Prepare device ------------------
    # 检测CUDA可用性，优先使用GPU（cuda:0），否则回退到CPU。
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ------------------ Build the policy/policies and environment ------------------    
    # 构建策略:
    # 如果未指定 agent_eval_class，则所有智能体共用同一个策略；
    # 否则分别构建 ego 策略和 agents 策略。
    # single policy for all agents:
    if eval_cfg.agent_eval_class is None:
        policy, exp_config = build_single_policy(eval_cfg, device)
        agent_policy = None
    # dual policies for ego and agents:
    else:
        policy, agent_policy, exp_config = build_dual_policies(eval_cfg, device, modify_cfg) # policy, agent_policy:Pos2YawWrapper,Pos2YawWrapper

    # 根据环境类型和配置，用 RolloutWrapper 对策略进行包装
    if eval_cfg.env in ["nusc","drivesim","nuplan"]:
        # 对于 trajdata 环境，若存在 agent_policy 则分别设置，否则所有智能体使用 agent_policy。
        if eval_cfg.agent_eval_class is not None:
            rollout_policy = RolloutWrapper(ego_policy=policy, agents_policy=agent_policy)
        else:
            rollout_policy = RolloutWrapper(agents_policy=policy)
    elif eval_cfg.ego_only:
        # 仅控制 ego 车辆（其他车辆按数据集真实轨迹行驶）
        rollout_policy = RolloutWrapper(ego_policy=policy)
    else:
        # 默认情况：若存在 agent_policy 则分别设置，否则 ego 和 agents 共用同一策略
        if eval_cfg.agent_eval_class is not None:
            rollout_policy = RolloutWrapper(ego_policy=policy, agents_policy=agent_policy)
        else:
            rollout_policy = RolloutWrapper(ego_policy=policy, agents_policy=policy)
            
    # 构建仿真环境并获取可用的场景列表。
    # build_environment_and_scenes 根据 eval_cfg 创建环境实例、加载场景数据，并根据 import_scene_list_mode 筛选场景，同时返回预定义的初始状态（predefined_init）。
    env, eval_scenes, predefined_init = build_environment_and_scenes(eval_cfg, exp_config, device, agent_policy, import_scene_list_mode)
    # 3) Store that predefined init in the config so your environment/rollout can see it
    eval_cfg.init_recipe["predefined_scene_init"] = predefined_init
    
    obs_to_torch = eval_cfg.eval_class not in ["GroundTruth", "ReplayAction"]

    # We will store stats, etc.
    result_stats = None
    scene_i = 0

    # Prepare for data collection
    total_adjust_plan = {}   # 累积所有episode的调整计划
    total_trace = {}         # 累积所有episode的轨迹追踪数据
    total_info = {}          # 累积所有episode的辅助信息

    # -------------- Actual simulation rollout loop --------------
    # 循环处理场景，每次处理一批（batch），直到达到评估数量上限或场景列表用尽。
    while scene_i < min(eval_cfg.num_scenes_to_evaluate, len(eval_scenes)):
        scene_indices = eval_scenes[scene_i : scene_i + eval_cfg.num_scenes_per_batch]
        scene_i += eval_cfg.num_scenes_per_batch

        # 调用 rollout_episodes 执行仿真，初始化每个场景为独立episode；在每个时间步调用 rollout_policy 获取动作；将动作应用到环境，推进状态；收集统计量、渲染图像、调整计划和轨迹数据。
        stats, info, renderings, adjust_plans, trace = rollout_episodes(
            env,
            rollout_policy,
            num_episodes=eval_cfg.num_episode_repeats,
            n_step_action=eval_cfg.n_step_action,
            render=render_to_video,
            skip_first_n=eval_cfg.skip_first_n,
            scene_indices=scene_indices,
            obs_to_torch=obs_to_torch,
            start_frame_index_each_episode=eval_cfg.start_frame_index_each_episode,
            seed_each_episode=eval_cfg.seed_each_episode,
            horizon=eval_cfg.num_simulation_steps,
            adjust_plan_recipe=eval_cfg.adjustment.to_dict() if eval_cfg.adjustment.enabled else None,
            init_recipe=eval_cfg.init_recipe,
            control_config=modify_cfg,
            device=device,
        )

        # Merge stats
        if result_stats is None:
            result_stats = stats
            result_stats["scene_index"] = np.array(info["scene_index"])
        else:
            # 沿第0维（批次维度）拼接所有数值型统计量
            for k in stats:
                result_stats[k] = np.concatenate([result_stats[k], stats[k]], axis=0)
            result_stats["scene_index"] = np.concatenate(
                [result_stats["scene_index"], np.array(info["scene_index"])]
            )

        # Collect adjustments and traces
        # 每个episode可能有多个调整计划（如多个时间步的调整），以字典形式存储。
        for ei, adjust_plan in enumerate(adjust_plans):
            for k, v in adjust_plan.items():
                total_adjust_plan[f"{k}_{ei}"] = v
        for ei, trace_i in enumerate(trace):
            for k, v in trace_i.items():
                total_trace[f"{k}_{ei}"] = v

        # Print the stats for this batch
        # print(info["scene_index"])
        # pprint(stats)

        # Save stats to disk
        stats_filepath = os.path.join(eval_cfg.results_dir, "stats.json")
        stats_to_write = map_ndarray(result_stats, lambda x: x.tolist())
        with open(stats_filepath, "w") as fp:
            json.dump(stats_to_write, fp)

        # ------------------ 保存视频（如果启用） ------------------
        if render_to_video:
            video_dir = os.path.join(eval_cfg.results_dir, "videos")
            os.makedirs(video_dir, exist_ok=True)
            for ei, episode_rendering in enumerate(renderings):
                for i, scene_images in enumerate(episode_rendering):
                    outname = f"{info['scene_index'][i]}_{ei}.mp4"
                    writer = get_writer(os.path.join(video_dir, outname), fps=10)
                    print(f"Video -> {os.path.join(video_dir, outname)}")
                    for im in scene_images:
                        writer.append_data(im)
                    writer.close()

        # Possibly save data to disk
        if data_to_disk and "buffer" in info:
            dump_episode_buffer(
                info["buffer"],
                info["scene_index"],
                h5_path=os.path.join(eval_cfg.results_dir, "data.hdf5")
            )

        # ------------------ 保存调整计划 ------------------
        if total_adjust_plan:
            with open(os.path.join(eval_cfg.results_dir, "adjust_plan.json"), "w") as fp:
                json.dump(total_adjust_plan, fp)
                print("Saved adjust_plan.json")

        info_except_buffer = {k: v for k, v in info.items() if k != "buffer"}
        for k, v in info_except_buffer.items():
            if k not in total_info:
                total_info[k] = v
            else:
                if isinstance(v, list):
                    total_info[k].extend(v)
                elif isinstance(v, dict):
                    total_info[k].update(v)

        # 保存总信息
        with open(os.path.join(eval_cfg.results_dir, "sim_info.json"), "w") as fp:
            json.dump(total_info, fp)
            print("Saved sim_info.json")

        # 保存trace数据
        if total_trace:
            with open(os.path.join(eval_cfg.results_dir, "trace.pkl"), "wb") as fp:
                pickle.dump(total_trace, fp)
                print("Saved trace.pkl")

        torch.cuda.empty_cache()

    return result_stats, total_info, None, total_adjust_plan, total_trace


def dump_episode_buffer(buffer, scene_index, h5_path):
    """
    Example method to dump data from each scene into an HDF5 file.
    """
    import h5py
    h5_file = h5py.File(h5_path, "a")
    ep_count = Counter()
    for si, scene_buffer in zip(scene_index, buffer):
        ep_i = ep_count[si]
        ep_count[si] += 1
        for mk in scene_buffer:
            h5key = f"/{si}_{ep_i}/{mk}"
            h5_file.create_dataset(h5key, data=scene_buffer[mk])
    h5_file.close()
    print(f"scene {scene_index} written to {h5_path}")


# ------------------------------------------------------------------
# CLI argument parser setup
# ------------------------------------------------------------------

def add_simulation_args(parser):
    """Add simulation-related arguments (already done in your code)."""
    parser.add_argument("--config_file", type=str, default=None,
                       help="A json file containing evaluation configs")
    parser.add_argument("--local_rank", type=int, default=0,
                       help="local rank for torch.distributed")
    parser.add_argument("--env", type=str, required=True,
                       choices=["nusc", "drivesim", "nuplan", "l5kit"],
                       help="Environment to use")
    parser.add_argument("--eval_class", type=str, default=None,
                       help="Optionally specify the evaluation class through argparse")
    parser.add_argument("--agent_eval_class", type=str, default=None,
                       help="Optionally specify the evaluation class for agents if it's different from ego")
    parser.add_argument("--ckpt_root_dir", type=str, default=None,
                       help="Root directory to look for training run directories")
    parser.add_argument("--policy_ckpt_dir", type=str, default=None,
                       help="Directory to look for saved checkpoints")
    parser.add_argument("--policy_ckpt_key", type=str, default=None,
                       help="A string that uniquely identifies a checkpoint file within a directory, e.g., iter50000")
    parser.add_argument("--dataset_path", type=str, default=None,
                       help="Root directory of the dataset")
    parser.add_argument("--num_scenes_per_batch", type=int, default=None,
                       help="Number of scenes to run concurrently (to accelerate eval)")
    parser.add_argument("--results_root_dir", type=str, required=True,
                       help="Root directory for results")
    parser.add_argument("--render", action="store_true",
                       help="Whether to render simulation to video")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--guidance", action="store_true")
    parser.add_argument("--guidance_fn_weights", type=lambda s: list(map(float, s.split('_'))), default=None)
    parser.add_argument("--load-cache", action="store_true",
                       help="whether to evaluate the model")
    parser.add_argument("--data_filter_fn", type=str, choices=["has_lane"], default="")
    parser.add_argument("--scene_select_mode", type=str, default="intersection",
                       choices=["human", "ttc", "partial_diffusion", "no_collision"])
    parser.add_argument("--visualize-diffusion", action="store_true",
                       help="whether to evaluate the model")
    parser.add_argument("--sim-steps", type=int, default=100)
    parser.add_argument("--num_scenes_to_evaluate", type=int, default=100)
    parser.add_argument("--skip_first_n", type=int, default=0)
    parser.add_argument("--n_step_action", type=int, default=5)
    parser.add_argument("--guidance_fn", type=lambda s: s.split('_'),
                       default=["offroad"])
    parser.add_argument("--guidance_params", type=str,
                       help="""
                       Guidance parameters in format: guidance_name,param_name,value;guidance_name2,param_name2,value2
                       
                       Examples:
                       - Basic parameters:
                         params,inner_lr,0.2;params,scale_grad,False
                       - Lists and weights:
                         combine_loss,weights,[0.5,0.5,0.0,0.0]
                       - Dictionaries:
                         causecollision,adv_term_weight,{"distance":1.0,"speed_penalty":0.0}
                       - Multiple parameters:
                         params,inner_lr,0.3;collision,radius,2.0;speed,desired_speed,2.5
                       """)
    parser.add_argument(
        "--ckpt_yaml",
        type=str,
        help="specify a yaml file that specifies checkpoint and config location of each model",
        default=None
    )
    parser.add_argument(
        "--metric_ckpt_yaml",
        type=str,
        help="specify a yaml file that specifies checkpoint and config location for the learned metric",
        default=None
    )


def main():
    parser = argparse.ArgumentParser(description="Run adversarial simulation")
    add_simulation_args(parser)
    args = parser.parse_args()

    # Load base config
    cfg = EvaluationConfig()
    if args.config_file is not None:
        external_cfg = json.load(open(args.config_file, "r"))
        cfg.update(**external_cfg)

    # Update config with command line arguments
    if args.eval_class is not None:
        cfg.eval_class = args.eval_class

    if args.ckpt_root_dir is not None:
        cfg.ckpt_root_dir = args.ckpt_root_dir

    if args.policy_ckpt_dir is not None:
        assert args.policy_ckpt_key is not None, "Please specify a key to look for the checkpoint, e.g., 'iter50000'"
        cfg.ckpt.policy.ckpt_dir = args.policy_ckpt_dir
        cfg.ckpt.policy.ckpt_key = args.policy_ckpt_key

    if args.num_scenes_per_batch is not None:
        cfg.num_scenes_per_batch = args.num_scenes_per_batch
        
    # Set simulation related parameters
    cfg.nusc.num_simulation_steps = args.sim_steps
    cfg.num_scenes_to_evaluate = args.num_scenes_to_evaluate
    cfg.nusc.skip_first_n = args.skip_first_n
    cfg.nusc.n_step_action = args.n_step_action

    if args.dataset_path is not None:
        cfg.dataset_path = args.dataset_path

    if cfg.name is None:
        cfg.name = cfg.eval_class

    if args.prefix is not None:
        cfg.name = args.prefix + cfg.name

    if args.agent_eval_class is not None:
        cfg.agent_eval_class = args.agent_eval_class

    if args.seed is not None:
        cfg.seed = args.seed
        
    if args.results_root_dir is not None:
        cfg.results_dir = os.path.join(args.results_root_dir, cfg.name)
    else:
        cfg.results_dir = os.path.join(cfg.results_dir, cfg.name)

    if args.env is not None:
        cfg.env = args.env
    else:
        assert cfg.env is not None
        

    # Set guidance parameters
    cfg.guidance = args.guidance
    cfg.guidance_fn = args.guidance_fn

    # Set cache parameters
    assert not args.load_cache
    cfg.train.load_cache = args.load_cache
    cfg.train.data_filter = args.data_filter_fn

    # Update environment sub-config
    for k in cfg["nusc"]:  # hardcoded for now! copy env-specific config to the global-level
        cfg[k] = cfg["nusc"][k]

    # Remove env keys if needed
    cfg.pop("nusc")
    cfg.pop("drivesim")
    cfg.pop("l5kit")

    # Load checkpoint YAMLs if specified
    if args.ckpt_yaml is not None:
        with open(args.ckpt_yaml, "r") as f:
            ckpt_info = yaml.safe_load(f)
            cfg.ckpt.update(**ckpt_info)
    if args.metric_ckpt_yaml is not None:
        with open(args.metric_ckpt_yaml, "r") as f:
            ckpt_info = yaml.safe_load(f)
            cfg.ckpt.update(**ckpt_info)

    # Lock config
    cfg.lock()

    # Run simulation
    stats, info, renderings, adjust_plans, trace = run_adv_simulation(
        eval_cfg=cfg,
        data_to_disk=True,
        render_to_video=args.render,
        import_scene_list_mode=args.scene_select_mode,
        visualize_diffusion=args.visualize_diffusion,
        guidance_params=args.guidance_params
    )

    print("Simulation completed successfully!")
    print(f"Results saved to: {cfg.results_dir}")


if __name__ == "__main__":
    main()