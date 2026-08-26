from typing import OrderedDict
import numpy as np
import pytorch_lightning as pl
import torch
import importlib
import os
from imageio import get_writer

from tbsim.envs.base import BatchedEnv, BaseEnv
from tbsim.configs.env_configs import get_eval_scenes_and_predefined_init, import_env_configs

import tbsim.utils.tensor_utils as TensorUtils
from tbsim.utils.timer import Timers
from tbsim.evaluation.env_builders import EnvNuscBuilder,EnvNuplanBuilder

from trajdata.simulation import SimulationScene
import random

from collections import defaultdict

def rollout_episodes(
    env,
    policy,
    num_episodes,
    skip_first_n=1,
    n_step_action=1,
    render=False,
    scene_indices=None,
    start_frame_index_each_episode=None,
    device=None,
    obs_to_torch=True,
    adjust_plan_recipe=None,
    init_recipe = None,
    control_config = None,
    horizon=None,
    seed_each_episode=None,
    reset_scene_index_map=False,
    initialize=False,    
):
    """
    Rollout an environment for a number of episodes
    Args:
        env (BaseEnv):一个基础模拟环境(类似gym)
        policy (RolloutWrapper):控制环境中智能体的策略,如DiffusionTrafficModel
        num_episodes (int):要运行的回合数
        skip_first_n (int):每 episode 开始时，跳过前 N 个时间步, 使用 ground truth 动作，用于预热环境状态
        n_step_action (int):每次查询模型之间执行的步数
        render (bool):如果为True,返回一系列渲染帧
        scene_indices (tuple, list):可选,要运行的场景索引
        start_frame_index_each_episode (List):可选,每个模拟回合从哪个帧开始
        device:将观测值转换为目标设备
        obs_to_torch:是否将观测值转换为torch
        adjust_plan_recipe (dict):可选，场景调整计划（例如改变车辆位置、速度等），用于生成对抗性场景。
        init_recipe (dict): 包含预定义初始状态的字典（如 ego 和 agents 的初始位姿），由 run_adv_simulation 传入，用于精确控制仿真起点。
        control_config (dict): 包含引导配置(如 GuidanceConfig)的字典,会传递给策略模型。
        horizon (int):(可选) 覆盖模拟的回合数
        seed_each_episode (List):(可选) 用于每个回合的种子列表
        reset_scene_index_map (bool): 是否重置场景索引映射表（用于批处理环境中重复使用同一场景时）。

    Returns:
        stats (dict):每个回合的运行统计信息字典（指标、奖励等）
        info (dict):每个回合的环境信息字典
        renderings (list):每个回合对应的渲染帧列表,以np.ndarray形式表示
    """
    stats = {}          # 聚合所有场景的统计量
    info = {}           # 聚合所有场景的信息
    renderings = []     # 存储每个 episode 的渲染帧序列
    is_batched_env = isinstance(env, BatchedEnv)  # 判断是否为批量环境（一次处理多场景）
    timers = Timers()   # 性能计时工具（用于记录各阶段耗时）
    adjust_plans = list()   # 存储每个 episode 的调整计划

    # ---------- 参数校验 ----------
    if seed_each_episode is not None:
        assert len(seed_each_episode) == num_episodes
    if start_frame_index_each_episode is not None:
        assert len(start_frame_index_each_episode) == num_episodes
        
    # 解包策略，获取 ego_policy
    ego_policy = policy.unwrap()["Rollout.ego_policy"] # 自车策略为 StrivePlanner_trajdata
    trace = list()
    # ---------- 主循环：依次运行每个 episode ----------
    for ei in range(num_episodes):
        # 确定该 episode 的起始帧索引
        if start_frame_index_each_episode is not None:
            start_frame_index = start_frame_index_each_episode[ei]
        else:
            start_frame_index = None
        
        env.reset(scene_indices=scene_indices, start_frame_index=start_frame_index)
        if adjust_plan_recipe is not None:
            if "random_init_plan" in adjust_plan_recipe:
                # recipe provided
                if adjust_plan_recipe["random_init_plan"]:
                    adjust_recipe = adjust_plan_recipe
                    raise NotImplementedError("Random initialization is not implemented yet")
                    adjust_plan = random_initial_adjust_plan(env,adjust_recipe)
                    
                else:
                    adjust_plan = None
                    adjust_recipe = None 
            else:
                # explicit plan provided
                adjust_plan = adjust_plan_recipe
        else:
            adjust_plan = None
            adjust_recipe = None
        # 如果存在调整计划，则应用到环境
        if adjust_plan is not None:
            env.adjust_scene(adjust_plan)
        #initialize the relations
        # if initialize:
        #     env.save_relationships_to_file()

        # ---------- 初始化智能体状态（使用 init_recipe） ----------
        if init_recipe["predefined_scene_init"] is not None:
            # 若需要重置场景索引映射（用于批处理中相同场景重复运行时区分）
            if not hasattr(env,"scene_index_map") or reset_scene_index_map:
                 #for multiple same scenes and run batch each iteration
                env.scene_index_map = defaultdict(int)
                env.scene_occurrence_map = defaultdict(int)
            # 根据 init_recipe 初始化 ego 和目标 agents 的状态（位置、速度、朝向等）
            env.init_ego_and_target_agents(init_recipe)
            env.adjust_ego() #set ego indices
            
            # 将控制索引信息传递到 control_config，供策略的引导模块使用
            if control_config is not None:
                control_config.guide_config.batch_ctrl_indices = env.batch_ctrl_indices
                control_config.guide_config.batch_ego_indices = env.batch_ego_indices
                policy.agents_policy.policy.nets["policy"].update_guide_config(control_config,device) # 更新策略内部的引导配置，指定设备
        else:
            # 若没有预定义初始状态，则仅初始化 ego/agents 的索引分离（用于不同控制策略）
            env.init_split_indices()
       
        # ---------- 设置该 episode 的随机种子（若指定） ----------
        if seed_each_episode is not None:
            env.update_random_seed(seed_each_episode[ei])
            np.random.seed(seed_each_episode[ei])
            random.seed(seed_each_episode[ei])
            torch.manual_seed(seed_each_episode[ei])
            torch.cuda.manual_seed(seed_each_episode[ei])
            
        # ---------- 开始仿真循环 ----------
        done = env.is_done()
        counter = 0
        step_since_last_update = 0
        frames = list()
        while not done:
            timers.tic("step") # 计时开始

            # 1. 读取环境数据
            with timers.timed("obs"): 
                obs = env.get_observation(include_ego_obs=True) # env_trajdata.py 中的 get_observation 方法会返回一个字典，包含 ego 和 agents 的观测信息
            # 2. 转换观测为 torch 张量
            with timers.timed("to_torch"):
                # 默认为 True，将观测转换为 torch 张量，并移动到指定设备（如 GPU）
                if obs_to_torch:  
                    device = policy.device if device is None else device
                    obs_torch = TensorUtils.to_torch(obs, device=device, ignore_if_unspecified=True) # dict
                else:
                    obs_torch = obs

            # print("obs_torch:", obs_torch)

            # 3. 策略推理:计算动作
            with timers.timed("network"):
                # 将输入观测传入策略模型，获取动作输出
                print("type of policy:",type(policy))
                action = policy.get_action(obs_torch, step_index=counter) # 调用 RolloutWrapper.get_action 方法
                print(f"Episode {ei}, Step {counter}: Action computed: {action}")
                
            # 4. 动作执行(分为预热阶段和正常阶段)
            if counter < skip_first_n:
                # 预热阶段：使用 ground truth 动作（即数据集中真实记录的动作）
                # 目的是让环境状态（如速度）从真实数据平滑过渡，避免仿真初始不匹配
                gt_action = env.get_gt_action(obs) #TODO we should eliminate ego action in agents
                # 覆盖 ego 动作为真实动作
                action.ego = gt_action.ego
                # 从 agents 的动作中剔除 ego 的部分(防止重复控制)
                gt_action.agents.eliminate_ego_action(obs["agents"]["ego_idx"])
                action.agents = gt_action.agents
                # 执行一步（n_step_action 在预热阶段固定为1）
                env.step(action, num_steps_to_take=1, render=False)
                counter += 1
                step_since_last_update+=1
            else:
                # 正常阶段：使用策略输出动作，并连续执行 n_step_action 步
                with timers.timed("env_step"):
                    ims = env.step(
                        action, num_steps_to_take=n_step_action, render=render
                    )  # ims 为列表，每个元素是当前帧的渲染图像（形状 [num_scene, H, W, 3]）
                if render:
                    frames.extend(ims)
                counter += n_step_action
                step_since_last_update += n_step_action
            timers.toc("step")
            print(timers)

            # 检查是否终止（可能因为到达终点、碰撞或超出 horizon）
            done = env.is_done()

            assert False, "Single Step test."
            
            if horizon is not None and counter >= horizon:
                break

        # ---------- 该 episode 结束，收集指标和追踪数据 ----------
        metrics = env.get_metrics()
        if hasattr(ego_policy,"savetrace") and ego_policy.savetrace:
            trace.append(ego_policy.trace.copy())

        for k, v in metrics.items():
            if k not in stats:
                stats[k] = []
            if is_batched_env:  # concatenate by scene
                stats[k] = np.concatenate([stats[k], v], axis=0)
            else:
                stats[k].append(v)

        # 收集环境信息（如场景索引、帧号等）
        env_info = env.get_info()
        for k, v in env_info.items():
            if k not in info:
                if isinstance(v,dict):
                    info[k] = dict()
                else:
                    info[k] = list()

            if is_batched_env:
                if isinstance(v,dict):
                    info[k].update(v)
                else:
                    info[k].extend(v)
            else:
                info[k].append(v)
        del env_info
        if hasattr(ego_policy,"reset"):
            ego_policy.reset()

        # 若渲染启用，将收集的帧堆叠成数组并加入到 renderings 列表
        if render:
            frames = np.stack(frames)
            if is_batched_env:
                # [step, scene] -> [scene, step]
                frames = frames.transpose((1, 0, 2, 3, 4))
            renderings.append(frames)
        if adjust_plan is not None:
            adjust_plans.append(adjust_plan)

    # ---------- 所有 episode 完成后，收集跨 episode 的聚合指标 ----------
    multi_episodes_metrics = env.get_multi_episode_metrics()
    stats.update(multi_episodes_metrics)
    env.reset_multi_episodes_metrics()

    return stats, info, renderings, adjust_plans, trace

def build_environment_and_scenes(eval_cfg, exp_config, device, agent_policy=None, import_scene_list_mode="None"):
    """
        构建仿真环境并选择待评价的场景

    返回值：
        env:环境对象
        eval_scenes:待评估的场景索引列表
        predefined_init:场景初始化参数字典
    """
    # 1. Import environment-specific configs
    TRAIN_SCENE_IDX_MAP, SCENE_IDX_MAP, SCENE_NAMES, \
    PREDEFINED_SCENE_INIT, PREDEFINED_SCENE_ALL_INIT = import_env_configs(eval_cfg.env)

    # 2. Build environment based on type
    split_ego = agent_policy is not None # 若ego和agents使用不同的策略，则需要分离ego和agents的观测
    parse_obs = exp_config.env.data_generation_params.get("parse_obs", True)

    if eval_cfg.env == "nusc":
        env_builder = EnvNuscBuilder(eval_config=eval_cfg, exp_config=exp_config, device=device)
    elif eval_cfg.env == "nuplan":
        env_builder = EnvNuplanBuilder(eval_config=eval_cfg, exp_config=exp_config, device=device)
    else:
        raise ValueError(f"Unknown environment: {eval_cfg.env}")

    env = env_builder.get_env(split_ego=split_ego, parse_obs=parse_obs)

    # 3. Get scene selection based on mode
    eval_scenes, predefined_init = get_eval_scenes_and_predefined_init(
        import_scene_list_mode,
        eval_cfg,
        SCENE_IDX_MAP,
        SCENE_NAMES,
        PREDEFINED_SCENE_INIT,
        PREDEFINED_SCENE_ALL_INIT,
        TRAIN_SCENE_IDX_MAP
    )

    return env, eval_scenes, predefined_init
