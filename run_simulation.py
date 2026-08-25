import hydra
from simulator import Simulator
from policies.idm_policy import IDMPolicy
from policies.rl_policy import RLPolicy
from cfgs.config import CONFIG_PATH

import numpy as np
import torch
import random 
from tqdm import tqdm
from utils.viz import generate_video
import json
import os
from scenario_generator import AdversarialScenarioGenerator
from policies.risk_metrics import compute_scenario_danger_score

class PolicyEvaluator:
    """ Evaluate a given policy in a simulation environment over multiple scenarios."""
    def __init__(self, cfg, policy, env, user_instruction):
        """ Initialize the PolicyEvaluator."""
        self.cfg = cfg
        # policy being evaluated
        self.policy = policy
        # simulation environment
        self.env = env

        # 实例化对抗场景生成器
        self.generator = AdversarialScenarioGenerator(
            cfg,
            user_instruction,
            llm_planner=env.llm_planner,
        )
    
    def reset(self):
        """ Reset the evaluator's statistics and random seeds."""
        torch.manual_seed(self.cfg.seed)
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        
        self.collision = []
        self.off_route = []
        self.completed = []
        self.progress = []
        # 重置CARLA数据
    
    def update_running_statistics(self, info):
        """ Update running statistics with info from the latest episode."""
        self.collision.append(info['collision'])
        self.off_route.append(info['off_route'])
        self.completed.append(info['completed'])
        self.progress.append(info['progress'])

    
    def compute_metrics(self):
        """ Compute evaluation metrics based on accumulated statistics."""
        base_metrics = {
            'collision rate': np.array(self.collision).astype(float).mean(),
            'off route rate': np.array(self.off_route).astype(float).mean(),
            'completed rate': np.array(self.completed).astype(float).mean(),
            'progress': np.array(self.progress).astype(float).mean()
        }
        # 在不改变原有指标含义的前提下，合并 TTC、EA 和可达性指标。
        adv_metrics = self.generator.compute_final_metrics()

        all_metrics = {**base_metrics, **adv_metrics}
        # 将所有独立危险性指标汇总为统一的零到一综合得分。
        composite_cfg = getattr(getattr(self.cfg, "evaluation", None), "composite", None)
        all_metrics["scenario_danger_score"] = compute_scenario_danger_score(
            all_metrics,
            composite_cfg,
        )
        return all_metrics, ["{}: {:.6f}".format(k,v) for (k,v) in all_metrics.items()]

    def evaluate_policy(self):
        """ Evaluate the policy over all test scenarios in the environment."""
        self.reset()
        
        # 遍历所有测试场景
        for i in tqdm(range(self.env.num_test_scenarios)):
            print(f"Simulating environment {i}")
            obs = self.env.reset(i)

            # 重置单回合对抗统计
            self.generator.reset_episode_stats()

            if hasattr(self.policy, 'reset'):
                self.policy.reset(obs)

            # 在单个场景中执行固定步数的交互
            for _ in range(self.env.steps):
                current_t = self.env.current_step
                # 1. 调用生成器：低频触发大模型规划与轨迹注入
                self.generator.step(self.env, current_t)

                # 2. Safe-Sim 先基于当前快照预测所有受控非自车参与者；Simulator.step 仅执行第一帧。
                self.env.prepare_background_traffic()
                # 必须在联合轨迹已准备但尚未执行时，与攻击前同源帧可达集对比。
                self.generator.evaluate_reachability(self.env)
                self.generator.set_diffusion_trajectory(
                    self.env.get_attack_target_prediction()
                )
                anchors, refined_traj = self.generator.get_anchors_and_trajectory()

                if self.cfg.visualize:
                    # render_frame = True
                    # if self.cfg.lightweight:
                    #     if t%3 != 0:
                    #         render_frame = False
                    # observations always rendered in local frame of agent
                    # if render_frame:
                    self.env.render_state(name=f'{i}', movie_path=self.cfg.movie_path)
                
                # 3. 自车决策与环境步进
                action = self.policy.act(obs)
                obs, terminated, info = self.env.step(action, anchors, refined_traj)

                # 4. 采集原有碰撞/TTC指标和新增的二维 EA。
                self.generator.evaluate_reaction(self.env, info)

                if terminated:
                    self.env.dump_step_data(i)
                    break

                print("step:", current_t, " of ", self.env.steps)

            # 场景结束时保留原有的回合最小 TTC 汇总。
            self.generator.finalize_episode_stats()
            self.update_running_statistics(info)
            
            if self.cfg.visualize:
                generate_video(name=f'{i}', output_dir=self.cfg.movie_path, delete_images=False)
            
            if self.cfg.verbose:
                if self.cfg.behaviour_model.compute_metrics and self.env.behaviour_model is not None:
                    print("behaviour model metrics: ", self.env.behaviour_model.compute_metrics()[-1])
                # policy metrics
                print(self.compute_metrics()[-1])


        return self.compute_metrics()

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    torch.manual_seed(cfg.sim.seed)
    random.seed(cfg.sim.seed)
    np.random.seed(cfg.sim.seed)

    # 从 attack_request.txt 文件读取攻击指令
    attack_request_file = "attack_request.txt"
    if os.path.exists(attack_request_file):
        with open(attack_request_file, "r") as f:
            user_instruction = [line.strip() for line in f.readlines()]
    else:
        print(f"[Error] {attack_request_file} not found. No attack instructions loaded.")
        user_instruction = []

    # initialize simulation environments
    # cfg.sim contains all simulation related configurations
    env = Simulator(cfg)
    
    if cfg.sim.policy == 'rl':
        policy = RLPolicy(cfg.sim)
    else:
        policy = IDMPolicy(cfg, env)
    


    evaluator = PolicyEvaluator(cfg.sim, policy, env, user_instruction)
    _, metrics_str = evaluator.evaluate_policy()
    print(metrics_str)

if __name__ == "__main__":
    main()
