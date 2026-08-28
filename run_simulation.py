import hydra
from simulator import Simulator
from policies.idm_policy import IDMPolicy
from policies.rl_policy import RLPolicy
from cfgs.config import CONFIG_PATH

import numpy as np
import torch
import random 
from tqdm import tqdm
from utils.viz import generate_video, generate_multi_scenario_evaluation_visualization
import csv
import json
import os
from pathlib import Path
from scenario_generator import AdversarialScenarioGenerator
from policies.risk_metrics import compute_scenario_danger_score


def enable_llm_joint_guidance_for_natural_language(cfg, user_instruction):
    """在存在自然语言攻击需求时启用锚点联合引导。

    仅替换默认的 unguided 模式，保留用户显式指定的 default、manual 等实验配置。
    """
    has_instruction = any(str(item).strip() for item in (user_instruction or []))
    if not has_instruction:
        return False

    guidance = cfg.sim.traffic_model.guidance
    if str(guidance.mode).lower() != "unguided":
        return False

    guidance.mode = "llm_joint"
    return True

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
        # 保存逐场景评分，综合能力分严格取其简单平均。
        self.episode_results = []
        # 重置CARLA数据
    
    def update_running_statistics(self, info):
        """ Update running statistics with info from the latest episode."""
        self.collision.append(info['collision'])
        self.off_route.append(info['off_route'])
        self.completed.append(info['completed'])
        self.progress.append(info['progress'])

    
    def _metric_offsets(self):
        """记录当前累计指标边界，以便从全局统计中切出单场景数据。"""
        risk = self.generator.risk_metrics
        return {"collision": len(self.generator.collision_list), "near_miss": len(self.generator.near_miss_list), "ttc": len(self.generator.ttc_list), "ea": len(risk.ea_values), "reachability": len(risk.reachability_events)}

    def _build_episode_result(self, scenario_index, info, offsets):
        """计算单个场景的危险度与能力分，不使用跨场景聚合指标。"""
        risk = self.generator.risk_metrics
        ea_values = risk.ea_values[offsets["ea"]:]
        events = risk.reachability_events[offsets["reachability"]:]
        ttc_values = self.generator.ttc_list[offsets["ttc"]:]
        finite_events = [event for event in events if np.isfinite(event["difficulty"])]
        metrics = {"collision rate": float(bool(info.get("collision", False))), "off route rate": float(bool(info.get("off_route", False))), "completed rate": float(bool(info.get("completed", False))), "progress": float(np.clip(info.get("progress", 0.0), 0.0, 1.0)), "collision_rate": float(np.mean(self.generator.collision_list[offsets["collision"]:])) if len(self.generator.collision_list) > offsets["collision"] else 0.0, "near_miss_rate": float(np.mean(self.generator.near_miss_list[offsets["near_miss"]:])) if len(self.generator.near_miss_list) > offsets["near_miss"] else 0.0, "avg_min_ttc": float(np.mean(ttc_values)) if ttc_values else float("inf"), "max_evasive_acceleration_mps2": float(max(ea_values)) if ea_values else 0.0, "mean_evasive_acceleration_mps2": float(np.mean(ea_values)) if ea_values else 0.0, "reachability_event_count": len(events)}
        if finite_events:
            hardest = max(finite_events, key=lambda event: event["difficulty"])
            metrics.update({"reachability_difficulty": float(hardest["difficulty"]), "mean_reachability_difficulty": float(np.mean([event["difficulty"] for event in finite_events])), "dangerous_scene_solvable": float(hardest["dangerous_solvable"])})
        danger = compute_scenario_danger_score(metrics, getattr(getattr(self.cfg, "evaluation", None), "composite", None))
        ability_cfg = getattr(getattr(self.cfg, "evaluation", None), "autonomous_driving", None)
        coefficient = float(np.clip(getattr(ability_cfg, "partial_progress_coefficient", 0.5), 0.0, 1.0))
        complete = bool(metrics["completed rate"] and not metrics["collision rate"] and not metrics["off route rate"])
        partial_credit = 0.0 if complete else coefficient * metrics["progress"]
        return {"scenario_index": int(scenario_index), "scenario_danger_score": float(danger), "complete_success": complete, "progress": metrics["progress"], "partial_progress_coefficient": coefficient, "partial_credit": float(partial_credit), "autonomous_driving_ability_score": float(100.0 * danger * (float(complete) + partial_credit)), "collision": bool(metrics["collision rate"]), "off_route": bool(metrics["off route rate"]), "completed": bool(metrics["completed rate"]), "attack_plan_count": int(self.generator._episode_attack_plan_count), "attack_active_frames": int(self.generator._episode_attack_active_frames), "obstacle_plan_count": int(self.generator._episode_obstacle_plan_count), "video_path": str(Path(str(self.cfg.movie_path)) / f"scenario_{scenario_index:03d}" / f"scenario_{scenario_index:03d}.mp4")}

    def _write_multi_scenario_results(self, all_metrics):
        """输出逐场景 JSON、CSV 与汇总图，供多场景模型能力对比使用。"""
        if not self.episode_results:
            return
        output_dir = Path(str(self.cfg.movie_path)); output_dir.mkdir(parents=True, exist_ok=True)
        json_path, csv_path, plot_path = output_dir / "multi_scenario_ability_results.json", output_dir / "multi_scenario_ability_results.csv", output_dir / "multi_scenario_ability_summary.png"
        # 先写入产物路径，使 JSON 本身也能作为多场景评测的完整索引。
        all_metrics.update({"multi_scenario_results_json": str(json_path), "multi_scenario_results_csv": str(csv_path), "multi_scenario_results_visualization": str(plot_path)})
        with json_path.open("w", encoding="utf-8") as handle: json.dump({"aggregate": all_metrics, "episodes": self.episode_results}, handle, ensure_ascii=False, indent=2)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.episode_results[0])); writer.writeheader(); writer.writerows(self.episode_results)
        generate_multi_scenario_evaluation_visualization(self.episode_results, str(plot_path))
        all_metrics.update({"multi_scenario_results_json": str(json_path), "multi_scenario_results_csv": str(csv_path), "multi_scenario_results_visualization": str(plot_path)})

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
        # 综合能力分为所有场景独立表现分的简单算术平均。
        if self.episode_results:
            scores = np.asarray([item["autonomous_driving_ability_score"] for item in self.episode_results], dtype=float)
            all_metrics["autonomous_driving_ability_score"] = float(scores.mean())
            all_metrics["evaluated_scenario_count"] = int(len(scores))
            all_metrics["configured_max_scenarios"] = int(getattr(getattr(self.cfg, "evaluation", None), "max_scenarios", 0))
            all_metrics["mean_per_scene_danger_score"] = float(np.mean([item["scenario_danger_score"] for item in self.episode_results]))
            all_metrics["danger_weighted_complete_success"] = float(np.mean([item["scenario_danger_score"] * float(item["complete_success"]) for item in self.episode_results]))
            all_metrics["danger_weighted_partial_progress"] = float(np.mean([item["scenario_danger_score"] * item["partial_credit"] for item in self.episode_results]))
        else:
            all_metrics["autonomous_driving_ability_score"] = 0.0
            all_metrics["evaluated_scenario_count"] = 0
            all_metrics["configured_max_scenarios"] = int(getattr(getattr(self.cfg, "evaluation", None), "max_scenarios", 0))
        return all_metrics, ["{}: {:.6f}".format(k,v) for (k,v) in all_metrics.items()]

    def evaluate_policy(self):
        """ Evaluate the policy over all test scenarios in the environment."""
        self.reset()
        
        # 遍历所有测试场景
        max_scenarios = int(getattr(getattr(self.cfg, "evaluation", None), "max_scenarios", 0))
        scenario_count = self.env.num_test_scenarios if max_scenarios <= 0 else min(max_scenarios, self.env.num_test_scenarios)
        for i in tqdm(range(scenario_count)):
            print(f"Simulating environment {i}")
            obs = self.env.reset(i)

            # 记录累计指标边界；后续只使用本场景产生的数据计算独立得分。
            metric_offsets = self._metric_offsets()
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
                    # 每个场景独立目录，防止视频合成时混入其它场景的 PNG 帧。
                    scenario_movie_dir = os.path.join(str(self.cfg.movie_path), f"scenario_{i:03d}")
                    self.env.render_state(name=f'scenario_{i:03d}', movie_path=scenario_movie_dir)
                
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
            self.episode_results.append(self._build_episode_result(i, info, metric_offsets))
            
            if self.cfg.visualize:
                scenario_movie_dir = os.path.join(str(self.cfg.movie_path), f"scenario_{i:03d}")
                generate_video(name=f'scenario_{i:03d}', output_dir=scenario_movie_dir, delete_images=False)
            
            if self.cfg.verbose:
                if self.cfg.behaviour_model.compute_metrics and self.env.behaviour_model is not None:
                    print("behaviour model metrics: ", self.env.behaviour_model.compute_metrics()[-1])
                # policy metrics
                print(self.compute_metrics()[-1])


        metrics, _ = self.compute_metrics()
        self._write_multi_scenario_results(metrics)
        return metrics, ["{}: {:.6f}".format(k, v) if isinstance(v, (int, float, np.floating)) else "{}: {}".format(k, v) for (k, v) in metrics.items()]

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

    if enable_llm_joint_guidance_for_natural_language(cfg, user_instruction):
        print("[自然语言攻击] 已自动启用 sim.traffic_model.guidance.mode=llm_joint。")

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
