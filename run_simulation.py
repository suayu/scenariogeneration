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
import time
from pathlib import Path
from scenario_generator import AdversarialScenarioGenerator
from policies.risk_metrics import compute_scenario_danger_score
from policies.difficulty_control import ability_score
from policies.difficulty_audit import summarize_attempts, json_safe
from policies.joint_safety import NoSafeJointCandidate
from policies.route_progress import normalized_route_progress
from policies.evaluation_trace import AttemptTrace
from policies.planner_smoke_gate import validate_planner_smoke


def enable_llm_joint_guidance_for_natural_language(cfg, user_instruction):
    """在存在自然语言攻击需求时启用锚点联合引导。

    常规模式仅替换 unguided；画像开启时启用完整锚点联合引导链路。
    """
    has_instruction = any(str(item).strip() for item in (user_instruction or []))
    iterative = getattr(cfg.sim.traffic_model, 'iterative_adversarial', None)
    if bool(getattr(iterative, 'profile_enabled', False)):
        # 单开关启用完整画像攻击，不能在无锚点引导模式下静默降级。
        cfg.sim.traffic_model.guidance.mode = 'llm_joint'
        return True
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
            env.cfg,
            user_instruction,
            llm_planner=env.llm_planner,
        )
        if bool(getattr(self.cfg.evaluation.difficulty_control, "smoke_acceptance", False)):
            g = self.generator
            if not (g.difficulty_mode == "target" and abs(g.target_difficulty-.5) < 1e-9
                    and abs(g.difficulty_tolerance-.1) < 1e-9
                    and g.profile_enabled):
                raise RuntimeError("冒烟失败：实际生成器配置未满足 target=0.50/tolerance=0.10/画像攻击")
            if list(env.diffusion_controller.active_guidance_functions) != ["scenario_collision","route","scenario_ttc","llm_anchor"]:
                raise RuntimeError("冒烟失败：联合引导项不完整")

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
        # 全场旧聚合只保留为诊断；目标控制和能力分共用窗口加权危险度。
        legacy_danger = compute_scenario_danger_score(metrics, getattr(getattr(self.cfg, "evaluation", None), "composite", None))
        measured = self.generator.episode_difficulty()
        danger = float(measured) if measured is not None else float("nan")
        ability_cfg = getattr(getattr(self.cfg, "evaluation", None), "autonomous_driving", None)
        coefficient = float(np.clip(getattr(ability_cfg, "partial_progress_coefficient", 0.5), 0.0, 1.0))
        complete = bool(metrics["completed rate"] and not metrics["collision rate"] and not metrics["off route rate"])
        partial_credit = 0.0 if complete else coefficient * metrics["progress"]
        score = ability_score(measured, complete, bool(metrics["collision rate"]), bool(metrics["off route rate"]), metrics["progress"], coefficient)
        return {"scenario_index": int(scenario_index), "scenario_danger_score": danger,
                "danger_valid": measured is not None, "danger_basis": "frame_weighted_observation_windows",
                "legacy_full_scene_danger_diagnostic": float(legacy_danger),
                "complete_success": complete, "progress": metrics["progress"],
                "partial_progress_coefficient": coefficient, "partial_credit": float(partial_credit),
                "autonomous_driving_ability_score": float(score) if score is not None else float("nan"),
                "collision": bool(metrics["collision rate"]), "off_route": bool(metrics["off route rate"]),
                "completed": bool(metrics["completed rate"]),
                "attack_plan_count": int(self.generator._episode_attack_plan_count),
                "attack_active_frames": int(self.generator._episode_attack_active_frames),
                "obstacle_plan_count": int(self.generator._episode_obstacle_plan_count),
                "video_path": str(Path(str(self.cfg.movie_path)) / f"scenario_{scenario_index:03d}" / f"scenario_{scenario_index:03d}.mp4")}

    def _rollback_attempt_metrics(self, offsets):
        """目标难度重放前撤销失败尝试的累计统计，避免重复计权。"""
        del self.generator.collision_list[offsets["collision"]:]
        del self.generator.near_miss_list[offsets["near_miss"]:]
        del self.generator.ttc_list[offsets["ttc"]:]
        del self.generator.risk_metrics.ea_values[offsets["ea"]:]
        del self.generator.risk_metrics.reachability_events[offsets["reachability"]:]

    def _run_scenario_attempt(self, scenario_index, replay_index, target_mode):
        """从场景起点执行一次完整闭环；重放不增加额外校准攻击窗口。"""
        obs = self.env.reset(scenario_index)
        self.generator.reset_episode_stats(advance_instruction=replay_index == 0)
        # 重置风险算法回合状态后再读取边界，避免旧长度截断新回合数据。
        metric_offsets = self._metric_offsets()
        if hasattr(self.policy, 'reset'):
            self.policy.reset(obs)

        scenario_movie_dir = Path(str(self.cfg.movie_path)) / f"scenario_{scenario_index:03d}"
        if target_mode:
            # 每次重放单独保存，防止提前终止尝试遗留的帧混入最终视频。
            scenario_movie_dir = scenario_movie_dir / f"attempt_{replay_index}"

        trace = AttemptTrace(scenario_movie_dir)
        self.generator.query_trace = trace
        trace.state(self.env, initial=True)
        profile_pipeline = self.generator.profile_pipeline
        # 最小配置及默认关闭场景没有 traffic_model 节点，此时不启用投影。
        projection_cfg = getattr(getattr(self.cfg, "traffic_model", None), "dynamics_projection", None)
        projection_enabled = bool(getattr(projection_cfg, "enabled", False))
        if projection_enabled and profile_pipeline is None:
            raise ValueError("轨迹投影需要画像硬门禁完成全时域安全复核")
        if profile_pipeline is not None:
            profile_pipeline.begin_attempt(self.env, scenario_movie_dir)

        info = {"collision": False, "off_route": False, "completed": False, "progress": 0.0}
        for _ in range(self.env.steps):
            current_t = self.env.current_step
            planning_start = time.monotonic()
            self.generator.step(self.env, current_t)
            profile_attack_active = profile_pipeline is not None and (self.env.attack_intent is not None or bool(self.env.get_static_obstacles()))
            # 每次预测前应用当前强度，包含无攻击、LLM 失败与刚结算窗口的路径。
            if self.generator.difficulty_mode != 'off':
                self.generator._apply_iterative_stage(self.env)
            # 可选的攻击期道路软引导只影响连续预测；攻击结束即恢复场景基线权重。
            route_audit = None
            traffic_cfg = getattr(self.cfg, 'traffic_model', None)
            if (getattr(traffic_cfg, 'attack_route_weight', None) is not None
                    or getattr(traffic_cfg, 'attack_route_lane_margin', None) is not None):
                route_audit = self.generator.apply_attack_route_weight(self.env, self.env.attack_intent is not None)
            if route_audit is not None and route_audit['changed']:
                trace.append(dict(kind='attack_route_guidance', step=int(current_t), **route_audit))
            raw_joint_for_audit = None
            try:
                self.env.prepare_background_traffic()
                raw_joint_for_audit = self.env.pending_joint_trajectory
                if profile_attack_active:
                    if projection_enabled:
                        from policies.trajectory_projection import project_joint
                        corrected = project_joint(self.env.pending_joint_trajectory,
                            np.asarray(self.env.data_dict["agent"][-1], float), self.env.dt,
                            profile_pipeline.dynamic_limits, projection_cfg,
                            getattr(self.env, 'previously_attacked_agent_ids', ()))
                        self.env.inject_joint_trajectory(corrected)
                        trace.append(dict(kind="dynamics_projection", step=int(self.env.current_step),
                                          audit=corrected.metadata["dynamics_projection"]))
                    profile_pipeline.validate_joint(self.env)
            except NoSafeJointCandidate as error:
                from policies.rejection_audit import rejection_record
                trace.append(rejection_record(self.env, profile_pipeline, str(error), raw_joint_for_audit))
                # 保留生成失败状态，禁止把拒绝不安全候选计为无碰撞成功。
                info = dict(collision=False, off_route=False, progress=0.0, **{
                    k:v for k,v in info.items() if k not in {"collision","off_route","progress"}})
                info.update(generation_failure=str(error),completed=False)
                if profile_pipeline is not None and profile_pipeline.pending:
                    profile_pipeline.pending['generation_failure_reason'] = str(error)
                if self.cfg.visualize:
                    self.env.render_state(name=f'scenario_{scenario_index:03d}', movie_path=str(scenario_movie_dir))
                break
            trace.prediction(self.env, self.generator, time.monotonic() - planning_start)
            if profile_attack_active:
                # 每个画像攻击执行前检查实际联合预测，而不沿用上次候选的规避判断。
                if not self.generator._reachability_query_pending:
                    self.generator.risk_metrics.begin_attack(self.env)
            reachability_event = self.generator.evaluate_reachability(self.env)
            if profile_attack_active:
                if (not reachability_event or not np.isfinite(reachability_event['difficulty'])
                        or not reachability_event['dangerous_solvable']):
                    reason = ('profile_reachability_invalid' if not reachability_event
                              or not np.isfinite(reachability_event['difficulty'])
                              else 'profile_zero_theoretical_drivable_area')
                    info.update(generation_failure=reason,completed=False)
                    if profile_pipeline.pending:
                        profile_pipeline.pending['generation_failure_reason'] = reason
                    trace.append({'kind':'profile_execution_rejected','step':int(self.env.current_step),'reason':reason})
                    if self.cfg.visualize:
                        self.env.render_state(name=f'scenario_{scenario_index:03d}',movie_path=str(scenario_movie_dir))
                    break
            self.generator.set_diffusion_trajectory(
                self.env.get_attack_target_prediction()
            )
            anchors, refined_traj = self.generator.get_anchors_and_trajectory()

            if self.cfg.visualize:
                self.env.render_state(
                    name=f'scenario_{scenario_index:03d}',
                    movie_path=str(scenario_movie_dir),
                )

            action = self.policy.act(obs)
            # 执行前冻结联合预测；执行后核对真实坐标，不以意图存在代替执行。
            execution_joint = self.env.pending_joint_trajectory
            obs, terminated, info = self.env.step(action, anchors, refined_traj)
            execution_audit = trace.attack_execution(self.env, execution_joint)
            self.generator.evaluate_reaction(self.env, info)
            trace.state(self.env, info)
            if profile_pipeline is not None:
                profile_pipeline.executed(self.env, execution_audit, info, trace)
            if terminated:
                # 保存终止后的真实状态，使完整视频包含最终碰撞或完成帧。
                if self.cfg.visualize:
                    self.env.render_state(name=f'scenario_{scenario_index:03d}', movie_path=str(scenario_movie_dir))
                self.env.dump_step_data(scenario_index)
                break
            print("step:", current_t, " of ", self.env.steps)

        self.generator.finalize_episode_stats()
        attack_difficulty = self.generator.finalize_episode_difficulty()
        progress = normalized_route_progress(self.env.ego_state[:2], self.env.scenario_dict["route"])
        if progress is None:
            raise ValueError("路线为空或退化，无法计算真实能力分进度")
        info["progress"] = progress
        result = self._build_episode_result(scenario_index, info, metric_offsets)
        result["generation_failure"] = info.get("generation_failure")
        result['feasibility'] = trace.summary()
        result['attack_executed_frames'] = trace.attack_executed_frames
        result['planner_service_failure_count'] = self.generator._episode_planner_service_failures
        if result['planner_service_failure_count']:
            result['generation_failure'] = 'planner_service_failure'
        if profile_pipeline is not None:
            result['ego_profile'] = profile_pipeline.finish_attempt(self.env, result['generation_failure'])
        result['scenario_source'] = str(self.env.test_files[scenario_index])
        result['configured_steps'] = int(self.env.steps)
        result.update({
            "attack_difficulty": attack_difficulty,
            "attack_window_difficulties": list(
                self.generator._episode_attack_difficulties
            ),
            "attack_reasons": list(self.generator._episode_attack_reasons),
            "difficulty_target": float(self.generator.target_difficulty),
            "difficulty_tolerance": float(self.generator.difficulty_tolerance),
            "replay_count": int(replay_index),
            "executed_steps": int(self.env.current_step),
            "llm_call_count": int(self.generator._llm_call_times),
            "actual_controls": {
                "mode": self.generator.difficulty_mode,
                "target": self.generator.target_difficulty,
                "tolerance": self.generator.difficulty_tolerance,
                "profile": self.generator.profile_enabled,
                "escalation": self.generator.escalation_enabled,
                "full_method": self.generator.full_method_enabled,
                "guidance_functions": list(self.env.diffusion_controller.active_guidance_functions),
                "final_parameters": self.generator.difficulty_controller.parameters(self.generator._iterative_stage),
                "bounds": self.generator.difficulty_controller.bounds,
            },
        })
        # 即使冒烟验收失败，也先保存完整诊断与视频，避免失败原因被总括错误掩盖。
        if self.cfg.visualize:
            video_path = generate_video(name=f'scenario_{scenario_index:03d}',
                                        output_dir=str(scenario_movie_dir), delete_images=False)
            result["video_path"] = str(video_path or (scenario_movie_dir / f"scenario_{scenario_index:03d}.mp4"))
        scenario_movie_dir.mkdir(parents=True, exist_ok=True)
        (scenario_movie_dir / "attempt_result.json").write_text(
            json.dumps(json_safe(result), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        if bool(getattr(self.cfg.evaluation.difficulty_control, "smoke_acceptance", False)):
            if getattr(self.env.llm_planner, 'provider', None) == 'codex':
                validate_planner_smoke(result)
            valid_windows = [w for w in result["attack_window_difficulties"]
                             if w.get("valid") and w.get("ea_count",0)>0 and w.get("reachability_count",0)>0]
            if not valid_windows or not result["danger_valid"] or result["generation_failure"]:
                raise RuntimeError("冒烟失败：缺少 EA/可达集有效窗口或存在生成失败")
            if result['feasibility']['background_collision_frames'] or result['feasibility']['background_static_collision_frames']:
                raise RuntimeError('冒烟失败：实际执行存在背景车碰撞')
            if not all("control" in w for w in valid_windows):
                raise RuntimeError("冒烟失败：缺少控制调整前后参数与原因")
            expected = ability_score(attack_difficulty,result["completed"],result["collision"],result["off_route"],result["progress"],result["partial_progress_coefficient"])
            if expected is None or not np.isclose(expected,result["autonomous_driving_ability_score"]):
                raise RuntimeError("冒烟失败：能力分与控制危险度不一致")
        return info, metric_offsets, result, attack_difficulty

    def _write_multi_scenario_results(self, all_metrics):
        """输出逐场景 JSON、CSV 与汇总图，供多场景模型能力对比使用。"""
        if not self.episode_results:
            return
        output_dir = Path(str(self.cfg.movie_path)); output_dir.mkdir(parents=True, exist_ok=True)
        json_path, csv_path, plot_path = output_dir / "multi_scenario_ability_results.json", output_dir / "multi_scenario_ability_results.csv", output_dir / "multi_scenario_ability_summary.png"
        # 先写入产物路径，使 JSON 本身也能作为多场景评测的完整索引。
        all_metrics.update({"multi_scenario_results_json": str(json_path), "multi_scenario_results_csv": str(csv_path), "multi_scenario_results_visualization": str(plot_path)})
        with json_path.open("w", encoding="utf-8") as handle: json.dump(json_safe({"aggregate": all_metrics, "episodes": self.episode_results}), handle, ensure_ascii=False, indent=2, allow_nan=False)
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
        all_metrics["legacy_full_scene_danger_diagnostic"] = compute_scenario_danger_score(
            all_metrics,
            composite_cfg,
        )
        # 综合能力分为所有场景独立表现分的简单算术平均。
        if self.episode_results:
            scores = np.asarray([item["autonomous_driving_ability_score"] for item in self.episode_results], dtype=float)
            all_metrics["autonomous_driving_ability_score"] = float(scores.mean())
            all_metrics["evaluated_scenario_count"] = int(len(scores))
            all_metrics["generation_failure_count"] = sum(bool(item.get("generation_failure")) for item in self.episode_results)
            all_metrics["invalid_danger_count"] = sum(not item.get("danger_valid",False) for item in self.episode_results)
            all_metrics["configured_max_scenarios"] = int(getattr(getattr(self.cfg, "evaluation", None), "max_scenarios", 0))
            all_metrics["mean_per_scene_danger_score"] = float(np.mean([item["scenario_danger_score"] for item in self.episode_results]))
            all_metrics["scenario_danger_score"] = all_metrics["mean_per_scene_danger_score"]
            all_metrics["difficulty_audit"] = summarize_attempts(self.episode_results)
            all_metrics["danger_weighted_complete_success"] = float(np.mean([item["scenario_danger_score"] * float(item["complete_success"]) for item in self.episode_results]))
            all_metrics["danger_weighted_partial_progress"] = float(np.mean([item["scenario_danger_score"] * item["partial_credit"] for item in self.episode_results]))
            difficulty_errors = np.asarray([
                item.get("difficulty_absolute_error", np.nan)
                for item in self.episode_results
            ], dtype=float)
            finite_difficulty = np.isfinite(difficulty_errors)
            all_metrics["difficulty_target_hit_rate"] = float(np.mean([
                item.get("difficulty_control_status") == "accepted"
                for item in self.episode_results
            ]))
            all_metrics["difficulty_mean_absolute_error"] = float(
                np.mean(difficulty_errors[finite_difficulty])
            ) if finite_difficulty.any() else float("nan")
            all_metrics["difficulty_uncontrollable_count"] = int(sum(
                item.get("difficulty_control_status") == "uncontrollable"
                for item in self.episode_results
            ))
        else:
            all_metrics["autonomous_driving_ability_score"] = 0.0
            all_metrics["evaluated_scenario_count"] = 0
            all_metrics["configured_max_scenarios"] = int(getattr(getattr(self.cfg, "evaluation", None), "max_scenarios", 0))
        return all_metrics, ["{}: {:.6f}".format(k,v) if isinstance(v, (int, float, np.floating)) else "{}: {}".format(k,v) for (k,v) in all_metrics.items()]

    def evaluate_policy(self):
        """ Evaluate the policy over all test scenarios in the environment."""
        self.reset()

        # 遍历所有测试场景
        max_scenarios = int(getattr(getattr(self.cfg, "evaluation", None), "max_scenarios", 0))
        scenario_count = self.env.num_test_scenarios if max_scenarios <= 0 else min(max_scenarios, self.env.num_test_scenarios)
        difficulty_cfg = getattr(getattr(self.cfg, "evaluation", None), "difficulty_control", None)
        target_mode = str(getattr(difficulty_cfg, "mode", "off")).lower() == "target"
        max_replays = max(0, int(getattr(difficulty_cfg, "max_replays", 2)))
        for i in tqdm(range(scenario_count)):
            print(f"Simulating environment {i}")
            self.generator.reset_scene_replay_control()
            attempts = []
            for replay_index in range(max_replays + 1 if target_mode else 1):
                info, metric_offsets, result, difficulty = self._run_scenario_attempt(
                    i, replay_index, target_mode
                )
                valid = difficulty is not None and np.isfinite(difficulty)
                target_error = abs(difficulty - self.generator.target_difficulty) if valid else float("nan")
                accepted = valid and not result["generation_failure"] and (not target_mode or target_error <= self.generator.difficulty_tolerance)
                exhausted = not target_mode or replay_index >= max_replays
                attempts.append(dict(result, difficulty_absolute_error=target_error,
                                     accepted=bool(accepted), attempt_index=replay_index))
                if accepted or exhausted:
                    result["difficulty_control_status"] = (
                        "accepted" if accepted else "uncontrollable"
                    )
                    result["difficulty_absolute_error"] = float(target_error)
                    result["attempts"] = attempts
                    result["acceptance_reason"] = "target_hit" if accepted and target_mode else (
                        "valid_uncontrolled_run" if accepted else "no_safe_joint_candidate" if result["generation_failure"] else "missing_measurement" if not valid else "replay_budget_exhausted")
                    self.update_running_statistics(info)
                    self.episode_results.append(result)
                    # 每个场景即刻落盘，避免长实验中断后丢失全部尝试记录。
                    output_dir = Path(str(self.cfg.movie_path)) / f"scenario_{i:03d}"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    (output_dir / "attempts.json").write_text(json.dumps(json_safe(result),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
                    break

                print(
                    "[难度控制] 场景 "
                    f"{i} 难度={difficulty} 未命中 "
                    f"{self.generator.target_difficulty:.3f}±"
                    f"{self.generator.difficulty_tolerance:.3f}，从起点重放。"
                )
                self.generator.apply_scene_replay_feedback(difficulty, collision=bool(info.get("collision", False)))
                attempts[-1]["replay_feedback"] = getattr(self.generator, "_last_replay_feedback", None)
                self._rollback_attempt_metrics(metric_offsets)

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
    attack_request_file = str(getattr(cfg.sim, "attack_request_file", "attack_request.txt"))
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
