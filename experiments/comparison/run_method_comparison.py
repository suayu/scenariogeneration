"""在固定场景、种子与帧预算下统一比较危险场景生成方法。"""

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
from hydra import compose, initialize

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from experiments.comparison.methods import METHODS
from policies.idm_policy import IDMPolicy
from policies.risk_metrics import compute_scenario_danger_score
from scenario_generator import AdversarialScenarioGenerator
from simulator import Simulator
from utils.sim_helpers import ego_progress
from tests.run_full_adversarial_evaluation import (
    TEST_INSTRUCTION, background_collision_count, choose_scenarios, generate_video,
    min_center_distance_and_ttc, set_seed,
)


class PassiveGenerator:
    """基线不调用 LLM，仅满足统一闭环记录接口。"""
    diffusion_trajectory = None
    _llm_call_times = 0
    _episode_attack_reasons = ()
    _episode_attack_difficulties = ()

    def reset_episode_stats(self):
        return None

    def step(self, env, current_t):
        return True

    def set_diffusion_trajectory(self, trajectory):
        self.diffusion_trajectory = trajectory

    def episode_difficulty(self):
        return 0.0


def run_episode(env, policy, generator, scenario_index, steps, frame_dir):
    """统一运行 Safe-Sim 扩散或 Scenario-Dreamer log-replay 场景。"""
    obs = env.reset(scenario_index)
    generator.reset_episode_stats()
    policy.reset(obs)
    min_distance, min_ttc = float("inf"), float("inf")
    planner_times, prepare_times, environment_step_times = [], [], []
    background_frames, background_agents = 0, 0
    terminal = {"collision": False, "off_route": False, "completed": False, "progress": 0.0}
    guidance_frames, attack_target_id = 0, None
    for _ in range(steps):
        start = time.perf_counter()
        generator.step(env, env.current_step)
        planner_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        joint = env.prepare_background_traffic()
        prepare_times.append(time.perf_counter() - start)
        if joint is not None and joint.metadata.get("adversarial_guidance_active"):
            guidance_frames += 1
        if env.attack_intent is not None:
            attack_target_id = int(env.attack_intent["target_id"])
        selected = env.get_attack_target_prediction()
        generator.set_diffusion_trajectory(selected)
        env.render_state(name=f"scenario_{scenario_index:03d}", movie_path=str(frame_dir))
        action = policy.act(obs)
        anchors = None if env.attack_intent is None else env.attack_intent["anchors"]
        start = time.perf_counter()
        obs, terminated, info = env.step(action, anchors, selected)
        environment_step_times.append(time.perf_counter() - start)
        distance, ttc = min_center_distance_and_ttc(env.ego_state, env.data_dict["agent"][-1], env.agent_active)
        min_distance, min_ttc = min(min_distance, distance), min(min_ttc, ttc)
        count = background_collision_count(env.data_dict["agent"][-1], env.agent_active)
        background_frames += int(count > 0)
        background_agents += count
        if info:
            terminal = info
        if terminated:
            break
    if not terminal.get("progress"):
        terminal["progress"] = ego_progress(
            env.local_frame["center"], env.scenario_dict["route"]
        )
    route_progress_m = max(float(terminal.get("progress", 0.0)), 0.0)
    route_points = np.asarray(env.scenario_dict["route"], dtype=np.float64)
    route_length_m = float(np.linalg.norm(np.diff(route_points[:, :2], axis=0), axis=1).sum())
    # ego_progress 返回米数，能力分中的 P 需使用完成比例。
    terminal["progress"] = float(np.clip(
        route_progress_m / max(route_length_m, 1e-6), 0.0, 1.0
    ))
    near_miss = bool(np.isfinite(min_ttc) and min_ttc < 1.5 and not terminal.get("collision", False))
    common_metrics = {
        "collision rate": float(bool(terminal.get("collision", False))),
        "near_miss_rate": float(near_miss),
        "avg_min_ttc": float(min_ttc),
        "off route rate": float(bool(terminal.get("off_route", False))),
        "completed rate": float(bool(terminal.get("completed", False))),
        "progress": float(terminal.get("progress", 0.0)),
    }
    danger = compute_scenario_danger_score(common_metrics, env.cfg.sim.evaluation.composite)
    complete = bool(
        terminal.get("completed", False)
        and not terminal.get("collision", False)
        and not terminal.get("off_route", False)
    )
    ability_cfg = env.cfg.sim.evaluation.autonomous_driving
    coefficient = float(np.clip(ability_cfg.partial_progress_coefficient, 0.0, 1.0))
    partial_credit = 0.0 if complete else coefficient * common_metrics["progress"]
    video = generate_video(f"scenario_{scenario_index:03d}", str(frame_dir), delete_images=False)
    return {
        "scenario_index": int(scenario_index), "scenario_id": str(env.current_scene_id),
        "executed_steps": len(prepare_times), "llm_call_count": int(generator._llm_call_times),
        "attack_accepted": attack_target_id is not None, "attack_target_id": attack_target_id,
        "static_obstacle_count": len(env.get_static_obstacles()), "guidance_active_frames": guidance_frames,
        "collision": bool(terminal.get("collision", False)), "off_route": bool(terminal.get("off_route", False)),
        "completed": bool(terminal.get("completed", False)), "progress": float(terminal.get("progress", 0.0)),
        "route_progress_m": route_progress_m, "route_length_m": route_length_m,
        "near_miss": near_miss, "scenario_danger_score": float(danger),
        "autonomous_driving_ability_score": float(
            100.0 * danger * (float(complete) + partial_credit)
        ),
        "min_center_distance_m": float(min_distance), "min_ttc_s": float(min_ttc),
        "background_collision_frames": background_frames, "background_colliding_agent_count": background_agents,
        "mean_planner_seconds": float(np.mean(planner_times)),
        "mean_prepare_seconds": float(np.mean(prepare_times)),
        "mean_environment_step_seconds": float(np.mean(environment_step_times)),
        "mean_step_generation_seconds": float(np.mean(
            np.asarray(planner_times) + np.asarray(prepare_times) + np.asarray(environment_step_times)
        )),
        "video_path": str(video),
        "attack_reasons": list(generator._episode_attack_reasons),
    }


def aggregate(records):
    """使用所有方法共有的闭环指标生成同口径汇总。"""
    mean = lambda key: float(np.mean([float(row[key]) for row in records]))
    return {
        "collision_rate": mean("collision"), "off_route_rate": mean("off_route"),
        "completion_rate": mean("completed"), "mean_progress": mean("progress"),
        "attack_acceptance_rate": mean("attack_accepted"),
        "obstacle_deployment_rate": float(np.mean([row["static_obstacle_count"] > 0 for row in records])),
        "background_collision_episode_rate": float(np.mean([row["background_collision_frames"] > 0 for row in records])),
        "mean_min_ttc_s": mean("min_ttc_s"), "mean_min_center_distance_m": mean("min_center_distance_m"),
        "mean_generation_seconds": mean("mean_step_generation_seconds"),
        "near_miss_rate": mean("near_miss"),
        "mean_scenario_danger_score": mean("scenario_danger_score"),
        "mean_autonomous_driving_ability_score": mean("autonomous_driving_ability_score"),
    }


def bootstrap_statistics(records, seed=0, bootstrap_samples=2000):
    """输出核心指标的均值、P50/P95 和 bootstrap 95% 置信区间。"""
    metric_names = (
        "collision", "near_miss", "scenario_danger_score",
        "autonomous_driving_ability_score", "min_ttc_s",
        "background_collision_frames", "mean_step_generation_seconds",
    )
    rng = np.random.default_rng(seed)
    statistics = {}
    for metric_name in metric_names:
        values = np.asarray([float(row[metric_name]) for row in records], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            statistics[metric_name] = {
                "mean": None, "p50": None, "p95": None, "bootstrap_ci95": [None, None]
            }
            continue
        indices = rng.integers(0, len(finite), size=(bootstrap_samples, len(finite)))
        bootstrap_means = finite[indices].mean(axis=1)
        statistics[metric_name] = {
            "mean": float(finite.mean()),
            "p50": float(np.percentile(finite, 50)),
            "p95": float(np.percentile(finite, 95)),
            "bootstrap_ci95": [
                float(np.percentile(bootstrap_means, 2.5)),
                float(np.percentile(bootstrap_means, 97.5)),
            ],
        }
    return statistics


def plot_summary(results, path):
    """绘制三方法核心风险—可行性对比图。"""
    names = list(results)
    metrics = ["collision_rate", "near_miss_rate", "mean_scenario_danger_score", "background_collision_episode_rate"]
    values = np.asarray([[results[name]["aggregate"][key] for key in metrics] for name in names])
    x = np.arange(len(metrics)); width = 0.8 / len(names)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axis = axes[0]
    for index, name in enumerate(names):
        axis.bar(x + (index - (len(names)-1)/2)*width, values[index], width, label=name)
    axis.set_xticks(x, metrics, rotation=12); axis.set_ylim(0, 1.05); axis.legend(); axis.grid(axis="y", alpha=.25)
    axis.set_title("风险效果与背景交通可行性")
    for name in names:
        aggregate_data = results[name]["aggregate"]
        axes[1].scatter(
            aggregate_data["background_collision_episode_rate"],
            aggregate_data["mean_scenario_danger_score"],
            s=110,
            label=name,
        )
    axes[1].set_xlabel("背景车碰撞场景率（越低越好）")
    axes[1].set_ylabel("平均危险度（越高越强）")
    axes[1].set_xlim(-.03, 1.03); axes[1].set_ylim(-.03, 1.03)
    axes[1].grid(alpha=.25); axes[1].legend(); axes[1].set_title("风险—可行性权衡")
    figure.tight_layout(); figure.savefig(path, dpi=180); plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="ours,safe_sim,scenario_dreamer")
    parser.add_argument("--scenario-indices", default="")
    parser.add_argument("--scenarios", type=int, default=2)
    parser.add_argument("--steps", type=int, default=45)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    scenarios = [int(item) for item in args.scenario_indices.split(",") if item.strip()]
    unknown = [name for name in methods if name not in METHODS]
    if unknown: raise ValueError(f"unknown methods: {unknown}; registered={sorted(METHODS)}")
    root = Path(args.output_dir).resolve(); root.mkdir(parents=True, exist_ok=True)
    results = {}
    for method_name in methods:
        spec = METHODS[method_name]
        overrides = ["sim.visualize=false", f"sim.steps={args.steps}", *spec.overrides]
        with initialize(version_base=None, config_path="../../cfgs"):
            cfg = compose(config_name="config", overrides=overrides)
        set_seed(int(cfg.sim.seed))
        env = Simulator(cfg)
        # os.listdir 的顺序不是稳定协议，对比运行器显式固定场景索引到文件的映射。
        env.test_files = sorted(env.test_files)
        policy = IDMPolicy(cfg, env)
        if not scenarios:
            # 只用初始几何关系选场景，不根据任何方法的结果挑选。
            scenarios = choose_scenarios(env, args.scenarios)
        generator = AdversarialScenarioGenerator(cfg, TEST_INSTRUCTION, llm_planner=env.llm_planner) if spec.uses_llm else PassiveGenerator()
        records = [run_episode(env, policy, generator, idx, args.steps, root / method_name / f"scenario_{idx:03d}") for idx in scenarios]
        results[method_name] = {
            "fidelity": spec.fidelity,
            "overrides": list(spec.overrides),
            "aggregate": aggregate(records),
            "statistics": bootstrap_statistics(records, seed=int(cfg.sim.seed)),
            "episodes": records,
        }
        method_dir = root / method_name; method_dir.mkdir(parents=True, exist_ok=True)
        (method_dir / "config_snapshot.yaml").write_text(
            OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8"
        )
        with (method_dir / "results.json").open("w", encoding="utf-8") as handle: json.dump(results[method_name], handle, ensure_ascii=False, indent=2)
        with (method_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    plot_summary(results, root / "method_comparison.png")
    with (root / "comparison_summary.json").open("w", encoding="utf-8") as handle: json.dump(results, handle, ensure_ascii=False, indent=2)
    print(json.dumps({name: data["aggregate"] for name, data in results.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
