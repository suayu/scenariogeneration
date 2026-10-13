"""固定清单运行原始 SAFE-SIM 或 Scenario Dreamer 被动基线。"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

# 允许直接执行本文件，不要求调用者额外设置 PYTHONPATH。
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# 与正式启动脚本保持一致；调用方显式设置时不覆盖其数据根目录。
os.environ.setdefault("PROJECT_ROOT", str(ROOT))
os.environ.setdefault("SCRATCH_ROOT", str(ROOT))
os.environ.setdefault("DATASET_ROOT", str(ROOT / "metadata"))

import numpy as np
from hydra import compose, initialize
from omegaconf import OmegaConf

from experiments.comparison.formal_metrics import FormalRiskObserver
from experiments.comparison.methods import METHODS
from policies.difficulty_control import ability_score
from policies.evaluation_trace import AttemptTrace
from policies.idm_policy import IDMPolicy
from policies.joint_safety import NoSafeJointCandidate
from policies.route_progress import normalized_route_progress
from run_simulation import merge_episode_info
from simulator import Simulator
from tests.run_full_adversarial_evaluation import (
    background_collision_count,
    min_center_distance_and_ttc,
    set_seed,
)
from utils.viz import generate_video


PASSIVE_METHODS = {"safe_sim", "scenario_dreamer"}


def sanitize_nonfinite(value):
    """将嵌套结果中的非有限浮点测量转换为缺失值，保证严格 JSON 可写。"""
    if isinstance(value, dict):
        return {key: sanitize_nonfinite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_nonfinite(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_nonfinite(item) for item in value]
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    return value


def csv_fieldnames(rows):
    """按首次出现顺序汇总字段，兼容成功与生成失败记录。"""
    names = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                names.append(key)
                seen.add(key)
    return names


def validate_manifest(path):
    """验证冻结清单及每个场景内容哈希，防止静默换样本。"""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    files = [Path(item) for item in manifest["scenario_files"]]
    digests = manifest["scenario_sha256"]
    if len(files) != len(digests) or not files:
        raise ValueError("冻结清单文件与哈希数量不一致或为空")
    for file_path, expected in zip(files, digests):
        actual = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"场景哈希不一致：{file_path}")
    return manifest, files


def run_episode(env, policy, scenario_index, steps, output_dir, expect_joint, forecaster=None):
    """执行不含 RiskWeaver 高级攻击的原始方法闭环，并保存统一审计。"""
    obs = env.reset(scenario_index)
    policy.reset(obs)
    observer = FormalRiskObserver(env.cfg.sim.evaluation)
    observer.reset()
    trace = AttemptTrace(output_dir)
    trace.state(env, initial=True)
    terminal = {
        "collision": False,
        "off_route": False,
        "completed": False,
        "progress": 0.0,
    }
    min_distance, min_ttc = float("inf"), float("inf")
    background_collision_frames = 0
    executed_steps = 0
    for _ in range(steps):
        observer.begin_frame(env, expect_joint_prediction=expect_joint)
        joint = env.prepare_background_traffic()
        if joint is not None:
            trace.append({"kind": "execution_prediction", "step": int(env.current_step),
                          "joint": {"agent_ids": joint.agent_ids.tolist(),
                                    "positions_global": joint.positions_global.tolist(),
                                    "valid_mask": joint.valid_mask.tolist()}})
        forecast = None
        forecast_error = None
        if forecaster is not None:
            try:
                forecast = forecaster.predict(env, policy)
                if joint is not None and np.array_equal(joint.agent_ids, forecast.agent_ids) and len(joint.agent_ids):
                    delta = np.linalg.norm(joint.positions_global[:, 0] - forecast.positions_global[:, 0], axis=-1)
                    forecast.metadata['native_first_step_position_difference_mean_m'] = float(np.mean(delta))
                    forecast.metadata['native_first_step_position_difference_max_m'] = float(np.max(delta))
            except Exception as exc:
                # 预测服务失败不能自动切回旧支路或伪造有效 D；保留失败使冒烟门禁拒绝。
                forecast_error = type(exc).__name__ + ": " + str(exc)
            observer.finish_prediction(env, forecast, "ctrl_sim_rollout", forecast_error)
            trace.append({"kind": "evaluation_forecast", "step": int(env.current_step),
                          "metadata": None if forecast is None else forecast.metadata,
                          "validation_error": getattr(observer, "_prediction_error", None),
                          "joint": None if forecast is None else {
                              "agent_ids": forecast.agent_ids.tolist(),
                              "positions_global": forecast.positions_global.tolist(),
                              "yaws_global": forecast.yaws_global.tolist(),
                              "valid_mask": forecast.valid_mask.tolist()}})
        else:
            observer.finish_prediction(env)
        if env.cfg.sim.visualize:
            env.render_state(
                name=f"scenario_{scenario_index:03d}", movie_path=str(output_dir)
            )
        action = policy.act(obs)
        obs, terminated, step_info = env.step(action, None, None)
        terminal = merge_episode_info(terminal, step_info)
        distance, ttc = min_center_distance_and_ttc(
            env.ego_state, env.data_dict["agent"][-1], env.agent_active
        )
        min_distance, min_ttc = min(min_distance, distance), min(min_ttc, ttc)
        observer.finish_transition(step_info, ttc)
        trace.state(env, step_info)
        background_collision_frames += int(
            background_collision_count(
                env.data_dict["agent"][-1], env.agent_active
            ) > 0
        )
        executed_steps += 1
        if terminated:
            if env.cfg.sim.visualize:
                env.render_state(
                    name=f"scenario_{scenario_index:03d}", movie_path=str(output_dir)
                )
            env.dump_step_data(scenario_index)
            break

    progress = normalized_route_progress(
        env.ego_state[:2], env.scenario_dict["route"]
    )
    if progress is None:
        raise ValueError("路线为空或退化，无法计算进度")
    terminal["progress"] = progress
    danger = observer.result()
    complete = bool(
        terminal["completed"]
        and not terminal["collision"]
        and not terminal["off_route"]
    )
    coefficient = float(
        env.cfg.sim.evaluation.autonomous_driving.partial_progress_coefficient
    )
    score = ability_score(
        danger["danger"],
        complete,
        terminal["collision"],
        terminal["off_route"],
        progress,
        coefficient,
    )
    video = None
    if env.cfg.sim.visualize:
        video = generate_video(
            name=f"scenario_{scenario_index:03d}",
            output_dir=str(output_dir),
            delete_images=False,
        )
    finite_ttc = float(min_ttc) if np.isfinite(min_ttc) else None
    finite_distance = float(min_distance) if np.isfinite(min_distance) else None
    ea_values = [float(value) for value in observer.risk.ea_values if np.isfinite(value)]
    result = {
        "scenario_index": int(scenario_index),
        "scenario_source": str(env.test_files[scenario_index]),
        "executed_steps": int(executed_steps),
        "collision": bool(terminal["collision"]),
        "off_route": bool(terminal["off_route"]),
        "completed": bool(terminal["completed"]),
        "progress": float(progress),
        "min_ttc_s": finite_ttc,
        "min_center_distance_m": finite_distance,
        "max_evasive_acceleration_mps2": max(ea_values) if ea_values else None,
        "mean_evasive_acceleration_mps2": (
            float(np.mean(ea_values)) if ea_values else None
        ),
        "ea_valid_frame_count": len(ea_values),
        "scenario_danger_score": danger["danger"],
        "danger_valid": danger["valid"],
        "danger_basis": "common_per_frame_ea_and_aligned_joint_reachability",
        "reachability_source": "native" if forecaster is None else "ctrl_sim_rollout",
        "forecast_failure_count": sum(bool(r.get("prediction_error")) for r in danger["records"]),
        "danger_valid_frame_count": danger["valid_frame_count"],
        "danger_total_frame_count": danger["total_frame_count"],
        "danger_coverage": danger["coverage"],
        "danger_windows": danger["records"],
        "autonomous_driving_ability_score": score,
        "ability_valid": score is not None,
        "attack_plan_count": 0,
        "attack_executed_frames": 0,
        "obstacle_plan_count": 0,
        "planner_service_failure_count": 0,
        "background_collision_frames_crosscheck": int(background_collision_frames),
        "feasibility": trace.summary(),
        "video_path": str(video) if video is not None else None,
    }
    # 风险观测器可能用正无穷表示“全程没有有限 TTC”；落盘时保留为 null。
    result = sanitize_nonfinite(result)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "attempt_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return result


def rejected_episode(env, scenario_index, output_dir, error):
    """记录联合安全筛选拒绝，保留失败尝试并允许冻结清单继续运行。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "execution_trace.jsonl"
    result = {
        "scenario_index": int(scenario_index),
        "scenario_source": str(env.test_files[scenario_index]),
        "status": "generation_failure",
        "generation_failure": True,
        "generation_failure_type": type(error).__name__,
        "generation_failure_reason": str(error),
        "executed_steps": 0,
        "collision": None,
        "off_route": None,
        "completed": None,
        "progress": None,
        "min_ttc_s": None,
        "min_center_distance_m": None,
        "max_evasive_acceleration_mps2": None,
        "mean_evasive_acceleration_mps2": None,
        "ea_valid_frame_count": 0,
        "scenario_danger_score": None,
        "danger_valid": False,
        "danger_basis": "common_per_frame_ea_and_aligned_joint_reachability",
        "danger_valid_frame_count": 0,
        "danger_total_frame_count": 0,
        "danger_coverage": 0.0,
        "danger_windows": [],
        "autonomous_driving_ability_score": None,
        "ability_valid": False,
        "attack_plan_count": 0,
        "attack_executed_frames": 0,
        "obstacle_plan_count": 0,
        "planner_service_failure_count": 0,
        "background_collision_frames_crosscheck": 0,
        "feasibility": {
            "background_collision_frames": 0,
            "background_static_collision_frames": 0,
        },
        "execution_trace_path": str(trace_path) if trace_path.exists() else None,
        "video_path": None,
    }
    (output_dir / "attempt_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return result


def run_scenarios(env, policy, scenario_count, steps, output_dir, expect_joint, forecaster=None):
    """逐场运行；仅将联合安全候选耗尽视为可审计的生成失败。"""
    records = []
    for index in range(scenario_count):
        scenario_output = output_dir / "movies" / f"scenario_{index:03d}"
        try:
            extra = {} if forecaster is None else {"forecaster": forecaster}
            result = run_episode(
                env, policy, index, steps, scenario_output, expect_joint, **extra
            )
            result.setdefault("status", "completed")
            result.setdefault("generation_failure", False)
        except NoSafeJointCandidate as error:
            result = rejected_episode(env, index, scenario_output, error)
        records.append(result)
    return records


def aggregate(records):
    def finite_values(key):
        return [
            float(row[key]) for row in records
            if row.get(key) is not None and np.isfinite(float(row[key]))
        ]

    valid_d = [
        float(row["scenario_danger_score"])
        for row in records if row["danger_valid"]
    ]
    valid_scores = [
        float(row["autonomous_driving_ability_score"])
        for row in records if row["ability_valid"]
    ]
    min_ttc_values = finite_values("min_ttc_s")
    max_ea_values = finite_values("max_evasive_acceleration_mps2")
    mean_ea_values = finite_values("mean_evasive_acceleration_mps2")
    executed = [row for row in records if not row.get("generation_failure", False) and not row.get("runtime_failure", False)]
    attempted_count = len(records)
    executed_count = len(executed)
    return {
        "scenario_count": len(records),
        "attempted_scenario_count": attempted_count,
        "executed_scenario_count": executed_count,
        "generation_failure_count": sum(bool(row.get("generation_failure")) for row in records),
        "runtime_failure_count": sum(bool(row.get("runtime_failure")) for row in records),
        "actual_execution_scenario_count": sum(int(row.get("executed_steps", 0) > 0) for row in records),
        "execution_coverage": executed_count / attempted_count if attempted_count else 0.0,
        "outcome_rate_denominator": "executed_scenarios",
        "collision_rate": (
            float(np.mean([row["collision"] for row in executed]))
            if executed else None
        ),
        "off_route_rate": (
            float(np.mean([row["off_route"] for row in executed]))
            if executed else None
        ),
        "completion_rate": (
            float(np.mean([row["completed"] for row in executed]))
            if executed else None
        ),
        "mean_progress": (
            float(np.mean([row["progress"] for row in executed]))
            if executed else None
        ),
        "mean_min_ttc_s": (
            float(np.mean(min_ttc_values)) if min_ttc_values else None
        ),
        "mean_episode_max_ea_mps2": (
            float(np.mean(max_ea_values)) if max_ea_values else None
        ),
        "mean_ea_mps2": (
            float(np.mean(mean_ea_values)) if mean_ea_values else None
        ),
        "danger_valid_count": len(valid_d),
        "danger_valid_coverage": len(valid_d) / attempted_count if attempted_count else 0.0,
        "mean_valid_danger": float(np.mean(valid_d)) if valid_d else None,
        "ability_valid_count": len(valid_scores),
        "mean_valid_ability_score": (
            float(np.mean(valid_scores)) if valid_scores else None
        ),
        "background_collision_frames": int(sum(
            row["feasibility"]["background_collision_frames"] for row in records
        )),
        "background_static_collision_frames": int(sum(
            row["feasibility"]["background_static_collision_frames"] for row in records
        )),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=sorted(PASSIVE_METHODS))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-scenarios", type=int, default=0)
    parser.add_argument("--reachability-source", choices=("native", "ctrl_sim_rollout"), default="native")
    parser.add_argument("--forecast-tracking", choices=("observation", "persistent"), default="observation")
    args = parser.parse_args()
    if not 1 <= args.steps <= 400:
        parser.error("steps must be between 1 and 400")
    if args.output_dir.exists():
        raise FileExistsError(f"输出目录已存在：{args.output_dir}")
    manifest, scenario_files = validate_manifest(args.manifest)
    if args.max_scenarios > 0:
        scenario_files = scenario_files[:args.max_scenarios]
    spec = METHODS[args.method]
    overrides = [
        "sim.seed=42",
        "sim.policy=idm",
        f"sim.steps={args.steps}",
        "sim.visualize=true",
        "sim.lightweight=true",
        *spec.overrides,
    ]
    with initialize(version_base=None, config_path="../../cfgs"):
        cfg = compose(config_name="config", overrides=overrides)
    set_seed(int(cfg.sim.seed))
    env = Simulator(cfg)
    env.test_files = [str(path) for path in scenario_files]
    policy = IDMPolicy(cfg, env)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8"
    )
    forecaster = None
    if args.reachability_source == "ctrl_sim_rollout":
        from experiments.comparison.ctrl_sim_forecast import CtrlSimJointForecaster
        forecaster = CtrlSimJointForecaster(env, cfg.sim.evaluation.reachability.horizon_seconds, int(cfg.sim.seed), tracking=args.forecast_tracking)
    expect_joint = args.method == "safe_sim" or forecaster is not None
    records = run_scenarios(
        env, policy, len(scenario_files), args.steps, args.output_dir, expect_joint, forecaster
    )
    report = {
        "method": args.method,
        "reachability_source": args.reachability_source,
        "forecast_tracking": args.forecast_tracking,
        "metric_protocol": "ea_empty_set_zero_v2",
        "fidelity": spec.fidelity,
        "overrides": list(spec.overrides),
        "aggregate": aggregate(records),
        "episodes": records,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    with (args.output_dir / "results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = [{key: value for key, value in row.items()
                 if key not in {"danger_windows", "feasibility"}} for row in records]
        # 生成失败记录含额外审计字段，表头须覆盖全部记录。
        writer = csv.DictWriter(handle, fieldnames=csv_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
