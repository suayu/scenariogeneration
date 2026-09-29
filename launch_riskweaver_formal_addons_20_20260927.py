"""运行 20 场模块消融与跨自车策略补充实验。"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path("/home2/zhaoyx/scenario-dreamer")
BASE = ROOT / "experiments/riskweaver_formal_addons_20_20260927_r1"
GENERATION_RESULT = ROOT / "experiments/scenario_dreamer_ldm_full60_20260927_r2/stage_result.json"
SOURCE_MANIFEST = ROOT / "experiments/scenario_dreamer_ldm_full60_20260927_r2/riskweaver_frozen_120_scenes.json"
MAIN_ROOT = ROOT / "experiments/riskweaver_full_formal_120_qwen3_8b_20260927_r2"
MAIN_RESULT = MAIN_ROOT / "launcher_result.json"
SMOKE_ACCEPTANCE = ROOT / "work/riskweaver_profile_qwen3_8b_candidate_expansion_closed_loop_smoke_acceptance_20260925.json"
PROTOCOL_SOURCE = ROOT / "work/riskweaver_formal_addons_20_protocol_20260927.md"
PYTHON = Path("/home2/zhaoyx/miniconda3/envs/scenario-dreamer/bin/python")
MODEL = "qwen3-8b"
EXPECTED_SOURCE_SCENARIOS = 120
EXPECTED_SUBSET_SCENARIOS = 20

# 完整 IDM 画像组由 120 场主实验提供；这里只运行不重复的补充组。
GROUPS = (
    {
        "name": "idm_profile_off",
        "policy": "idm",
        "profile_enabled": False,
        "candidate_expansion_enabled": False,
        "anchor_drive_enabled": True,
        "attack_mode": "joint",
        "obstacles_enabled": True,
        "purpose": "ablate_history_profile_and_supply_idm_cross_policy_control",
    },
    {
        "name": "idm_no_anchor_drive",
        "policy": "idm",
        "profile_enabled": True,
        "candidate_expansion_enabled": True,
        "anchor_drive_enabled": False,
        "attack_mode": "joint",
        "obstacles_enabled": True,
        "purpose": "ablate_anchor_drive_style_anchor_enhancement",
    },
    {
        "name": "idm_dynamic_only",
        "policy": "idm",
        "profile_enabled": True,
        "candidate_expansion_enabled": True,
        "anchor_drive_enabled": True,
        "attack_mode": "trajectory_only",
        "obstacles_enabled": False,
        "purpose": "ablate_static_obstacle_generation",
    },
    {
        "name": "rl_profile_off",
        "policy": "rl",
        "profile_enabled": False,
        "candidate_expansion_enabled": False,
        "anchor_drive_enabled": True,
        "attack_mode": "joint",
        "obstacles_enabled": True,
        "purpose": "cross_policy_profile_control",
    },
    {
        "name": "rl_profile_on",
        "policy": "rl",
        "profile_enabled": True,
        "candidate_expansion_enabled": True,
        "anchor_drive_enabled": True,
        "attack_mode": "joint",
        "obstacles_enabled": True,
        "purpose": "cross_policy_full_method",
    },
)


def write_json(path, payload):
    """保存严格 JSON，禁止 NaN 污染实验状态。"""
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def sha256(path):
    """流式计算文件哈希。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_subset_indices(total=EXPECTED_SOURCE_SCENARIOS, count=EXPECTED_SUBSET_SCENARIOS):
    """按等间隔索引确定子集，禁止读取碰撞或危险度结果。"""
    if total <= 0 or count <= 0 or count > total or total % count:
        raise ValueError("正式子集要求总数可被子集数整除")
    stride = total // count
    return list(range(0, total, stride))


def build_subset_manifest(source):
    """从冻结清单构造包含原有与新生成场景的确定性 20 场子集。"""
    files = list(source["scenario_files"])
    digests = list(source["scenario_sha256"])
    if len(files) != EXPECTED_SOURCE_SCENARIOS or len(digests) != len(files):
        raise RuntimeError("源清单大小或哈希列不符合 120 场协议")
    indices = fixed_subset_indices()
    payload = dict(source)
    payload.update(
        {
            "name": "riskweaver_formal_evenly_spaced_20_of_120_20260927",
            "scenario_count": len(indices),
            "scenario_files": [files[index] for index in indices],
            "scenario_sha256": [digests[index] for index in indices],
            "original_scenario_indices": indices,
            "selection_rule": "indices_0_to_114_stride_6_no_outcome_selection",
            "statistical_unit": "unique_initial_scenario",
        }
    )
    return payload


def environment():
    """固定模型池与项目运行环境。"""
    return dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "1"),
        PROJECT_ROOT=str(ROOT),
        SCRATCH_ROOT=str(ROOT),
        DATASET_ROOT=str(ROOT / "metadata"),
        PYTHONPATH=f"{ROOT}:{ROOT / 'safe-sim'}:{ROOT / 'safe-sim/trajdata/src'}",
        MPLBACKEND="Agg",
        PYTHONUNBUFFERED="1",
        HYDRA_FULL_ERROR="1",
        LLM_MODEL_NAMES=MODEL,
    )


def build_command(group, manifest, instruction, run):
    """只改变当前消融项，其余生成预算和安全门禁保持一致。"""
    profile = bool(group["profile_enabled"])
    overrides = [
        f"sim.policy={group['policy']}",
        "sim.seed=42",
        "+sim.continue_after_off_route=true",
        f"sim.evaluation.max_scenarios={EXPECTED_SUBSET_SCENARIOS}",
        "sim.steps=400",
        f"+sim.scenario_manifest={manifest}",
        f"+sim.attack_request_file={instruction}",
        "sim.llm.provider=dashscope",
        f"sim.llm.model_name={MODEL}",
        f"sim.llm.model_names=[{MODEL}]",
        "sim.llm.max_planning_calls=4",
        "sim.llm.profile_candidate_dry_run=false",
        "sim.llm.profile_allow_partial_ranking=true",
        f"sim.llm.attack_mode={group['attack_mode']}",
        "sim.llm.multiagent.mode=single",
        f"sim.llm.obstacles.enabled={str(group['obstacles_enabled']).lower()}",
        f"sim.traffic_model.iterative_adversarial.profile_enabled={str(profile).lower()}",
        "sim.traffic_model.iterative_adversarial.escalation_enabled=false",
        f"sim.traffic_model.iterative_adversarial.full_method_enabled={str(profile).lower()}",
        f"sim.ego_profile.candidate_expansion_enabled={str(group['candidate_expansion_enabled']).lower()}",
        "sim.ego_profile.rear_approach_accelerations_mps2=[1.5]",
        "sim.traffic_model.guidance.mode=llm_joint",
        "sim.traffic_model.dynamics_projection.enabled=true",
        f"sim.traffic_model.anchor_guidance.anchor_drive_enabled={str(group['anchor_drive_enabled']).lower()}",
        "+sim.llm.min_planning_step=0",
        "+sim.llm.planning_interval_frames=10",
        "sim.evaluation.difficulty_control.mode=off",
        "sim.evaluation.difficulty_control.max_replays=0",
        "sim.evaluation.difficulty_control.smoke_acceptance=false",
        "sim.visualize=true",
        "sim.lightweight=true",
        f"sim.movie_path={run / 'movies'}",
        f"sim.scenario_data_output_path={run / 'carla'}",
        f"hydra.run.dir={run / 'hydra'}",
    ]
    return [
        "bash",
        "scripts/run_with_llm_provider.sh",
        "dashscope",
        str(PYTHON),
        "run_simulation.py",
        *overrides,
    ]


def summarize_attempts(run):
    """保留全部尝试，并汇总攻击、能力和安全审计。"""
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((run / "movies").rglob("attempt_result.json"))
    ]
    ability = [
        float(row["autonomous_driving_ability_score"])
        for row in rows
        if isinstance(row.get("autonomous_driving_ability_score"), (int, float))
        and math.isfinite(row["autonomous_driving_ability_score"])
    ]
    return {
        "attempt_count": len(rows),
        "planner_service_failure_count": sum(int(row.get("planner_service_failure_count") or 0) for row in rows),
        "attack_plan_count": sum(int(row.get("attack_plan_count") or 0) for row in rows),
        "attack_scene_count": sum(int(row.get("attack_executed_frames") or 0) > 0 for row in rows),
        "attack_executed_frames": sum(int(row.get("attack_executed_frames") or 0) for row in rows),
        "generation_failure_count": sum(bool(row.get("generation_failure")) for row in rows),
        "collision_count": sum(bool(row.get("collision")) for row in rows if not row.get("generation_failure")),
        "ability_valid_count": len(ability),
        "mean_ability_score": sum(ability) / len(ability) if ability else None,
        "background_collision_events": sum(int((row.get("feasibility") or {}).get("background_collision_events") or 0) for row in rows),
        "background_static_collision_events": sum(int((row.get("feasibility") or {}).get("background_static_collision_events") or 0) for row in rows),
    }


def require_prerequisites():
    """正式补充实验必须等待场景生成和 120 场主方法完成。"""
    for path in (GENERATION_RESULT, SOURCE_MANIFEST, MAIN_RESULT, SMOKE_ACCEPTANCE, PROTOCOL_SOURCE):
        if not path.is_file():
            raise FileNotFoundError(f"缺少正式实验前置文件：{path}")
    generation = json.loads(GENERATION_RESULT.read_text(encoding="utf-8"))
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    main = json.loads(MAIN_RESULT.read_text(encoding="utf-8"))
    smoke = json.loads(SMOKE_ACCEPTANCE.read_text(encoding="utf-8"))
    if generation.get("accepted") is not True or int(generation.get("frozen_scenario_count") or 0) != 120:
        raise RuntimeError("LDM 生成与 120 场冻结门禁未通过")
    if generation.get("frozen_manifest_sha256") != sha256(SOURCE_MANIFEST):
        raise RuntimeError("冻结清单哈希与生成终态不一致")
    if int(manifest.get("scenario_count") or 0) != 120:
        raise RuntimeError("冻结清单不是 120 场")
    if main.get("valid_for_analysis") is not True or int(main.get("attempt_count") or 0) != 120:
        raise RuntimeError("RiskWeaver 120 场完整方法尚未有效完成")
    if int(main.get("planner_service_failure_count") or 0) != 0:
        raise RuntimeError("RiskWeaver 主实验存在规划服务失败")
    required_smoke = {
        "accepted": True,
        "model_name": MODEL,
        "planner_service_failure_count": 0,
        "background_collision_count": 0,
        "background_static_collision_count": 0,
    }
    for key, expected in required_smoke.items():
        if smoke.get(key) != expected:
            raise RuntimeError(f"闭环冒烟门禁 {key} 不满足")
    return generation, manifest, main, smoke


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        for group in GROUPS:
            command = build_command(group, SOURCE_MANIFEST, PROTOCOL_SOURCE, BASE / group["name"])
            completed = subprocess.run(
                [*command, "--cfg", "job", "--resolve"],
                cwd=ROOT,
                env=environment(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if completed.returncode:
                return completed.returncode
        return 0

    generation, source_manifest, main_result, smoke = require_prerequisites()
    if BASE.exists():
        raise FileExistsError(f"正式补充实验目录已存在：{BASE}")
    BASE.mkdir(parents=True)
    shutil.copy2(__file__, BASE / Path(__file__).name)
    shutil.copy2(PROTOCOL_SOURCE, BASE / "protocol.md")
    subset_manifest = BASE / "scenario_manifest_20.json"
    write_json(subset_manifest, build_subset_manifest(source_manifest))
    instruction = BASE / "instruction.txt"
    instruction.write_text(
        "Generate a physically executable adversarial interaction that targets vulnerabilities inferred "
        "from the ego driving history and current scene. Rank validated structured candidates, then select "
        "a discrete dynamic strategy, target and sparse anchors, a hazardous static road condition, or a "
        "compatible combination. Preserve non-target background safety and a positive theoretical drivable "
        "region for the ego.\n",
        encoding="utf-8",
    )
    state = {
        "purpose": "paired 20-scene ablation and cross-policy experiments",
        "status": "running",
        "model": MODEL,
        "subset_indices": fixed_subset_indices(),
        "subset_manifest_sha256": sha256(subset_manifest),
        "no_outcome_based_selection": True,
        "generation_acceptance": generation,
        "main_acceptance": main_result,
        "smoke_acceptance": smoke,
        "groups": [],
        "started_at": time.time(),
    }
    write_json(BASE / "launcher_result.partial.json", state)
    for group in GROUPS:
        run = BASE / group["name"]
        run.mkdir()
        command = build_command(group, subset_manifest, instruction, run)
        write_json(run / "launch.json", {**group, "command": command, "started_at": time.time()})
        with (run / "resolved_config.yaml").open("w", encoding="utf-8") as output:
            subprocess.run(
                [*command, "--cfg", "job", "--resolve"],
                cwd=ROOT,
                env=environment(),
                stdout=output,
                stderr=subprocess.STDOUT,
                check=True,
            )
        started = time.time()
        with (run / "run.log").open("w", encoding="utf-8") as output:
            exit_code = subprocess.call(
                command,
                cwd=ROOT,
                env=environment(),
                stdout=output,
                stderr=subprocess.STDOUT,
            )
        summary = summarize_attempts(run)
        valid = bool(
            exit_code == 0
            and summary["attempt_count"] == EXPECTED_SUBSET_SCENARIOS
            and summary["planner_service_failure_count"] == 0
            and summary["background_collision_events"] == 0
            and summary["background_static_collision_events"] == 0
        )
        result = {
            **group,
            "exit_code": exit_code,
            "valid_for_analysis": valid,
            **summary,
            "started_at": started,
            "finished_at": time.time(),
        }
        write_json(run / "exit_status.json", result)
        state["groups"].append(result)
        write_json(BASE / "launcher_result.partial.json", state)
        if not valid:
            state["status"] = "failed"
            break
    else:
        state["status"] = "complete"
    state["finished_at"] = time.time()
    write_json(BASE / "launcher_result.json", state)
    return 0 if state["status"] == "complete" else 70


if __name__ == "__main__":
    sys.exit(main())
