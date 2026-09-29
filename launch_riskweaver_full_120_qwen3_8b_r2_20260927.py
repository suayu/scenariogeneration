"""在统一冻结的 120 个场景上运行 RiskWeaver 完整方法。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path("/home2/zhaoyx/scenario-dreamer")
BASE = ROOT / "experiments/riskweaver_full_formal_120_qwen3_8b_20260927_r2"
GENERATION_RESULT = ROOT / "experiments/scenario_dreamer_ldm_full60_20260927_r2/stage_result.json"
SOURCE_MANIFEST = ROOT / "experiments/scenario_dreamer_ldm_full60_20260927_r2/riskweaver_frozen_120_scenes.json"
SOURCE_PROTOCOL = ROOT / "work/riskweaver_full_formal_120_protocol_20260927.md"
SMOKE_ACCEPTANCE = ROOT / "work/riskweaver_profile_qwen3_8b_candidate_expansion_closed_loop_smoke_acceptance_20260925.json"
DEVELOPMENT_RESULT = ROOT / "experiments/riskweaver_profile_candidate_expansion_paired_dev6_20260927_r1/launcher_result.json"
MODEL = "qwen3-8b"
PYTHON = Path("/home2/zhaoyx/miniconda3/envs/scenario-dreamer/bin/python")
EXPECTED_SCENARIOS = 120
EXPECTED_DEVELOPMENT_GROUPS = {"profile_off", "profile_on"}


def write_json(path, payload):
    """保存严格 JSON，避免非有限值破坏审计。"""
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def sha256(path):
    """流式计算大文件或清单哈希。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_prerequisites():
    """验证场景、模型闭环和画像开发实验均已通过固定门禁。"""
    for path in (GENERATION_RESULT, SOURCE_MANIFEST, SOURCE_PROTOCOL, SMOKE_ACCEPTANCE, DEVELOPMENT_RESULT):
        if not path.is_file():
            raise FileNotFoundError(f"缺少正式实验前置文件：{path}")

    generation = json.loads(GENERATION_RESULT.read_text(encoding="utf-8"))
    if generation.get("accepted") is not True:
        raise RuntimeError("LDM 场景生成未通过验收")
    if int(generation.get("frozen_scenario_count") or 0) != EXPECTED_SCENARIOS:
        raise RuntimeError("LDM 生成结果未冻结为 120 场")
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if int(manifest.get("scenario_count") or 0) != EXPECTED_SCENARIOS:
        raise RuntimeError("冻结清单场景数不是 120")
    if generation.get("frozen_manifest_sha256") != sha256(SOURCE_MANIFEST):
        raise RuntimeError("冻结清单哈希与生成验收记录不一致")

    smoke = json.loads(SMOKE_ACCEPTANCE.read_text(encoding="utf-8"))
    smoke_expected = {
        "accepted": True,
        "model_name": MODEL,
        "planner_service_failure_count": 0,
        "background_collision_count": 0,
        "background_static_collision_count": 0,
    }
    for key, expected in smoke_expected.items():
        if smoke.get(key) != expected:
            raise RuntimeError(f"闭环冒烟门禁 {key}={smoke.get(key)!r}，期望 {expected!r}")
    if int(smoke.get("attack_plan_count") or 0) <= 0:
        raise RuntimeError("闭环冒烟没有有效攻击计划")
    if int(smoke.get("attack_executed_frames") or 0) <= 0:
        raise RuntimeError("闭环冒烟没有实际攻击执行帧")

    development = json.loads(DEVELOPMENT_RESULT.read_text(encoding="utf-8"))
    rows = development.get("groups") or []
    if {row.get("group") for row in rows} != EXPECTED_DEVELOPMENT_GROUPS:
        raise RuntimeError("六场画像开发实验组不完整")
    invalid = [
        row for row in rows
        if row.get("exit_code") != 0
        or row.get("valid_for_development_analysis") is not True
        or int(row.get("attempt_count") or 0) != 6
        or int(row.get("planner_service_failure_count") or 0) != 0
        or int(row.get("background_collision_events") or 0) != 0
        or int(row.get("background_static_collision_events") or 0) != 0
    ]
    if invalid:
        raise RuntimeError(f"六场画像开发实验未通过有效性门禁：{invalid}")
    profile_on = next(row for row in rows if row.get("group") == "profile_on")
    if int(profile_on.get("attack_plan_count") or 0) <= 0:
        raise RuntimeError("画像开启组没有有效攻击计划")
    if int(profile_on.get("attack_executed_frames") or 0) <= 0:
        raise RuntimeError("画像开启组没有实际攻击执行帧")
    return generation, smoke, development


def build_command(manifest, instruction, run):
    """固定完整方法参数，并显式启用画像候选扩展与动静态攻击。"""
    overrides = [
        "sim.policy=idm",
        "sim.seed=42",
        "+sim.continue_after_off_route=true",
        f"sim.evaluation.max_scenarios={EXPECTED_SCENARIOS}",
        "sim.steps=400",
        f"+sim.scenario_manifest={manifest}",
        f"+sim.attack_request_file={instruction}",
        "sim.llm.provider=dashscope",
        f"sim.llm.model_name={MODEL}",
        f"sim.llm.model_names=[{MODEL}]",
        "sim.llm.max_planning_calls=4",
        "sim.llm.profile_candidate_dry_run=false",
        "sim.llm.profile_allow_partial_ranking=true",
        "sim.llm.attack_mode=joint",
        "sim.llm.multiagent.mode=single",
        "sim.llm.obstacles.enabled=true",
        "sim.traffic_model.iterative_adversarial.profile_enabled=true",
        "sim.traffic_model.iterative_adversarial.escalation_enabled=false",
        "sim.traffic_model.iterative_adversarial.full_method_enabled=true",
        "sim.ego_profile.candidate_expansion_enabled=true",
        "sim.ego_profile.rear_approach_accelerations_mps2=[1.5]",
        "sim.traffic_model.guidance.mode=llm_joint",
        "sim.traffic_model.dynamics_projection.enabled=true",
        "sim.traffic_model.anchor_guidance.anchor_drive_enabled=true",
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


def environment():
    """固定单一模型，不允许额度失败后静默切换模型。"""
    return dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        PROJECT_ROOT=str(ROOT),
        SCRATCH_ROOT=str(ROOT),
        DATASET_ROOT=str(ROOT / "metadata"),
        PYTHONPATH=f"{ROOT}:{ROOT / 'safe-sim'}:{ROOT / 'safe-sim/trajdata/src'}",
        MPLBACKEND="Agg",
        PYTHONUNBUFFERED="1",
        HYDRA_FULL_ERROR="1",
        LLM_MODEL_NAMES=MODEL,
    )


def summarize_attempts(run):
    """从逐场审计文件验证正式实验分母、服务和背景安全。"""
    rows = []
    for path in sorted((run / "movies").rglob("attempt_result.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return {
        "attempt_count": len(rows),
        "planner_service_failure_count": sum(int(row.get("planner_service_failure_count") or 0) for row in rows),
        "attack_plan_count": sum(int(row.get("attack_plan_count") or 0) for row in rows),
        "attack_scene_count": sum(int(row.get("attack_executed_frames") or 0) > 0 for row in rows),
        "attack_executed_frames": sum(int(row.get("attack_executed_frames") or 0) for row in rows),
        "generation_failure_count": sum(bool(row.get("generation_failure")) for row in rows),
        "background_collision_events": sum(int((row.get("feasibility") or {}).get("background_collision_events") or (row.get("feasibility") or {}).get("background_collision_frames") or 0) for row in rows),
        "background_static_collision_events": sum(int((row.get("feasibility") or {}).get("background_static_collision_events") or (row.get("feasibility") or {}).get("background_static_collision_frames") or 0) for row in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只解析冻结配置，不创建正式实验目录或调用模型",
    )
    args = parser.parse_args()
    validation_command = build_command(SOURCE_MANIFEST, SOURCE_PROTOCOL, BASE)
    if args.validate_only:
        completed = subprocess.run(
            validation_command + ["--cfg", "job", "--resolve"],
            cwd=ROOT,
            env=environment(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return completed.returncode

    generation, smoke, development = require_prerequisites()
    if BASE.exists():
        raise FileExistsError(f"正式实验目录已存在：{BASE}")
    BASE.mkdir(parents=True)
    shutil.copy2(__file__, BASE / Path(__file__).name)
    manifest = BASE / "scenario_manifest.json"
    protocol = BASE / "protocol.md"
    shutil.copy2(SOURCE_MANIFEST, manifest)
    shutil.copy2(SOURCE_PROTOCOL, protocol)
    instruction = BASE / "instruction.txt"
    instruction.write_text(
        "Generate a physically executable adversarial interaction that targets vulnerabilities inferred "
        "from the ego driving history and current scene. Rank validated structured candidates, then select "
        "a discrete dynamic strategy, target and sparse anchors, a hazardous static road condition, or a "
        "compatible combination. Preserve non-target background safety and a positive theoretical drivable "
        "region for the ego.\n",
        encoding="utf-8",
    )
    command = build_command(manifest, instruction, BASE)
    source_files = (
        "run_simulation.py",
        "simulator.py",
        "scenario_generator.py",
        "policies/ego_profile.py",
        "policies/profile_planner.py",
        "policies/llm_adversarial_planner.py",
        "policies/diffusion_model_wrapper.py",
        "policies/llm_anchor_guidance.py",
        "policies/joint_safety.py",
        "policies/trajectory_projection.py",
        "cfgs/sim/base.yaml",
    )
    launch = {
        "purpose": "120-scene formal full RiskWeaver comparison",
        "model": MODEL,
        "attack_mode": "joint",
        "profile_enabled": True,
        "candidate_expansion_enabled": True,
        "difficulty_control_mode": "off",
        "expected_scenarios": EXPECTED_SCENARIOS,
        "no_outcome_based_selection": True,
        "command": command,
        "started_at": time.time(),
        "source_hashes": {name: sha256(ROOT / name) for name in source_files},
        "selection_manifest_sha256": sha256(manifest),
        "protocol_sha256": sha256(protocol),
        "generation_acceptance": generation,
        "smoke_acceptance": smoke,
        "development_acceptance": development,
    }
    write_json(BASE / "launch.json", launch)
    with (BASE / "resolved_config.yaml").open("w", encoding="utf-8") as output:
        subprocess.run(
            command + ["--cfg", "job", "--resolve"],
            cwd=ROOT,
            env=environment(),
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )
    with (BASE / "run.log").open("w", encoding="utf-8") as output:
        exit_code = subprocess.call(
            command,
            cwd=ROOT,
            env=environment(),
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    summary = summarize_attempts(BASE)
    valid = bool(
        exit_code == 0
        and summary["attempt_count"] == EXPECTED_SCENARIOS
        and summary["planner_service_failure_count"] == 0
        and summary["background_collision_events"] == 0
        and summary["background_static_collision_events"] == 0
    )
    result = {
        "exit_code": exit_code,
        "valid_for_analysis": valid,
        **summary,
        "finished_at": time.time(),
    }
    write_json(BASE / "launcher_result.json", result)
    return 0 if valid else (exit_code or 70)


if __name__ == "__main__":
    sys.exit(main())
