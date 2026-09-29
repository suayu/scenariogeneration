"""在全部 75 个冻结场景上运行通过画像矩阵门禁的 RiskWeaver 完整方法。"""

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
BASE = ROOT / "experiments/riskweaver_full_formal_all75_qwen37_20260925_r2"
SOURCE_MANIFEST = ROOT / "work/riskweaver_all_75_unique_manifest_20260924.json"
SOURCE_PROTOCOL = ROOT / "work/riskweaver_full_formal_all75_protocol_qwen37_20260925.md"
SMOKE_ACCEPTANCE = ROOT / "work/riskweaver_profile_qwen37_closed_loop_smoke_acceptance_20260925.json"
PROFILE_MATRIX_RESULT = ROOT / "experiments/riskweaver_profile_formal_idm_rl_20260925_r6/launcher_result.json"
MODEL = "qwen3.7-plus"
PYTHON = Path("/home2/zhaoyx/miniconda3/envs/scenario-dreamer/bin/python")
EXPECTED_PROFILE_GROUPS = {
    "idm_profile_off",
    "idm_profile_on",
    "rl_profile_off",
    "rl_profile_on",
}


def write_json(path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def require_prerequisites():
    """正式主比较只接受通过真实闭环和完整画像矩阵门禁的同一模型。"""
    if not SMOKE_ACCEPTANCE.exists():
        raise RuntimeError(f"缺少闭环冒烟验收文件：{SMOKE_ACCEPTANCE}")
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
            raise RuntimeError(
                f"闭环冒烟门禁 {key}={smoke.get(key)!r}，期望 {expected!r}"
            )
    if int(smoke.get("attack_plan_count", 0)) <= 0:
        raise RuntimeError("闭环冒烟没有有效攻击计划")
    if int(smoke.get("attack_executed_frames", 0)) <= 0:
        raise RuntimeError("闭环冒烟没有实际攻击执行帧")

    if not PROFILE_MATRIX_RESULT.exists():
        raise RuntimeError(f"画像 R6 尚未完成：{PROFILE_MATRIX_RESULT}")
    matrix = json.loads(PROFILE_MATRIX_RESULT.read_text(encoding="utf-8"))
    rows = matrix.get("groups", [])
    names = {row.get("group") for row in rows}
    if names != EXPECTED_PROFILE_GROUPS:
        raise RuntimeError(
            f"画像 R6 组不完整：实际 {sorted(str(name) for name in names)}"
        )
    invalid = [
        row
        for row in rows
        if row.get("exit_code") != 0
        or row.get("planner_service_failure_count") != 0
        or row.get("valid_for_analysis") is not True
    ]
    if invalid:
        raise RuntimeError(f"画像 R6 未通过有效性门禁：{invalid}")
    return smoke, matrix


def build_command(manifest, instruction, run):
    """冻结正式实验参数，完整方法同时允许动态与静态风险。"""
    overrides = [
        "sim.policy=idm",
        "sim.seed=42",
        "+sim.continue_after_off_route=true",
        "sim.evaluation.max_scenarios=75",
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
        "sim.traffic_model.guidance.mode=llm_joint",
        "sim.traffic_model.dynamics_projection.enabled=true",
        "sim.traffic_model.anchor_guidance.anchor_drive_enabled=true",
        "+sim.llm.min_planning_step=0",
        "+sim.llm.planning_interval_frames=10",
        "sim.evaluation.difficulty_control.mode=off",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只解析冻结配置，不创建正式实验目录或调用模型",
    )
    args = parser.parse_args()
    command = build_command(SOURCE_MANIFEST, SOURCE_PROTOCOL, BASE)
    if args.validate_only:
        completed = subprocess.run(
            command + ["--cfg", "job", "--resolve"],
            cwd=ROOT,
            env=environment(),
            stdout=subprocess.DEVNULL,
            check=False,
        )
        return completed.returncode

    smoke_acceptance, profile_matrix_result = require_prerequisites()
    if BASE.exists():
        raise FileExistsError(f"正式实验目录已存在：{BASE}")
    BASE.mkdir(parents=True)
    shutil.copy2(__file__, BASE / Path(__file__).name)
    shutil.copy2(SOURCE_MANIFEST, BASE / "scenario_manifest.json")
    shutil.copy2(SOURCE_PROTOCOL, BASE / "protocol.md")
    shutil.copy2(SMOKE_ACCEPTANCE, BASE / "smoke_acceptance.json")
    shutil.copy2(PROFILE_MATRIX_RESULT, BASE / "profile_matrix_acceptance.json")
    instruction = BASE / "instruction.txt"
    instruction.write_text(
        "Generate a physically executable adversarial interaction that probes the ego driving strategy. "
        "Select dynamic traffic, a hazardous static road condition, or a compatible combination from "
        "validated candidates. The attacker may leave the road topology. Preserve non-target background "
        "safety and a positive theoretical drivable region for the ego.\n",
        encoding="utf-8",
    )
    manifest = BASE / "scenario_manifest.json"
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
    )
    launch = {
        "purpose": "75-scene formal full RiskWeaver comparison after Qwen3.7 closed-loop and profile-matrix gates",
        "model": MODEL,
        "attack_mode": "joint",
        "profile_enabled": True,
        "command": command,
        "started_at": time.time(),
        "source_hashes": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in source_files
        },
        "selection_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256((BASE / "protocol.md").read_bytes()).hexdigest(),
        "smoke_acceptance_sha256": hashlib.sha256((BASE / "smoke_acceptance.json").read_bytes()).hexdigest(),
        "profile_matrix_acceptance_sha256": hashlib.sha256((BASE / "profile_matrix_acceptance.json").read_bytes()).hexdigest(),
        "smoke_acceptance": smoke_acceptance,
        "profile_matrix_result": profile_matrix_result,
    }
    write_json(BASE / "launch.json", launch)
    with (BASE / "resolved_config.yaml").open("w", encoding="utf-8") as output:
        subprocess.run(
            command + ["--cfg", "job", "--resolve"],
            cwd=ROOT,
            env=environment(),
            stdout=output,
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
    result_path = BASE / "movies/multi_scenario_ability_results.json"
    service_failures = None
    if exit_code == 0 and result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        service_failures = sum(
            int(row.get("planner_service_failure_count", 0) or 0)
            for row in payload.get("episodes", [])
        )
    valid = exit_code == 0 and service_failures == 0
    result = {
        "exit_code": exit_code,
        "planner_service_failure_count": service_failures,
        "valid_for_analysis": valid,
        "finished_at": time.time(),
    }
    write_json(BASE / "launcher_result.json", result)
    if valid:
        return 0
    return exit_code or 70


if __name__ == "__main__":
    sys.exit(main())
