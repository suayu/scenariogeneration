"""六场开发集画像配对：验证候选扩展修正后的完整画像链路。"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


ROOT = Path("/home2/zhaoyx/scenario-dreamer")
OUT = ROOT / "experiments/riskweaver_profile_candidate_expansion_paired_dev6_20260927_r1"
SOURCE_MANIFEST = ROOT / "experiments/riskweaver_profile_scene_screen_20260922/initial_candidate_manifest.json"
SMOKE_ACCEPTANCE = ROOT / "work/riskweaver_profile_qwen3_8b_candidate_expansion_closed_loop_smoke_acceptance_20260925.json"
MODEL_NAME = "qwen3-8b"
EXPECTED_SCENARIOS = 6


def write_json(path, value):
    """以严格 JSON 保存实验状态，禁止 NaN 混入审计产物。"""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def summarize_group(run_dir):
    """按固定六场分母汇总门禁；服务失败不能被当作正常无攻击。"""
    rows = []
    for path in sorted(run_dir.rglob("attempt_result.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return {
        "attempt_count": len(rows),
        "planner_service_failure_count": sum(
            int(row.get("planner_service_failure_count") or 0) for row in rows
        ),
        "attack_plan_count": sum(int(row.get("attack_plan_count") or 0) for row in rows),
        "attack_scene_count": sum(
            int(row.get("attack_executed_frames") or 0) > 0 for row in rows
        ),
        "attack_executed_frames": sum(
            int(row.get("attack_executed_frames") or 0) for row in rows
        ),
        "valid_danger_count": sum(bool(row.get("danger_valid")) for row in rows),
        "generation_failure_count": sum(bool(row.get("generation_failure")) for row in rows),
        "background_collision_events": sum(
            int((row.get("feasibility") or {}).get("background_collision_events") or 0)
            for row in rows
        ),
        "background_static_collision_events": sum(
            int((row.get("feasibility") or {}).get("background_static_collision_events") or 0)
            for row in rows
        ),
    }


acceptance = json.loads(SMOKE_ACCEPTANCE.read_text(encoding="utf-8"))
if not acceptance.get("accepted") or acceptance.get("model_name") != MODEL_NAME:
    raise RuntimeError("qwen3-8b 候选扩展闭环验收无效，拒绝启动配对实验")
if int(acceptance.get("planner_service_failure_count") or 0) != 0:
    raise RuntimeError("闭环验收包含规划服务失败，拒绝启动配对实验")

manifest_payload = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
if len(manifest_payload.get("scenario_files") or []) != EXPECTED_SCENARIOS:
    raise RuntimeError("开发集清单必须恰好包含六个预冻结场景")

OUT.mkdir(parents=True, exist_ok=False)
shutil.copy2(__file__, OUT / Path(__file__).name)
manifest = OUT / "scenario_manifest.json"
shutil.copy2(SOURCE_MANIFEST, manifest)
instruction = OUT / "instruction.txt"
instruction.write_text(
    "Generate a physically executable adversarial interaction that probes the ego driving strategy. "
    "The attacker may leave the road topology. Preserve non-target background safety and a positive "
    "theoretical drivable region for the ego.\n",
    encoding="utf-8",
)

source_names = [
    "scenario_generator.py",
    "policies/llm_adversarial_planner.py",
    "policies/ego_profile.py",
    "policies/profile_planner.py",
    "policies/diffusion_model_wrapper.py",
    "policies/joint_safety.py",
    "run_simulation.py",
    "cfgs/sim/base.yaml",
]
source_hashes = {
    name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_names
}
env = dict(
    os.environ,
    CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "1"),
    PROJECT_ROOT=str(ROOT),
    SCRATCH_ROOT=str(ROOT),
    DATASET_ROOT=str(ROOT / "metadata"),
    PYTHONPATH=f"{ROOT}:{ROOT / 'safe-sim'}:{ROOT / 'safe-sim/trajdata/src'}",
    MPLBACKEND="Agg",
    PYTHONUNBUFFERED="1",
    HYDRA_FULL_ERROR="1",
    LLM_MODEL_NAMES=MODEL_NAME,
)

groups = [("profile_off", False), ("profile_on", True)]
launcher_results = []
for group_name, profile_enabled in groups:
    run_dir = OUT / group_name
    run_dir.mkdir()
    # 候选扩展属于完整画像攻击链路；关闭画像时保留原始无画像流程。
    expansion_enabled = profile_enabled
    rear_accelerations = "[1.5]" if profile_enabled else "[]"
    overrides = [
        "sim.policy=idm",
        "sim.seed=42",
        "+sim.continue_after_off_route=true",
        f"sim.evaluation.max_scenarios={EXPECTED_SCENARIOS}",
        "sim.steps=400",
        f"+sim.scenario_manifest={manifest}",
        f"+sim.attack_request_file={instruction}",
        "sim.llm.provider=dashscope",
        f"sim.llm.model_name={MODEL_NAME}",
        f"sim.llm.model_names=[{MODEL_NAME}]",
        "sim.llm.max_planning_calls=4",
        "sim.llm.profile_candidate_dry_run=false",
        "sim.llm.attack_mode=trajectory_only",
        "sim.llm.multiagent.mode=single",
        "sim.llm.obstacles.enabled=false",
        f"sim.traffic_model.iterative_adversarial.profile_enabled={str(profile_enabled).lower()}",
        f"sim.traffic_model.iterative_adversarial.full_method_enabled={str(profile_enabled).lower()}",
        f"sim.ego_profile.candidate_expansion_enabled={str(expansion_enabled).lower()}",
        f"sim.ego_profile.rear_approach_accelerations_mps2={rear_accelerations}",
        "sim.traffic_model.iterative_adversarial.escalation_enabled=false",
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
        f"sim.movie_path={run_dir / 'movies'}",
        f"sim.scenario_data_output_path={run_dir / 'carla'}",
        f"hydra.run.dir={run_dir / 'hydra'}",
    ]
    command = [
        "bash",
        "scripts/run_with_llm_provider.sh",
        "dashscope",
        "/home2/zhaoyx/miniconda3/envs/scenario-dreamer/bin/python",
        "run_simulation.py",
        *overrides,
    ]
    write_json(
        run_dir / "launch.json",
        {
            "group": group_name,
            "profile_enabled": profile_enabled,
            "candidate_expansion_enabled": expansion_enabled,
            "rear_approach_accelerations_mps2": [1.5] if profile_enabled else [],
            "model_name": MODEL_NAME,
            "command": command,
            "started_at": time.time(),
            "source_hashes": source_hashes,
            "selection_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "development_only": True,
        },
    )
    with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as output:
        subprocess.run(
            command + ["--cfg", "job", "--resolve"],
            cwd=ROOT,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )
    started_at = time.time()
    with (run_dir / "run.log").open("w", encoding="utf-8") as output:
        exit_code = subprocess.call(
            command,
            cwd=ROOT,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    summary = summarize_group(run_dir)
    valid = (
        exit_code == 0
        and summary["attempt_count"] == EXPECTED_SCENARIOS
        and summary["planner_service_failure_count"] == 0
    )
    status = {
        "exit_code": exit_code,
        "started_at": started_at,
        "finished_at": time.time(),
        "valid_for_development_analysis": valid,
        **summary,
    }
    write_json(run_dir / "exit_status.json", status)
    launcher_results.append({"group": group_name, **status})
    write_json(
        OUT / "launcher_result.partial.json",
        {"groups": launcher_results, "updated_at": time.time()},
    )
    if not valid:
        break

write_json(
    OUT / "launcher_result.json",
    {"groups": launcher_results, "finished_at": time.time()},
)
raise SystemExit(0 if len(launcher_results) == len(groups) and all(x["valid_for_development_analysis"] for x in launcher_results) else 71)
