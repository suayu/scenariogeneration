import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


ROOT = Path("/home2/zhaoyx/scenario-dreamer")
BASE = ROOT / "experiments/riskweaver_profile_formal_idm_rl_20260925_r5"
SOURCE_MANIFEST = ROOT / "work/riskweaver_profile_formal_manifest_20260923.json"
SOURCE_PROTOCOL = ROOT / "work/riskweaver_profile_formal_protocol_20260925_r5.md"
SMOKE_ACCEPTANCE = ROOT / "work/riskweaver_profile_deepseek_closed_loop_smoke_acceptance_20260925.json"
MODEL_NAME = "deepseek-v4-flash-0731"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def require_closed_loop_smoke():
    """正式实验只能复用通过真实闭环攻击门禁的同一模型。"""
    if not SMOKE_ACCEPTANCE.exists():
        raise RuntimeError(f"缺少闭环冒烟验收文件：{SMOKE_ACCEPTANCE}")
    payload = json.loads(SMOKE_ACCEPTANCE.read_text(encoding="utf-8"))
    required = {
        "accepted": True,
        "model_name": MODEL_NAME,
        "planner_service_failure_count": 0,
        "background_collision_count": 0,
        "background_static_collision_count": 0,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"闭环冒烟门禁 {key}={payload.get(key)!r}，期望 {expected!r}")
    if int(payload.get("attack_plan_count", 0)) <= 0:
        raise RuntimeError("闭环冒烟没有有效攻击计划")
    if int(payload.get("attack_executed_frames", 0)) <= 0:
        raise RuntimeError("闭环冒烟没有实际攻击执行帧")
    return payload


smoke_acceptance = require_closed_loop_smoke()
BASE.mkdir(parents=True, exist_ok=False)
shutil.copy2(__file__, BASE / Path(__file__).name)
shutil.copy2(SOURCE_MANIFEST, BASE / "scenario_manifest.json")
shutil.copy2(SOURCE_PROTOCOL, BASE / "protocol.md")
shutil.copy2(SMOKE_ACCEPTANCE, BASE / "smoke_acceptance.json")
instruction = BASE / "instruction.txt"
instruction.write_text(
    "Generate a physically executable adversarial interaction that probes the ego driving strategy. "
    "The attacker may leave the road topology. Preserve non-target background safety and a positive "
    "theoretical drivable region for the ego.\n",
    encoding="utf-8",
)
manifest = BASE / "scenario_manifest.json"
# 四组固定为同一个已通过真实闭环门禁的免费模型，避免模型分布混杂画像效应。
model_pool = [MODEL_NAME]
base_env = dict(
    os.environ,
    CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    PROJECT_ROOT=str(ROOT),
    SCRATCH_ROOT=str(ROOT),
    DATASET_ROOT=str(ROOT / "metadata"),
    PYTHONPATH=f"{ROOT}:{ROOT / 'safe-sim'}:{ROOT / 'safe-sim/trajdata/src'}",
    MPLBACKEND="Agg",
    PYTHONUNBUFFERED="1",
    HYDRA_FULL_ERROR="1",
    LLM_MODEL_NAMES=",".join(model_pool),
)
source_files = [
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
]
source_hashes = {
    name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    for name in source_files
}
groups = (
    ("idm_profile_off", "idm", False),
    ("idm_profile_on", "idm", True),
    ("rl_profile_off", "rl", False),
    ("rl_profile_on", "rl", True),
)
results = []
for group, policy, profile_enabled in groups:
    run = BASE / group
    run.mkdir()
    overrides = [
        f"sim.policy={policy}",
        "sim.seed=42",
        "+sim.continue_after_off_route=true",
        "sim.evaluation.max_scenarios=20",
        "sim.steps=400",
        f"+sim.scenario_manifest={manifest}",
        f"+sim.attack_request_file={instruction}",
        "sim.llm.provider=dashscope",
        f"sim.llm.model_name={MODEL_NAME}",
        f"sim.llm.model_names=[{','.join(model_pool)}]",
        "sim.llm.max_planning_calls=4",
        "sim.llm.profile_candidate_dry_run=false",
        "sim.llm.attack_mode=trajectory_only",
        "sim.llm.multiagent.mode=single",
        "sim.llm.obstacles.enabled=false",
        f"sim.traffic_model.iterative_adversarial.profile_enabled={str(profile_enabled).lower()}",
        "sim.traffic_model.iterative_adversarial.escalation_enabled=false",
        f"sim.traffic_model.iterative_adversarial.full_method_enabled={str(profile_enabled).lower()}",
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
    command = [
        "bash",
        "scripts/run_with_llm_provider.sh",
        "dashscope",
        "/home2/zhaoyx/miniconda3/envs/scenario-dreamer/bin/python",
        "run_simulation.py",
        *overrides,
    ]
    write_json(
        run / "launch.json",
        {
            "group": group,
            "policy": policy,
            "profile_enabled": profile_enabled,
            "purpose": "fresh frozen 20-scene ego-profile matrix after DeepSeek closed-loop attack smoke",
            "command": command,
            "started_at": time.time(),
            "source_hashes": source_hashes,
            "selection_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "protocol_sha256": hashlib.sha256((BASE / "protocol.md").read_bytes()).hexdigest(),
            "smoke_acceptance_sha256": hashlib.sha256((BASE / "smoke_acceptance.json").read_bytes()).hexdigest(),
            "model_pool": model_pool,
        },
    )
    with (run / "resolved_config.yaml").open("w", encoding="utf-8") as output:
        subprocess.run(command + ["--cfg", "job", "--resolve"], cwd=ROOT, env=base_env, stdout=output, check=True)
    started = time.time()
    with (run / "run.log").open("w", encoding="utf-8") as output:
        exit_code = subprocess.call(command, cwd=ROOT, env=base_env, stdout=output, stderr=subprocess.STDOUT)
    service_failures = None
    result_path = run / "movies/multi_scenario_ability_results.json"
    if exit_code == 0 and result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        service_failures = sum(
            int(row.get("planner_service_failure_count", 0))
            for row in payload.get("episodes", [])
        )
    valid = exit_code == 0 and service_failures == 0
    write_json(
        run / "exit_status.json",
        {
            "exit_code": exit_code,
            "started_at": started,
            "finished_at": time.time(),
            "planner_service_failure_count": service_failures,
            "valid_for_analysis": valid,
        },
    )
    results.append(
        {
            "group": group,
            "exit_code": exit_code,
            "planner_service_failure_count": service_failures,
            "valid_for_analysis": valid,
        }
    )
    write_json(BASE / "launcher_result.partial.json", {"groups": results, "updated_at": time.time()})
    if not valid:
        break

write_json(BASE / "launcher_result.json", {"groups": results, "finished_at": time.time()})
raise SystemExit(next((item["exit_code"] or 70 for item in results if not item["valid_for_analysis"]), 0))
