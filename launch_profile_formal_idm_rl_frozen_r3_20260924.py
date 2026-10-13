import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


ROOT = Path("/home2/zhaoyx/scenario-dreamer")
BASE = ROOT / "experiments/riskweaver_profile_formal_idm_rl_20260924_r3"
SOURCE_MANIFEST = ROOT / "work/riskweaver_profile_formal_manifest_20260923.json"
SOURCE_PROTOCOL = ROOT / "work/riskweaver_profile_formal_protocol_20260924_r3.md"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


BASE.mkdir(parents=True, exist_ok=False)
shutil.copy2(__file__, BASE / Path(__file__).name)
shutil.copy2(SOURCE_MANIFEST, BASE / "scenario_manifest.json")
shutil.copy2(SOURCE_PROTOCOL, BASE / "protocol.md")
instruction = BASE / "instruction.txt"
instruction.write_text(
    "Generate a physically executable adversarial interaction that probes the ego driving strategy. "
    "The attacker may leave the road topology. Preserve non-target background safety and a positive "
    "theoretical drivable region for the ego.\n",
    encoding="utf-8",
)
manifest = BASE / "scenario_manifest.json"
# 四组冻结为同一个已通过真实排序 smoke 的免费模型，避免模型分布混杂画像效应。
model_pool = ["qwen3.5-flash-2026-02-23"]
base_env = dict(
    os.environ,
    # 允许调度器把正式组放到独立 GPU；未指定时保持原来的 GPU 0 行为。
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
        "sim.llm.model_name=qwen3.5-flash-2026-02-23",
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
            "purpose": "frozen 20-scene formal ego-profile experiment after full planner JSON and validation smoke",
            "command": command,
            "started_at": time.time(),
            "source_hashes": source_hashes,
            "selection_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "protocol_sha256": hashlib.sha256((BASE / "protocol.md").read_bytes()).hexdigest(),
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
