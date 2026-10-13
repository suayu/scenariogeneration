import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path("/home2/zhaoyx/scenario-dreamer")
BASE = ROOT / "experiments/riskweaver_rl_profile_on_recovery_8_2_20260923"
SOURCE = ROOT / "experiments/riskweaver_rl_profile_paired_calibration_extension_20260923/scenario_manifest.json"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


BASE.mkdir(parents=True, exist_ok=False)
source = json.loads(SOURCE.read_text(encoding="utf-8"))
selected = [
    (path, digest)
    for path, digest in zip(source["scenario_files"], source["scenario_sha256"])
    if Path(path).name == "8_2.pkl"
]
assert len(selected) == 1
manifest = BASE / "scenario_manifest.json"
write_json(
    manifest,
    {
        "scenario_files": [selected[0][0]],
        "scenario_sha256": [selected[0][1]],
        "selection_basis": "recovery of the sole unexecuted profile-on scene after terminal aggregation KeyError",
        "parent_failed_run": str(SOURCE.parent),
    },
)
instruction = BASE / "instruction.txt"
instruction.write_text(
    "Generate a physically executable adversarial interaction that probes the ego driving strategy. "
    "The attacker may leave the road topology. Preserve non-target background safety and a positive "
    "theoretical drivable region for the ego.\n",
    encoding="utf-8",
)
model_pool = [
    "qwen-plus",
    "qwen-plus-0112",
    "qwen3-8b",
    "qwen3-max-preview",
    "qwen3.7-plus",
    "qwen3.6-plus",
    "qwen-max",
]
run = BASE / "profile_on"
run.mkdir()
env = dict(
    os.environ,
    CUDA_VISIBLE_DEVICES="0",
    PROJECT_ROOT=str(ROOT),
    SCRATCH_ROOT=str(ROOT),
    DATASET_ROOT=str(ROOT / "metadata"),
    PYTHONPATH=f"{ROOT}:{ROOT / 'safe-sim'}:{ROOT / 'safe-sim/trajdata/src'}",
    MPLBACKEND="Agg",
    PYTHONUNBUFFERED="1",
    HYDRA_FULL_ERROR="1",
    LLM_MODEL_NAMES=",".join(model_pool),
)
overrides = [
    "sim.policy=rl",
    "sim.seed=42",
    "+sim.continue_after_off_route=true",
    "sim.evaluation.max_scenarios=1",
    "sim.steps=45",
    f"+sim.scenario_manifest={manifest}",
    f"+sim.attack_request_file={instruction}",
    "sim.llm.provider=dashscope",
    "sim.llm.model_name=qwen-plus",
    f"sim.llm.model_names=[{','.join(model_pool)}]",
    "sim.llm.max_planning_calls=4",
    "sim.llm.profile_candidate_dry_run=false",
    "sim.llm.attack_mode=trajectory_only",
    "sim.llm.multiagent.mode=single",
    "sim.llm.obstacles.enabled=false",
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
command = [
    "bash",
    "scripts/run_with_llm_provider.sh",
    "dashscope",
    "/home2/zhaoyx/miniconda3/envs/scenario-dreamer/bin/python",
    "run_simulation.py",
    *overrides,
]
source_files = [
    "run_simulation.py",
    "policies/rl_policy.py",
    "policies/ego_profile.py",
    "policies/profile_planner.py",
    "policies/diffusion_model_wrapper.py",
    "policies/joint_safety.py",
]
write_json(
    run / "launch.json",
    {
        "group": "profile_on",
        "policy": "rl",
        "profile_enabled": True,
        "purpose": "recover missing 8_2 profile-on calibration scene only",
        "command": command,
        "started_at": time.time(),
        "source_hashes": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in source_files
        },
        "selection_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "model_pool": model_pool,
    },
)
with (run / "resolved_config.yaml").open("w", encoding="utf-8") as output:
    subprocess.run(command + ["--cfg", "job", "--resolve"], cwd=ROOT, env=env, stdout=output, check=True)
started = time.time()
with (run / "run.log").open("w", encoding="utf-8") as output:
    exit_code = subprocess.call(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
write_json(run / "exit_status.json", {"exit_code": exit_code, "started_at": started, "finished_at": time.time()})
write_json(BASE / "launcher_result.json", {"groups": [{"group": "profile_on", "exit_code": exit_code}], "finished_at": time.time()})
raise SystemExit(exit_code)
