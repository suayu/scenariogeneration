import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


ROOT = Path("/home2/zhaoyx/scenario-dreamer")
BASE = ROOT / "experiments/riskweaver_rl_profile_paired_calibration_extension_20260923"
SOURCE_MANIFEST = (
    ROOT
    / "experiments/riskweaver_profile_paired_pilot_20260923_r1/scenario_manifest.json"
)
CALIBRATION_IDS = {"12_7.pkl", "31_11.pkl", "8_2.pkl"}


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


# 剩余三场来自同一开发先导集，只用于补足六场 RL 校准，不进入正式论文统计。
BASE.mkdir(parents=True, exist_ok=False)
shutil.copy2(__file__, BASE / Path(__file__).name)
source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
selected = [
    (path, digest)
    for path, digest in zip(
        source_manifest["scenario_files"], source_manifest["scenario_sha256"]
    )
    if Path(path).name in CALIBRATION_IDS
]
assert len(selected) == 3
manifest = BASE / "scenario_manifest.json"
write_json(
    manifest,
    {
        "scenario_files": [item[0] for item in selected],
        "scenario_sha256": [item[1] for item in selected],
        "selection_basis": "remaining three current-development scenes; six-scene calibration extension only",
    },
)
instruction = BASE / "instruction.txt"
instruction.write_text(
    "Generate a physically executable adversarial interaction that probes the ego driving strategy. "
    "The attacker may leave the road topology. Preserve non-target background safety and a positive "
    "theoretical drivable region for the ego.\n",
    encoding="utf-8",
)

base_env = dict(
    os.environ,
    CUDA_VISIBLE_DEVICES="0",
    PROJECT_ROOT=str(ROOT),
    SCRATCH_ROOT=str(ROOT),
    DATASET_ROOT=str(ROOT / "metadata"),
    PYTHONPATH=f"{ROOT}:{ROOT / 'safe-sim'}:{ROOT / 'safe-sim/trajdata/src'}",
    MPLBACKEND="Agg",
    PYTHONUNBUFFERED="1",
    HYDRA_FULL_ERROR="1",
)
model_pool = [
    "qwen3.7-max",
    "qwen-plus",
    "qwen-plus-0112",
    "qwen3-8b",
    "qwen3-max-preview",
    "qwen3.7-plus",
    "qwen3.6-plus",
    "qwen-max",
]
base_env["LLM_MODEL_NAMES"] = ",".join(model_pool)
source_files = [
    "run_simulation.py",
    "policies/rl_policy.py",
    "policies/ego_profile.py",
    "policies/profile_planner.py",
    "policies/diffusion_model_wrapper.py",
    "policies/joint_safety.py",
]
source_hashes = {
    name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    for name in source_files
}
results = []
for group, profile_enabled in (("profile_off", False), ("profile_on", True)):
    run = BASE / group
    run.mkdir()
    overrides = [
        "sim.policy=rl",
        "sim.seed=42",
        "+sim.continue_after_off_route=true",
        "sim.evaluation.max_scenarios=3",
        "sim.steps=45",
        f"+sim.scenario_manifest={manifest}",
        f"+sim.attack_request_file={instruction}",
        "sim.llm.provider=dashscope",
        "sim.llm.model_name=qwen3.7-max",
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
            "policy": "rl",
            "profile_enabled": profile_enabled,
            "purpose": "remaining three-scene extension; combined six-scene calibration only",
            "command": command,
            "started_at": time.time(),
            "source_hashes": source_hashes,
            "selection_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "model_pool": model_pool,
        },
    )
    with (run / "resolved_config.yaml").open("w", encoding="utf-8") as output:
        subprocess.run(
            command + ["--cfg", "job", "--resolve"],
            cwd=ROOT,
            env=base_env,
            stdout=output,
            check=True,
        )
    started = time.time()
    with (run / "run.log").open("w", encoding="utf-8") as output:
        exit_code = subprocess.call(
            command,
            cwd=ROOT,
            env=base_env,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    write_json(
        run / "exit_status.json",
        {"exit_code": exit_code, "started_at": started, "finished_at": time.time()},
    )
    results.append({"group": group, "exit_code": exit_code})
    if exit_code:
        break

write_json(BASE / "launcher_result.json", {"groups": results, "finished_at": time.time()})
raise SystemExit(next((item["exit_code"] for item in results if item["exit_code"]), 0))
