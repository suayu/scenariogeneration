"""运行锚点增强与静态风险的 20 场最小消融。"""

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
BASE = ROOT / "experiments/riskweaver_ablation_formal_20260924_r1"
MANIFEST_SOURCE = ROOT / "work/riskweaver_profile_formal_manifest_20260923.json"
PROTOCOL_SOURCE = ROOT / "work/riskweaver_ablation_formal_protocol_20260924.md"
MODEL = "qwen3-coder-plus"
PYTHON = Path("/home2/zhaoyx/miniconda3/envs/scenario-dreamer/bin/python")
GROUPS = (
    ("anchor_uniform", "trajectory_only", False, False),
    ("joint_dynamic_static", "joint", True, True),
)


def write_json(path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def env():
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


def command(manifest, instruction, run, attack_mode, obstacles, anchor_drive):
    overrides = [
        "sim.policy=idm",
        "sim.seed=42",
        "+sim.continue_after_off_route=true",
        "sim.evaluation.max_scenarios=20",
        "sim.steps=400",
        f"+sim.scenario_manifest={manifest}",
        f"+sim.attack_request_file={instruction}",
        "sim.llm.provider=dashscope",
        f"sim.llm.model_name={MODEL}",
        f"sim.llm.model_names=[{MODEL}]",
        "sim.llm.max_planning_calls=4",
        "sim.llm.profile_candidate_dry_run=false",
        f"sim.llm.attack_mode={attack_mode}",
        "sim.llm.multiagent.mode=single",
        f"sim.llm.obstacles.enabled={str(obstacles).lower()}",
        "sim.traffic_model.iterative_adversarial.profile_enabled=true",
        "sim.traffic_model.iterative_adversarial.escalation_enabled=false",
        "sim.traffic_model.iterative_adversarial.full_method_enabled=true",
        "sim.traffic_model.guidance.mode=llm_joint",
        "sim.traffic_model.dynamics_projection.enabled=true",
        f"sim.traffic_model.anchor_guidance.anchor_drive_enabled={str(anchor_drive).lower()}",
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
        "bash", "scripts/run_with_llm_provider.sh", "dashscope", str(PYTHON),
        "run_simulation.py", *overrides,
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        for name, mode, obstacles, anchor_drive in GROUPS:
            cmd = command(MANIFEST_SOURCE, PROTOCOL_SOURCE, BASE / name, mode, obstacles, anchor_drive)
            completed = subprocess.run(
                cmd + ["--cfg", "job", "--resolve"], cwd=ROOT, env=env(),
                stdout=subprocess.DEVNULL, check=False,
            )
            if completed.returncode:
                return completed.returncode
        return 0

    if BASE.exists():
        raise FileExistsError(f"正式消融目录已存在：{BASE}")
    BASE.mkdir(parents=True)
    shutil.copy2(__file__, BASE / Path(__file__).name)
    shutil.copy2(MANIFEST_SOURCE, BASE / "scenario_manifest.json")
    shutil.copy2(PROTOCOL_SOURCE, BASE / "protocol.md")
    instruction = BASE / "instruction.txt"
    instruction.write_text(
        "Generate a physically executable adversarial interaction that probes the ego driving strategy. "
        "Select only supplied validated candidates. The attacker may leave the road topology. Preserve "
        "non-target background safety and a positive theoretical drivable region for the ego.\n",
        encoding="utf-8",
    )
    manifest = BASE / "scenario_manifest.json"
    sources = (
        "run_simulation.py", "simulator.py", "scenario_generator.py",
        "policies/ego_profile.py", "policies/profile_planner.py",
        "policies/llm_adversarial_planner.py", "policies/diffusion_model_wrapper.py",
        "policies/llm_anchor_guidance.py", "policies/joint_safety.py",
        "policies/profile_obstacles.py",
    )
    source_hashes = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in sources
    }
    results = []
    for name, mode, obstacles, anchor_drive in GROUPS:
        run = BASE / name
        run.mkdir()
        cmd = command(manifest, instruction, run, mode, obstacles, anchor_drive)
        write_json(run / "launch.json", {
            "group": name,
            "attack_mode": mode,
            "obstacles_enabled": obstacles,
            "anchor_drive_enabled": anchor_drive,
            "reference_group": (
                "riskweaver_profile_formal_idm_rl_20260924_r4/idm_profile_on"
            ),
            "model": MODEL,
            "command": cmd,
            "started_at": time.time(),
            "source_hashes": source_hashes,
            "selection_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        })
        with (run / "resolved_config.yaml").open("w", encoding="utf-8") as output:
            subprocess.run(
                cmd + ["--cfg", "job", "--resolve"], cwd=ROOT, env=env(),
                stdout=output, check=True,
            )
        started = time.time()
        with (run / "run.log").open("w", encoding="utf-8") as output:
            exit_code = subprocess.call(
                cmd, cwd=ROOT, env=env(), stdout=output, stderr=subprocess.STDOUT
            )
        result_path = run / "movies/multi_scenario_ability_results.json"
        failures = None
        if exit_code == 0 and result_path.exists():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            failures = sum(
                int(row.get("planner_service_failure_count", 0) or 0)
                for row in payload.get("episodes", [])
            )
        valid = exit_code == 0 and failures == 0
        item = {
            "group": name,
            "exit_code": exit_code,
            "planner_service_failure_count": failures,
            "valid_for_analysis": valid,
            "started_at": started,
            "finished_at": time.time(),
        }
        write_json(run / "exit_status.json", item)
        results.append(item)
        write_json(BASE / "launcher_result.partial.json", {"groups": results})
        if not valid:
            break
    write_json(BASE / "launcher_result.json", {"groups": results})
    return next((item["exit_code"] or 70 for item in results if not item["valid_for_analysis"]), 0)


if __name__ == "__main__":
    sys.exit(main())
