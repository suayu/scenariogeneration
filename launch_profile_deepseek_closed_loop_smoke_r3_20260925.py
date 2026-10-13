"""冻结的 DeepSeek 单场闭环攻击冒烟；通过后生成 R5 通行证。"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import time


ROOT = Path("/home2/zhaoyx/scenario-dreamer")
OUT = ROOT / "experiments/riskweaver_profile_deepseek_closed_loop_smoke_20260925_r3"
SOURCE_MANIFEST = ROOT / "work/riskweaver_profile_closed_loop_smoke_scenario20_8_20260925.json"
ACCEPTANCE = ROOT / "work/riskweaver_profile_deepseek_closed_loop_smoke_acceptance_20260925.json"
MODEL_NAME = "deepseek-v4-flash-0731"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


if ACCEPTANCE.exists():
    raise RuntimeError(f"验收文件已存在，拒绝覆盖：{ACCEPTANCE}")
OUT.mkdir(parents=True, exist_ok=False)
shutil.copy2(__file__, OUT / Path(__file__).name)
shutil.copy2(SOURCE_MANIFEST, OUT / "scenario_manifest.json")
instruction = OUT / "instruction.txt"
instruction.write_text(
    "Generate a physically executable adversarial interaction that probes the ego driving strategy. "
    "The attacker may leave the road topology. Preserve non-target background safety and a positive "
    "theoretical drivable region for the ego.\n",
    encoding="utf-8",
)
env = dict(
    os.environ,
    CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    PROJECT_ROOT=str(ROOT),
    SCRATCH_ROOT=str(ROOT),
    DATASET_ROOT=str(ROOT / "metadata"),
    PYTHONPATH=f"{ROOT}:{ROOT / 'safe-sim'}:{ROOT / 'safe-sim/trajdata/src'}",
    MPLBACKEND="Agg",
    PYTHONUNBUFFERED="1",
    HYDRA_FULL_ERROR="1",
    RISKWEAVER_DIAGNOSTIC_DIR=str(OUT / "diagnostics"),
    LLM_MODEL_NAMES=MODEL_NAME,
)
overrides = [
    "sim.policy=idm",
    "sim.seed=42",
    "+sim.continue_after_off_route=true",
    "sim.evaluation.max_scenarios=1",
    "sim.steps=400",
    f"+sim.scenario_manifest={OUT / 'scenario_manifest.json'}",
    f"+sim.attack_request_file={instruction}",
    "sim.llm.provider=dashscope",
    f"sim.llm.model_name={MODEL_NAME}",
    f"sim.llm.model_names=[{MODEL_NAME}]",
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
    "sim.evaluation.difficulty_control.mode=target",
    "sim.evaluation.difficulty_control.target_difficulty=0.50",
    "sim.evaluation.difficulty_control.tolerance=0.10",
    "sim.evaluation.difficulty_control.max_replays=0",
    "sim.evaluation.difficulty_control.smoke_acceptance=true",
    "sim.visualize=true",
    "sim.lightweight=true",
    f"sim.movie_path={OUT / 'movies'}",
    f"sim.scenario_data_output_path={OUT / 'carla'}",
    f"hydra.run.dir={OUT / 'hydra'}",
]
command = [
    "bash", "scripts/run_with_llm_provider.sh", "dashscope",
    "/home2/zhaoyx/miniconda3/envs/scenario-dreamer/bin/python", "run_simulation.py",
    *overrides,
]
write_json(
    OUT / "launch.json",
    {
        "command": command,
        "model_name": MODEL_NAME,
        "selection_rule": "pre-frozen scenario index 6 with prior executed attack evidence",
        "started_at": time.time(),
    },
)
with (OUT / "resolved_config.yaml").open("w", encoding="utf-8") as output:
    subprocess.run(command + ["--cfg", "job", "--resolve"], cwd=ROOT, env=env, stdout=output, check=True)
started_at = time.time()
with (OUT / "run.log").open("w", encoding="utf-8") as output:
    simulation_exit_code = subprocess.call(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)

candidate = OUT / "smoke_acceptance_candidate.json"
validation_log = OUT / "acceptance_validation.log"
validation_command = [
    "/home2/zhaoyx/miniconda3/envs/scenario-dreamer/bin/python",
    "scripts/validate_closed_loop_smoke.py",
    "--smoke-dir", str(OUT),
    "--expected-model", MODEL_NAME,
    "--output", str(candidate),
]
with validation_log.open("w", encoding="utf-8") as output:
    validation_exit_code = subprocess.call(
        validation_command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT
    )
accepted = simulation_exit_code == 0 and validation_exit_code == 0 and candidate.is_file()
if accepted:
    # 通行证只在仿真退出和独立验收都成功时发布，且拒绝覆盖既有证据。
    os.link(candidate, ACCEPTANCE)
write_json(
    OUT / "exit_status.json",
    {
        "simulation_exit_code": simulation_exit_code,
        "validation_exit_code": validation_exit_code,
        "accepted": accepted,
        "started_at": started_at,
        "finished_at": time.time(),
    },
)
raise SystemExit(0 if accepted else (simulation_exit_code or validation_exit_code or 71))
