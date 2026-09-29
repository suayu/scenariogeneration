"""顺序运行冻结的 SAFE-SIM 与 Scenario Dreamer 正式被动基线。"""

import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiments" / "riskweaver_baseline_formal_20260924"
MANIFEST = ROOT / "work" / "riskweaver_profile_formal_manifest_20260923.json"
PYTHON = Path("/home2/zhaoyx/miniconda3/envs/scenario-dreamer/bin/python")
METHODS = ("safe_sim", "scenario_dreamer")


def write_json(path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    if OUT.exists():
        raise FileExistsError(f"正式基线目录已存在：{OUT}")
    OUT.mkdir(parents=True)
    started = time.time()
    state = {
        "started_at_unix": started,
        "manifest": str(MANIFEST),
        "methods": [],
        "status": "running",
    }
    write_json(OUT / "launcher_result.partial.json", state)
    env = dict(
        os.environ,
        PROJECT_ROOT=str(ROOT),
        SCRATCH_ROOT=str(ROOT),
        DATASET_ROOT=str(ROOT / "metadata"),
        PYTHONPATH=f"{ROOT}:{ROOT / 'safe-sim'}:{ROOT / 'safe-sim/trajdata/src'}",
    )
    for method in METHODS:
        method_dir = OUT / method
        log_path = OUT / f"{method}.log"
        command = [
            str(PYTHON),
            "experiments/comparison/run_passive_baseline.py",
            "--method", method,
            "--manifest", str(MANIFEST),
            "--steps", "400",
            "--output-dir", str(method_dir),
        ]
        began = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, check=False,
            )
        item = {
            "method": method,
            "exit_code": completed.returncode,
            "elapsed_seconds": time.time() - began,
            "output_dir": str(method_dir),
            "log": str(log_path),
        }
        state["methods"].append(item)
        write_json(OUT / "launcher_result.partial.json", state)
        if completed.returncode != 0:
            state["status"] = "failed"
            break
    else:
        state["status"] = "complete"
    state["elapsed_seconds"] = time.time() - started
    write_json(OUT / "launcher_result.json", state)
    return 0 if state["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
