"""在统一冻结的 120 个场景上顺序运行两种被动基线。"""

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
OUT = ROOT / "experiments/riskweaver_baseline_formal_120_20260927_r1"
GENERATION_RESULT = ROOT / "experiments/scenario_dreamer_ldm_full60_20260927_r1/stage_result.json"
SOURCE_MANIFEST = ROOT / "experiments/scenario_dreamer_ldm_full60_20260927_r1/riskweaver_frozen_120_scenes.json"
PYTHON = Path("/home2/zhaoyx/miniconda3/envs/scenario-dreamer/bin/python")
METHODS = ("safe_sim", "scenario_dreamer")
EXPECTED_SCENARIOS = 120


def write_json(path, payload):
    """保存严格 JSON，便于中途监控和最终审计。"""
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def sha256(path):
    """计算冻结清单哈希，防止组间样本发生变化。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_manifest():
    """只接受通过 LDM 生成门禁的 120 场冻结清单。"""
    if not GENERATION_RESULT.is_file():
        raise FileNotFoundError(f"缺少场景生成结果：{GENERATION_RESULT}")
    generation = json.loads(GENERATION_RESULT.read_text(encoding="utf-8"))
    if generation.get("accepted") is not True:
        raise RuntimeError("LDM 场景生成尚未通过验收")
    if int(generation.get("frozen_scenario_count") or 0) != EXPECTED_SCENARIOS:
        raise RuntimeError("LDM 生成结果未冻结为 120 场")
    if not SOURCE_MANIFEST.is_file():
        raise FileNotFoundError(f"缺少冻结清单：{SOURCE_MANIFEST}")
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if int(manifest.get("scenario_count") or 0) != EXPECTED_SCENARIOS:
        raise RuntimeError("冻结清单场景数不是 120")
    expected_hash = generation.get("frozen_manifest_sha256")
    actual_hash = sha256(SOURCE_MANIFEST)
    if expected_hash != actual_hash:
        raise RuntimeError("冻结清单与 LDM 生成验收记录的哈希不一致")
    return generation, actual_hash


def command(method, manifest, output_dir):
    """构造两种基线完全一致预算的命令。"""
    return [
        str(PYTHON),
        "experiments/comparison/run_passive_baseline.py",
        "--method",
        method,
        "--manifest",
        str(manifest),
        "--steps",
        "400",
        "--max-scenarios",
        str(EXPECTED_SCENARIOS),
        "--output-dir",
        str(output_dir),
    ]


def environment():
    """复用服务器已有环境，不引入额外依赖。"""
    return dict(
        os.environ,
        PROJECT_ROOT=str(ROOT),
        SCRATCH_ROOT=str(ROOT),
        DATASET_ROOT=str(ROOT / "metadata"),
        PYTHONPATH=f"{ROOT}:{ROOT / 'safe-sim'}:{ROOT / 'safe-sim/trajdata/src'}",
        MPLBACKEND="Agg",
        PYTHONUNBUFFERED="1",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只验证依赖脚本可解析，不创建正式实验目录",
    )
    args = parser.parse_args()
    if args.validate_only:
        subprocess.run(
            [str(PYTHON), "experiments/comparison/run_passive_baseline.py", "--help"],
            cwd=ROOT,
            env=environment(),
            stdout=subprocess.DEVNULL,
            check=True,
        )
        return 0

    generation, manifest_hash = require_manifest()
    if OUT.exists():
        raise FileExistsError(f"正式基线目录已存在：{OUT}")
    OUT.mkdir(parents=True)
    shutil.copy2(__file__, OUT / Path(__file__).name)
    manifest = OUT / "scenario_manifest.json"
    shutil.copy2(SOURCE_MANIFEST, manifest)
    state = {
        "purpose": "120-scene frozen SAFE-SIM and Scenario Dreamer comparison",
        "status": "running",
        "started_at": time.time(),
        "expected_scenarios": EXPECTED_SCENARIOS,
        "manifest_sha256": manifest_hash,
        "no_outcome_based_selection": True,
        "generation_acceptance": generation,
        "methods": [],
    }
    write_json(OUT / "launcher_result.partial.json", state)
    for method in METHODS:
        method_dir = OUT / method
        log_path = OUT / f"{method}.log"
        run_command = command(method, manifest, method_dir)
        began = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                run_command,
                cwd=ROOT,
                env=environment(),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        results_path = method_dir / "results.json"
        aggregate = None
        if results_path.is_file():
            aggregate = json.loads(results_path.read_text(encoding="utf-8")).get("aggregate")
        valid = bool(
            completed.returncode == 0
            and aggregate
            and int(aggregate.get("attempted_scenario_count") or 0) == EXPECTED_SCENARIOS
            and int(aggregate.get("background_collision_frames") or 0) == 0
            and int(aggregate.get("background_static_collision_frames") or 0) == 0
        )
        item = {
            "method": method,
            "exit_code": completed.returncode,
            "valid_for_analysis": valid,
            "elapsed_seconds": time.time() - began,
            "output_dir": str(method_dir),
            "log": str(log_path),
            "aggregate": aggregate,
        }
        state["methods"].append(item)
        write_json(OUT / "launcher_result.partial.json", state)
        if not valid:
            state["status"] = "failed"
            break
    else:
        state["status"] = "complete"
    state["finished_at"] = time.time()
    state["elapsed_seconds"] = state["finished_at"] - state["started_at"]
    write_json(OUT / "launcher_result.json", state)
    return 0 if state["status"] == "complete" else 71


if __name__ == "__main__":
    sys.exit(main())
