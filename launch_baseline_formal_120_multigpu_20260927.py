"""将 120 场 SAFE-SIM/Scenario Dreamer 基线按场景分片到多张 GPU。"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path("/home2/zhaoyx/scenario-dreamer")
OUT = ROOT / "experiments/riskweaver_baseline_formal_120_multigpu_20260927_r1"
GENERATION_RESULT = ROOT / "experiments/scenario_dreamer_ldm_full60_20260927_r2/stage_result.json"
SOURCE_MANIFEST = ROOT / "experiments/scenario_dreamer_ldm_full60_20260927_r2/riskweaver_frozen_120_scenes.json"
PYTHON = Path("/home2/zhaoyx/miniconda3/envs/scenario-dreamer/bin/python")
SUPPORTED_METHODS = ("safe_sim", "scenario_dreamer")
EXPECTED_SCENARIOS = 120


def write_json(path, value):
    """保存严格 JSON 终态，便于并行分片监控。"""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def sha256(path):
    """计算清单内容哈希。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_manifest():
    """正式运行只接受 LDM 门禁通过的 120 场冻结清单。"""
    if not GENERATION_RESULT.is_file() or not SOURCE_MANIFEST.is_file():
        raise FileNotFoundError("120 场生成结果或冻结清单尚未就绪")
    generation = json.loads(GENERATION_RESULT.read_text(encoding="utf-8"))
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if generation.get("accepted") is not True:
        raise RuntimeError("LDM 生成结果未通过验收")
    if int(manifest.get("scenario_count") or 0) != EXPECTED_SCENARIOS:
        raise RuntimeError("冻结清单不是 120 场")
    if generation.get("frozen_manifest_sha256") != sha256(SOURCE_MANIFEST):
        raise RuntimeError("冻结清单哈希与生成门禁不一致")
    return generation, manifest


def split_manifest(manifest, shard_count, output_dir):
    """连续等量分片并保留全局索引；不根据场景结果分配。"""
    files = list(manifest["scenario_files"])
    digests = list(manifest["scenario_sha256"])
    index_groups = np_array_split(range(len(files)), shard_count)
    outputs = []
    for shard_id, indices in enumerate(index_groups):
        payload = dict(manifest)
        payload.update(
            {
                "name": f"{manifest.get('name', 'frozen')}_shard_{shard_id:02d}",
                "scenario_count": len(indices),
                "scenario_files": [files[index] for index in indices],
                "scenario_sha256": [digests[index] for index in indices],
                "original_scenario_indices": indices,
                "shard_id": shard_id,
                "shard_count": shard_count,
                "shard_rule": "contiguous_numpy_array_split_no_outcome_selection",
            }
        )
        path = output_dir / f"manifest_shard_{shard_id:02d}.json"
        write_json(path, payload)
        outputs.append((path, indices))
    return outputs


def np_array_split(values, shard_count):
    """无额外依赖地复现等量连续 array_split。"""
    values = list(values)
    quotient, remainder = divmod(len(values), shard_count)
    groups = []
    start = 0
    for shard_id in range(shard_count):
        size = quotient + int(shard_id < remainder)
        groups.append(values[start : start + size])
        start += size
    return groups


def environment(gpu_id):
    """每个子进程只看见一张物理 GPU，防止分片争用同一卡。"""
    return dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu_id),
        PROJECT_ROOT=str(ROOT),
        SCRATCH_ROOT=str(ROOT),
        DATASET_ROOT=str(ROOT / "metadata"),
        PYTHONPATH=f"{ROOT}:{ROOT / 'safe-sim'}:{ROOT / 'safe-sim/trajdata/src'}",
        MPLBACKEND="Agg",
        PYTHONUNBUFFERED="1",
    )


def shard_command(method, manifest, output_dir):
    """构造单分片原始方法命令。"""
    return [
        str(PYTHON),
        "experiments/comparison/run_passive_baseline.py",
        "--method",
        method,
        "--manifest",
        str(manifest),
        "--steps",
        "400",
        "--output-dir",
        str(output_dir),
    ]


def merge_method(method_dir, shards):
    """合并分片结果并验证 120 个全局索引恰好出现一次。"""
    from experiments.comparison.run_passive_baseline import aggregate, csv_fieldnames

    records = []
    seen = set()
    for shard_id, (_, global_indices) in enumerate(shards):
        path = method_dir / f"shard_{shard_id:02d}" / "results.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        episodes = report.get("episodes") or []
        if len(episodes) != len(global_indices):
            raise RuntimeError(f"{method} 分片 {shard_id} 场景数不完整")
        for row, global_index in zip(episodes, global_indices):
            if global_index in seen:
                raise RuntimeError(f"{method} 全局场景索引重复：{global_index}")
            seen.add(global_index)
            item = dict(row)
            item["shard_id"] = shard_id
            item["shard_local_index"] = item.get("scenario_index")
            item["scenario_index"] = global_index
            records.append(item)
    records.sort(key=lambda row: row["scenario_index"])
    if [row["scenario_index"] for row in records] != list(range(EXPECTED_SCENARIOS)):
        raise RuntimeError(f"{method} 合并后没有覆盖全部 120 场")
    report = {
        "method": method,
        "parallel_shards": len(shards),
        "aggregate": aggregate(records),
        "episodes": records,
    }
    write_json(method_dir / "results.json", report)
    csv_rows = [
        {key: value for key, value in row.items() if key not in {"danger_windows", "feasibility"}}
        for row in records
    ]
    with (method_dir / "results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fieldnames(csv_rows))
        writer.writeheader()
        writer.writerows(csv_rows)
    return report["aggregate"]


def run_method(method, gpu_ids, shards, state):
    """同一方法的场景分片并发运行，方法之间串行以便完整利用 GPU。"""
    method_dir = OUT / method
    method_dir.mkdir()
    processes = []
    started = time.time()
    for shard_id, ((manifest, indices), gpu_id) in enumerate(zip(shards, gpu_ids)):
        shard_dir = method_dir / f"shard_{shard_id:02d}"
        log_path = method_dir / f"shard_{shard_id:02d}.log"
        command = shard_command(method, manifest, shard_dir)
        log = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment(gpu_id),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        processes.append((process, log, gpu_id, indices, command, log_path))
    shard_results = []
    for shard_id, (process, log, gpu_id, indices, command, log_path) in enumerate(processes):
        exit_code = process.wait()
        log.close()
        shard_results.append(
            {
                "shard_id": shard_id,
                "gpu_id": gpu_id,
                "global_indices": indices,
                "exit_code": exit_code,
                "command": command,
                "log": str(log_path),
            }
        )
    valid_shards = all(item["exit_code"] == 0 for item in shard_results)
    aggregate = merge_method(method_dir, shards) if valid_shards else None
    valid = bool(
        valid_shards
        and aggregate
        and int(aggregate.get("attempted_scenario_count") or 0) == EXPECTED_SCENARIOS
        and int(aggregate.get("background_collision_frames") or 0) == 0
        and int(aggregate.get("background_static_collision_frames") or 0) == 0
    )
    result = {
        "method": method,
        "valid_for_analysis": valid,
        "elapsed_seconds": time.time() - started,
        "shards": shard_results,
        "aggregate": aggregate,
    }
    state["methods"].append(result)
    write_json(OUT / "launcher_result.partial.json", state)
    return valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-ids", default="1,2", help="逗号分隔的物理 GPU 编号")
    parser.add_argument("--methods", default=",".join(SUPPORTED_METHODS))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    gpu_ids = [int(value) for value in args.gpu_ids.split(",") if value.strip()]
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("GPU 编号不能为空或重复")
    if not methods or any(method not in SUPPORTED_METHODS for method in methods):
        raise ValueError(f"方法必须来自 {SUPPORTED_METHODS}")
    if args.validate_only:
        subprocess.run(
            [str(PYTHON), "experiments/comparison/run_passive_baseline.py", "--help"],
            cwd=ROOT,
            env=environment(gpu_ids[0]),
            stdout=subprocess.DEVNULL,
            check=True,
        )
        return 0

    generation, manifest_payload = require_manifest()
    if OUT.exists():
        raise FileExistsError(f"正式多 GPU 基线目录已存在：{OUT}")
    OUT.mkdir(parents=True)
    shutil.copy2(__file__, OUT / Path(__file__).name)
    shutil.copy2(SOURCE_MANIFEST, OUT / "scenario_manifest.json")
    shard_dir = OUT / "manifests"
    shard_dir.mkdir()
    shards = split_manifest(manifest_payload, len(gpu_ids), shard_dir)
    state = {
        "purpose": "120-scene multi-GPU SAFE-SIM and Scenario Dreamer comparison",
        "status": "running",
        "started_at": time.time(),
        "gpu_ids": gpu_ids,
        "methods_requested": methods,
        "manifest_sha256": sha256(SOURCE_MANIFEST),
        "no_outcome_based_selection": True,
        "generation_acceptance": generation,
        "methods": [],
    }
    write_json(OUT / "launcher_result.partial.json", state)
    for method in methods:
        if not run_method(method, gpu_ids, shards, state):
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
