"""E0 数据集划分与 E1--E5 共用统计工具；不会直接启动仿真。"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import pickle
import random
from collections import defaultdict
from pathlib import Path


def _scene_key(path, scene):
    """优先使用数据内稳定 ID，缺失时回退到文件名。"""
    return str(next((scene[key] for key in ("scene_id", "scenario_id", "id") if scene.get(key) is not None), path.stem))


def _features(scene):
    """记录起始场景统计，供数据集审计；不使用生成器或 LLM。"""
    agents, kinds = scene.get("agents"), scene.get("agent_types")
    if agents is None or len(agents) == 0:
        return {"agent_count": 0, "vehicle_count": 0, "ego_speed_mps": 0.0, "nearest_vehicle_m": None}
    states, ego = agents[:, 0], agents[:, 0][-1]
    mask = kinds[:-1, 1] == 1 if kinds is not None and len(kinds) == len(states) else [True] * (len(states) - 1)
    vehicles = states[:-1][mask]
    distances = [float(((car[0] - ego[0]) ** 2 + (car[1] - ego[1]) ** 2) ** .5) for car in vehicles]
    return {"agent_count": int(len(states)), "vehicle_count": int(len(vehicles)), "ego_speed_mps": float((ego[2] ** 2 + ego[3] ** 2) ** .5), "nearest_vehicle_m": min(distances) if distances else None}


def build_manifest(dataset_path, split_seed):
    """以稳定哈希生成精确、互斥且可复现的 60/20/20 划分。"""
    files = sorted(Path(dataset_path).glob("*.pkl"))
    if len(files) < 5:
        raise ValueError("至少需要 5 个 pickle 场景才能建立开发/验证/测试划分")
    rows = []
    for index, path in enumerate(files):
        with path.open("rb") as handle:
            scene = pickle.load(handle)
        scene_id = _scene_key(path, scene)
        rank = hashlib.sha256(f"{split_seed}:{scene_id}".encode()).hexdigest()
        rows.append((rank, index, scene_id, path, _features(scene)))
    rows.sort(key=lambda item: (item[0], item[2]))
    dev_end, val_end = max(1, round(len(rows) * .6)), max(max(1, round(len(rows) * .6)) + 1, round(len(rows) * .8))
    manifest = []
    for rank, (_, index, scene_id, path, features) in enumerate(rows):
        split = "development" if rank < dev_end else "validation" if rank < val_end else "test"
        manifest.append({"scenario_index": index, "scenario_id": scene_id, "scenario_file": str(path.resolve()), "split": split, **features})
    return sorted(manifest, key=lambda item: item["scenario_index"])


def _percentile(values, q):
    values = sorted(values)
    if not values:
        return float("nan")
    pos = (len(values) - 1) * q / 100
    lower, upper = math.floor(pos), math.ceil(pos)
    return values[lower] + (values[upper] - values[lower]) * (pos - lower)


def _ci(values, seed, count):
    if not values:
        return float("nan"), float("nan")
    rng, size = random.Random(seed), len(values)
    means = sorted(sum(values[rng.randrange(size)] for _ in range(size)) / size for _ in range(count))
    return _percentile(means, 2.5), _percentile(means, 97.5)


def summarize_episodes(rows, bootstrap_seed=20260831, bootstrap_samples=2000):
    """输出各 split、方法、策略下的均值、P50/P95 和 bootstrap CI。"""
    groups = defaultdict(list)
    for row in rows:
        groups[(str(row["method"]), str(row["policy"]), str(row["split"]))].append(row)
    ignored = {"method", "policy", "split", "scenario_id", "scenario_index", "seed", "video_path"}
    output = []
    for index, ((method, policy, split), items) in enumerate(sorted(groups.items())):
        metrics = defaultdict(list)
        for item in items:
            for key, value in item.items():
                if key not in ignored and isinstance(value, (int, float)) and math.isfinite(float(value)):
                    metrics[key].append(float(value))
        result = {"method": method, "policy": policy, "split": split, "episodes": len(items), "unique_scenarios": len({item.get("scenario_id") for item in items}), "seeds": len({item.get("seed") for item in items})}
        for key, values in metrics.items():
            low, high = _ci(values, bootstrap_seed + index, bootstrap_samples)
            result.update({f"{key}_mean": sum(values) / len(values), f"{key}_p50": _percentile(values, 50), f"{key}_p95": _percentile(values, 95), f"{key}_ci95_low": low, f"{key}_ci95_high": high})
        output.append(result)
    return output


def _read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Scenario-Dreamer 实验协议工具")
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("build-manifest")
    manifest.add_argument("--dataset-path", required=True)
    manifest.add_argument("--output", required=True, type=Path)
    manifest.add_argument("--split-seed", type=int, default=20260831)
    summary = commands.add_parser("summarize")
    summary.add_argument("--episodes", required=True)
    summary.add_argument("--output-json", required=True, type=Path)
    summary.add_argument("--output-csv", required=True, type=Path)
    summary.add_argument("--bootstrap-seed", type=int, default=20260831)
    summary.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.command == "build-manifest":
        payload = {"protocol_version": 1, "split_seed": args.split_seed, "scenarios": build_manifest(args.dataset_path, args.split_seed)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        result = {"scenarios": len(payload["scenarios"]), "output": str(args.output)}
    elif args.command == "summarize":
        payload = summarize_episodes(_read_jsonl(args.episodes), args.bootstrap_seed, args.bootstrap_samples)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in payload for key in row}))
            writer.writeheader()
            writer.writerows(payload)
        result = {"groups": len(payload), "json": str(args.output_json), "csv": str(args.output_csv)}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
