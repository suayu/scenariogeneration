"""从真实 Waymo log-replay 场景构建 SAFE-SIM 兼容真实性参考直方图。"""

import argparse
import hashlib
import json
import pickle
from pathlib import Path, PurePosixPath
import tarfile

import numpy as np

from policies.realism_audit import DEFAULT_BINS, _DomainAccumulator


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_reference_payloads(sources, dt, dataset_label):
    """汇总所有真实背景车辆轨迹，保留固定分箱外比例以便审计。"""
    accumulator = _DomainAccumulator()
    source_hashes = []
    accepted_files = 0
    for file_index, (source_name, raw_payload) in enumerate(sources):
        payload = pickle.loads(raw_payload)
        agents = np.asarray(payload.get("agents"), dtype=np.float64)
        if agents.ndim != 3 or agents.shape[-1] < 5 or agents.shape[1] < 4:
            continue
        ego_index = int(payload.get("ego_index", len(agents) - 1))
        for step in range(agents.shape[1]):
            frame = agents[:, step]
            valid = np.isfinite(frame[:, :5]).all(axis=1)
            if frame.shape[1] > 7:
                valid &= frame[:, 7] > 0
            valid[ego_index] = False
            for agent_id in np.flatnonzero(valid):
                accumulator.observe((file_index, int(agent_id)), step, frame[agent_id], dt)
        source_hashes.append({
            "path": str(source_name),
            "sha256": hashlib.sha256(raw_payload).hexdigest(),
        })
        accepted_files += 1

    stats = {}
    ticks = {}
    counts = {}
    range_audit = {}
    for name, edges in DEFAULT_BINS.items():
        values = np.asarray(accumulator.values[name], dtype=np.float64)
        values = values[np.isfinite(values)]
        histogram = np.histogram(values, bins=edges)[0].astype(np.int64)
        stats[name] = histogram.tolist()
        ticks[name] = edges.tolist()
        counts[name] = int(len(values))
        range_audit[name] = {
            "below_count": int(np.sum(values < edges[0])),
            "above_count": int(np.sum(values > edges[-1])),
            "in_range_count": int(histogram.sum()),
        }
    if accepted_files == 0 or any(sum(stats[name]) == 0 for name in DEFAULT_BINS):
        raise ValueError("没有得到足够的真实轨迹统计，请检查输入是否为完整 log-replay 场景")
    manifest_digest = hashlib.sha256(
        "".join(item["sha256"] for item in source_hashes).encode("ascii")
    ).hexdigest()
    return {
        "schema_version": 1,
        "dataset_label": dataset_label,
        "dt_seconds": float(dt),
        "file_count": accepted_files,
        "source_manifest_sha256": manifest_digest,
        "sources": source_hashes,
        "stats": stats,
        "ticks": ticks,
        "sample_counts": counts,
        "range_audit": range_audit,
        "definition": "SAFE-SIM fixed-bin histograms over real background trajectories",
    }


def build_reference(paths, dt, dataset_label):
    """从已解压的 pickle 路径构建参考，保留原有调用接口。"""
    return _build_reference_payloads(
        ((str(path), path.read_bytes()) for path in paths), dt, dataset_label
    )


def build_reference_archive(archive_path, dt, dataset_label,
                            archive_prefix="waymo_sim_test_pickles"):
    """直接读取官方 tar 中的 Waymo log-replay pickle，不解压整份归档。"""
    with tarfile.open(archive_path, "r:*") as archive:
        members = [
            member for member in archive.getmembers()
            if member.isfile()
            and PurePosixPath(member.name).suffix.lower() in {".pkl", ".pickle"}
            and (
                not archive_prefix
                or archive_prefix in PurePosixPath(member.name).parts
            )
        ]
        members.sort(key=lambda member: member.name)
        if not members:
            raise FileNotFoundError(
                f"归档 {archive_path} 中未找到 {archive_prefix} 下的 pickle"
            )

        def payloads():
            for member in members:
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                # 只读取成员内容，不写入文件系统，避免路径穿越和整包解压。
                yield member.name, handle.read()

        return _build_reference_payloads(payloads(), dt, dataset_label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="真实 pickle 目录或 Scenario Dreamer 官方 tar")
    parser.add_argument("output", type=Path, help="输出 JSON 路径")
    parser.add_argument("--pattern", default="*.pkl")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--dataset-label", default="Waymo log-replay test")
    parser.add_argument(
        "--archive-prefix", default="waymo_sim_test_pickles",
        help="输入为 tar 时只读取该目录组件下的 pickle",
    )
    args = parser.parse_args()
    if args.input.is_file():
        result = build_reference_archive(
            args.input, args.dt, args.dataset_label, args.archive_prefix
        )
    else:
        paths = sorted(args.input.glob(args.pattern))
        if not paths:
            raise FileNotFoundError(f"未在 {args.input} 找到 {args.pattern}")
        result = build_reference(paths, args.dt, args.dataset_label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "file_count": result["file_count"],
        "sample_counts": result["sample_counts"],
        "source_manifest_sha256": result["source_manifest_sha256"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
