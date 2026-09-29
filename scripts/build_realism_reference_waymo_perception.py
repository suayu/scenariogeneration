"""从 Waymo Perception 连续帧构建 SAFE-SIM 兼容的真实轨迹参考分布。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from policies.realism_audit import DEFAULT_BINS, _DomainAccumulator


def _scene_state(frame, label, origin_inverse, origin_yaw):
    """转换到首帧自车坐标系，去除自车运动并匹配仿真场景朝向。"""
    transform = np.asarray(frame.pose.transform, dtype=np.float64).reshape(4, 4)
    local = np.array(
        [label.box.center_x, label.box.center_y, label.box.center_z, 1.0],
        dtype=np.float64,
    )
    world = transform @ local
    scene = origin_inverse @ world
    ego_yaw = float(np.arctan2(transform[1, 0], transform[0, 0]))
    scene_yaw = ego_yaw + label.box.heading - origin_yaw
    scene_yaw = float((scene_yaw + np.pi) % (2.0 * np.pi) - np.pi)
    return np.array(
        [scene[0], scene[1], 0.0, 0.0, scene_yaw],
        dtype=np.float64,
    )


def _histogram_payload(accumulator):
    stats = {}
    ticks = {}
    counts = {}
    range_audit = {}
    for name, edges in DEFAULT_BINS.items():
        values = np.asarray(accumulator.values[name], dtype=np.float64)
        values = values[np.isfinite(values)]
        # 刚体坐标旋转会产生约 1e-14 的浮点误差；量化后避免样本跨越分箱边界。
        values = np.round(values, decimals=10)
        histogram = np.histogram(values, bins=edges)[0].astype(np.int64)
        stats[name] = histogram.tolist()
        ticks[name] = edges.tolist()
        counts[name] = int(len(values))
        range_audit[name] = {
            "below_count": int(np.sum(values < edges[0])),
            "above_count": int(np.sum(values > edges[-1])),
            "in_range_count": int(histogram.sum()),
        }
    return stats, ticks, counts, range_audit


def build_reference_from_frames(frame_sources, dt, dataset_label, vehicle_type=1):
    """处理已解析帧；该入口也用于不依赖 TensorFlow 的单元测试。"""
    accumulator = _DomainAccumulator()
    source_audit = []
    accepted_frames = 0
    accepted_labels = 0
    continuous_intervals = 0
    discontinuous_intervals = 0

    for source_name, frames, source_digest in frame_sources:
        previous_timestamp = None
        origin_inverse = None
        origin_yaw = None
        source_frames = 0
        source_labels = 0
        for frame_index, frame in enumerate(frames):
            if origin_inverse is None:
                origin = np.asarray(frame.pose.transform, dtype=np.float64).reshape(4, 4)
                origin_inverse = np.linalg.inv(origin)
                origin_yaw = float(np.arctan2(origin[1, 0], origin[0, 0]))
            timestamp = int(frame.timestamp_micros)
            continuous = previous_timestamp is None or np.isclose(
                (timestamp - previous_timestamp) / 1_000_000.0,
                dt,
                rtol=0.0,
                atol=1e-3,
            )
            if previous_timestamp is not None:
                if continuous:
                    continuous_intervals += 1
                else:
                    discontinuous_intervals += 1
                    # 时间断点后的差分没有物理意义，清除轨迹历史。
                    accumulator.previous.clear()
            previous_timestamp = timestamp

            context_name = getattr(getattr(frame, "context", None), "name", "")
            source_track_prefix = context_name or str(source_name)
            for label in frame.laser_labels:
                if int(label.type) != int(vehicle_type):
                    continue
                if int(getattr(label, "num_lidar_points_in_box", 1)) <= 0:
                    continue
                state = _scene_state(frame, label, origin_inverse, origin_yaw)
                if not np.isfinite(state).all():
                    continue
                accumulator.observe(
                    (source_track_prefix, str(label.id)), frame_index, state, dt
                )
                source_labels += 1
                accepted_labels += 1
            source_frames += 1
            accepted_frames += 1

        source_audit.append(
            {
                "path": str(source_name),
                "record_payload_sha256": (
                    source_digest() if callable(source_digest) else source_digest
                ),
                "frame_count": source_frames,
                "vehicle_label_count": source_labels,
            }
        )

    stats, ticks, counts, range_audit = _histogram_payload(accumulator)
    if accepted_frames == 0 or any(sum(stats[name]) == 0 for name in DEFAULT_BINS):
        raise ValueError("没有得到足够的连续真实车辆轨迹统计")
    manifest = hashlib.sha256(
        "".join(item["record_payload_sha256"] for item in source_audit).encode("ascii")
    ).hexdigest()
    return {
        "schema_version": 2,
        "dataset_label": dataset_label,
        "source_format": "Waymo Open Dataset Perception v1.2.0 laser_labels",
        "coordinate_frame": "first_ego_frame_from_global_pose",
        "agent_filter": "TYPE_VEHICLE with positive lidar points; ego is not a laser label",
        "dt_seconds": float(dt),
        "file_count": len(source_audit),
        "frame_count": accepted_frames,
        "vehicle_label_count": accepted_labels,
        "continuous_interval_count": continuous_intervals,
        "discontinuous_interval_count": discontinuous_intervals,
        "source_manifest_sha256": manifest,
        "sources": source_audit,
        "stats": stats,
        "ticks": ticks,
        "sample_counts": counts,
        "range_audit": range_audit,
        "definition": "SAFE-SIM fixed-bin histograms over real background vehicle tracks",
    }


def build_reference(paths, dt, dataset_label):
    """流式解析 TFRecord，并在单次读取中计算记录内容哈希。"""
    import tensorflow as tf
    from waymo_open_dataset import dataset_pb2, label_pb2

    def sources():
        for path in paths:
            digest = hashlib.sha256()

            def frames(current_path=path, current_digest=digest):
                # 逐记录解析，避免把约 1 GB 的相机与激光帧同时留在内存中。
                for raw in tf.data.TFRecordDataset(str(current_path), compression_type=""):
                    payload = bytes(raw.numpy())
                    current_digest.update(payload)
                    frame = dataset_pb2.Frame()
                    frame.ParseFromString(payload)
                    yield frame

            yield str(path), frames(), digest.hexdigest

    return build_reference_from_frames(
        sources(), dt, dataset_label, vehicle_type=label_pb2.Label.TYPE_VEHICLE
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Waymo Perception TFRecord 目录")
    parser.add_argument("output", type=Path, help="输出参考 JSON")
    parser.add_argument("--pattern", default="*.tfrecord")
    parser.add_argument("--max-files", type=int, default=0, help="0 表示使用全部匹配文件")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument(
        "--dataset-label", default="Waymo Open Dataset Perception v1.2.0 validation"
    )
    args = parser.parse_args()
    if args.dt <= 0 or not np.isfinite(args.dt):
        raise ValueError("dt 必须为正有限数")
    paths = sorted(args.input.glob(args.pattern))
    if args.max_files > 0:
        paths = paths[: args.max_files]
    if not paths:
        raise FileNotFoundError(f"未在 {args.input} 找到 {args.pattern}")
    result = build_reference(paths, args.dt, args.dataset_label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "file_count": result["file_count"],
                "frame_count": result["frame_count"],
                "sample_counts": result["sample_counts"],
                "source_manifest_sha256": result["source_manifest_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
