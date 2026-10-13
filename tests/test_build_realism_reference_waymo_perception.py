from types import SimpleNamespace

import numpy as np

from scripts.build_realism_reference_waymo_perception import build_reference_from_frames


def _frame(step, moving_x, ego_x, timestamp_offset=0):
    transform = np.eye(4)
    transform[0, 3] = ego_x
    box = SimpleNamespace(
        center_x=moving_x - ego_x,
        center_y=2.0,
        center_z=0.0,
        heading=0.0,
    )
    label = SimpleNamespace(
        id="vehicle-1",
        type=1,
        num_lidar_points_in_box=10,
        box=box,
    )
    return SimpleNamespace(
        timestamp_micros=timestamp_offset + step * 100_000,
        pose=SimpleNamespace(transform=transform.reshape(-1).tolist()),
        context=SimpleNamespace(name="segment-test"),
        laser_labels=[label],
    )


def _rotated_frame(step, moving_distance, ego_distance, initial_yaw):
    rotation = np.array(
        [
            [np.cos(initial_yaw), -np.sin(initial_yaw)],
            [np.sin(initial_yaw), np.cos(initial_yaw)],
        ]
    )
    transform = np.eye(4)
    transform[:2, :2] = rotation
    transform[:2, 3] = rotation @ np.array([ego_distance, 0.0])
    box = SimpleNamespace(
        center_x=moving_distance - ego_distance,
        center_y=2.0,
        center_z=0.0,
        heading=0.0,
    )
    label = SimpleNamespace(
        id="vehicle-rotated",
        type=1,
        num_lidar_points_in_box=10,
        box=box,
    )
    return SimpleNamespace(
        timestamp_micros=step * 100_000,
        pose=SimpleNamespace(transform=transform.reshape(-1).tolist()),
        context=SimpleNamespace(name="segment-rotated"),
        laser_labels=[label],
    )


def test_world_transform_removes_ego_motion_and_keeps_vehicle_motion():
    frames = [
        _frame(step, moving_x=0.5 * step + 0.02 * step ** 2, ego_x=0.2 * step)
        for step in range(30)
    ]
    result = build_reference_from_frames(
        [("synthetic.tfrecord", frames, "a" * 64)],
        0.1,
        "synthetic-waymo-perception",
    )

    assert result["file_count"] == 1
    assert result["frame_count"] == 30
    assert result["sample_counts"]["velocity"] == 29
    assert result["sample_counts"]["jerk"] == 27
    assert result["continuous_interval_count"] == 29
    assert result["discontinuous_interval_count"] == 0
    assert len(result["source_manifest_sha256"]) == 64


def test_timestamp_gap_breaks_finite_difference_chain():
    first = [_frame(step, moving_x=0.5 * step, ego_x=0.0) for step in range(12)]
    second = [
        _frame(step + 12, moving_x=6.0 + 0.5 * step, ego_x=0.0, timestamp_offset=200_000)
        for step in range(12)
    ]
    result = build_reference_from_frames(
        [("gap.tfrecord", first + second, "b" * 64)], 0.1, "gap-test"
    )

    assert result["discontinuous_interval_count"] == 1
    assert result["sample_counts"]["velocity"] == 22
    assert result["sample_counts"]["jerk"] == 18


def test_global_map_rotation_does_not_change_reference_histogram():
    straight = [_frame(i, moving_x=0.5 * i + 0.02 * i ** 2, ego_x=0.2 * i) for i in range(30)]
    rotated = [
        _rotated_frame(i, 0.5 * i + 0.02 * i ** 2, 0.2 * i, initial_yaw=2.2)
        for i in range(30)
    ]
    left = build_reference_from_frames(
        [("straight", straight, "c" * 64)], 0.1, "straight"
    )
    right = build_reference_from_frames(
        [("rotated", rotated, "d" * 64)], 0.1, "rotated"
    )

    assert left["stats"] == right["stats"]
    assert right["coordinate_frame"] == "first_ego_frame_from_global_pose"
