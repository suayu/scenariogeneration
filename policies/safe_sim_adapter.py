"""不依赖 trajdata 仿真状态的 Scenario Dreamer 到 Safe-Sim 数据转换。"""

import math

import cv2
import numpy as np
import torch

from policies.traffic_types import JointTrajectory, SafeSimBatch, ScenarioFrame


def _wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _transform_points(points, transform):
    points = np.asarray(points, dtype=np.float32)
    homogeneous = np.concatenate([points, np.ones((*points.shape[:-1], 1), dtype=np.float32)], axis=-1)
    return np.einsum("...ij,...j->...i", transform, homogeneous)[..., :2]


class SafeSimBatchAdapter:
    """构建 Safe-Sim 扩散网络直接使用的张量。

    原始权重使用 NuScenes/trajdata 语义地图训练。Scenario Dreamer 不包含对应地图
    对象，因此适配器使用真实车道折线和路线构建确定性的三层语义栅格。动态通道
    和全部九维历史特征均来自仿真器的真实闭环历史。
    """

    RASTER_SOURCE = "scenario_dreamer_lane_raster_v1"

    def __init__(
        self,
        history_frames=11,
        max_neighbors=20,
        raster_size=224,
        pixel_size=0.5,
        raster_center=(0.25, 0.5),
    ):
        if history_frames < 2:
            raise ValueError("history_frames must include at least one past frame and the current frame")
        if max_neighbors < 1:
            raise ValueError("max_neighbors must be positive")
        self.history_frames = int(history_frames)
        self.max_neighbors = int(max_neighbors)
        self.raster_size = int(raster_size)
        self.pixel_size = float(pixel_size)
        self.pixels_per_meter = 1.0 / self.pixel_size
        self.raster_center = tuple(float(v) for v in raster_center)

    @staticmethod
    def _agent_transforms(states):
        batch_size = states.shape[0]
        agent_from_world = np.zeros((batch_size, 3, 3), dtype=np.float32)
        world_from_agent = np.zeros((batch_size, 3, 3), dtype=np.float32)
        for row, state in enumerate(states):
            x, y, yaw = float(state[0]), float(state[1]), float(state[4])
            c, s = math.cos(yaw), math.sin(yaw)
            world_from_agent[row] = np.array(
                [[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]], dtype=np.float32
            )
            agent_from_world[row] = np.array(
                [[c, s, -(c * x + s * y)], [-s, c, s * x - c * y], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
        return agent_from_world, world_from_agent

    def _history_features(self, history, mask, transform, focal_yaw, dt):
        features = np.zeros((self.history_frames, 9), dtype=np.float32)
        valid_indices = np.flatnonzero(mask)
        if len(valid_indices) == 0:
            return features

        local_positions = _transform_points(history[:, :2], transform)
        rotation = transform[:2, :2]
        local_velocities = history[:, 2:4] @ rotation.T
        local_accelerations = np.zeros_like(local_velocities)
        for idx in valid_indices[1:]:
            if mask[idx - 1]:
                local_accelerations[idx] = (local_velocities[idx] - local_velocities[idx - 1]) / dt
        local_yaws = _wrap_angle(history[:, 4] - focal_yaw)

        features[:, 0:2] = local_positions
        features[:, 2] = 0.0
        features[:, 3:5] = local_velocities
        features[:, 5:7] = local_accelerations
        features[:, 7] = np.sin(local_yaws)
        features[:, 8] = np.cos(local_yaws)
        features[~mask] = 0.0
        return features

    def _to_pixels(self, local_points):
        pixels = np.empty_like(local_points, dtype=np.float32)
        pixels[..., 0] = local_points[..., 0] * self.pixels_per_meter + self.raster_center[0] * self.raster_size
        pixels[..., 1] = local_points[..., 1] * self.pixels_per_meter + self.raster_center[1] * self.raster_size
        return np.rint(pixels).astype(np.int32)

    def _draw_polyline(self, image, points, value=1.0, thickness=1):
        if len(points) < 2:
            return
        pixels = self._to_pixels(points)
        finite = np.isfinite(pixels).all(axis=-1)
        pixels = pixels[finite]
        if len(pixels) >= 2:
            cv2.polylines(image, [pixels.reshape(-1, 1, 2)], False, float(value), thickness, cv2.LINE_AA)

    def _draw_static_obstacle(self, image, obstacle, transform, value=-1.0):
        """将全局静态障碍物绘制到既有动态参与者通道，不增加模型输入通道。"""
        center_local = _transform_points(np.asarray(obstacle[:2]), transform)
        heading_local = transform[:2, :2] @ np.array(
            [math.cos(float(obstacle[2])), math.sin(float(obstacle[2]))],
            dtype=np.float32,
        )
        yaw_local = math.atan2(float(heading_local[1]), float(heading_local[0]))
        half_length = float(obstacle[3]) * 0.5
        half_width = float(obstacle[4]) * 0.5
        corners = np.array(
            [
                [-half_length, -half_width],
                [half_length, -half_width],
                [half_length, half_width],
                [-half_length, half_width],
            ],
            dtype=np.float32,
        )
        rotation = np.array(
            [[math.cos(yaw_local), -math.sin(yaw_local)],
             [math.sin(yaw_local), math.cos(yaw_local)]],
            dtype=np.float32,
        )
        pixels = self._to_pixels(center_local + corners @ rotation.T)
        cv2.fillConvexPoly(image, pixels.reshape(-1, 1, 2), float(value))

    def _nearest_centerline(self, frame, agent_state, transform):
        lanes = frame.lanes_global
        if lanes.size == 0:
            return np.zeros((1, 3), dtype=np.float32), False, 0.0
        distances = np.linalg.norm(lanes[..., :2] - agent_state[None, None, :2], axis=-1)
        lane_index = int(np.unravel_index(np.nanargmin(distances), distances.shape)[0])
        lane_local = _transform_points(lanes[lane_index, :, :2], transform)
        centerline = np.concatenate(
            [lane_local, np.zeros((lane_local.shape[0], 1), dtype=np.float32)], axis=-1
        )
        if len(lane_local) > 1:
            delta = lane_local[1] - lane_local[0]
            initial_heading = float(math.atan2(delta[1], delta[0]))
        else:
            initial_heading = 0.0
        has_lane = bool(np.nanmin(distances[lane_index]) <= 5.0)
        return centerline.astype(np.float32), has_lane, initial_heading

    def _rasterize(self, frame, controlled_ids, transforms, context_agent_ids=None):
        batch_size = len(controlled_ids)
        if context_agent_ids is None:
            context_agent_ids = controlled_ids
        image = np.zeros(
            (batch_size, self.history_frames + 3, self.raster_size, self.raster_size),
            dtype=np.float32,
        )
        centerlines = []
        has_lanes = []
        initial_headings = []

        for row, agent_id in enumerate(controlled_ids):
            transform = transforms[row]
            focal_state = frame.states_global[agent_id]
            for lane in frame.lanes_global:
                self._draw_polyline(image[row, self.history_frames], _transform_points(lane, transform), 1.0, 1)
                self._draw_polyline(image[row, self.history_frames + 2], _transform_points(lane, transform), 1.0, 7)
            self._draw_polyline(
                image[row, self.history_frames + 1],
                _transform_points(frame.route_global, transform),
                1.0,
                2,
            )

            for hist_idx in range(self.history_frames):
                for other_id in context_agent_ids:
                    if not frame.history_mask[other_id, hist_idx]:
                        continue
                    local_point = _transform_points(frame.history_global[other_id, hist_idx, :2], transform)
                    pixel = self._to_pixels(local_point)
                    if 0 <= pixel[0] < self.raster_size and 0 <= pixel[1] < self.raster_size:
                        value = 1.0 if other_id == agent_id else -1.0
                        cv2.circle(image[row, hist_idx], tuple(pixel), 1, value, -1)

            # 障碍物在每个历史切片保持占用，复用“其他参与者”为负值的语义。
            for hist_idx in range(self.history_frames):
                for obstacle in frame.static_obstacles_global:
                    self._draw_static_obstacle(image[row, hist_idx], obstacle, transform)

            centerline, has_lane, initial_heading = self._nearest_centerline(frame, focal_state, transform)
            centerlines.append(centerline)
            has_lanes.append(has_lane)
            initial_headings.append(initial_heading)

        max_points = max((len(line) for line in centerlines), default=1)
        padded_centerlines = np.full((batch_size, max_points, 3), np.nan, dtype=np.float32)
        for row, centerline in enumerate(centerlines):
            padded_centerlines[row, : len(centerline)] = centerline
        return image, padded_centerlines, np.asarray(has_lanes), np.asarray(initial_headings, dtype=np.float32)

    def build(self, frame, controlled_agent_ids=None):
        if not isinstance(frame, ScenarioFrame):
            raise TypeError("frame must be a ScenarioFrame")
        if frame.history_global.shape[1] != self.history_frames:
            raise ValueError(
                f"frame history has {frame.history_global.shape[1]} frames; expected {self.history_frames}"
            )

        active_ids = frame.agent_ids[frame.active_mask].astype(np.int64, copy=False)
        if controlled_agent_ids is None:
            controlled_ids = active_ids
        else:
            controlled_ids = np.asarray(controlled_agent_ids, dtype=np.int64)
            if len(np.unique(controlled_ids)) != len(controlled_ids):
                raise ValueError('controlled_agent_ids must be unique')
            if not set(controlled_ids.tolist()).issubset(set(active_ids.tolist())):
                raise ValueError('controlled_agent_ids must be active participants')
        if len(controlled_ids) == 0:
            return SafeSimBatch(
                data={},
                row_to_agent_id=controlled_ids,
                agent_from_world=np.empty((0, 3, 3), dtype=np.float32),
                world_from_agent=np.empty((0, 3, 3), dtype=np.float32),
                raster_source=self.RASTER_SOURCE,
            )

        controlled_states = frame.states_global[controlled_ids]
        agent_from_world, world_from_agent = self._agent_transforms(controlled_states)
        batch_size = len(controlled_ids)
        agent_hist = np.zeros((batch_size, self.history_frames, 9), dtype=np.float32)
        neigh_hist = np.zeros(
            (batch_size, self.max_neighbors, self.history_frames, 9), dtype=np.float32
        )

        for row, agent_id in enumerate(controlled_ids):
            focal_yaw = float(frame.states_global[agent_id, 4])
            agent_hist[row] = self._history_features(
                frame.history_global[agent_id],
                frame.history_mask[agent_id],
                agent_from_world[row],
                focal_yaw,
                frame.dt,
            )
            other_ids = active_ids[active_ids != agent_id]
            if len(other_ids):
                distances = np.linalg.norm(
                    frame.states_global[other_ids, :2] - frame.states_global[agent_id, :2], axis=-1
                )
                other_ids = other_ids[np.argsort(distances)[: self.max_neighbors]]
            for neighbor_row, other_id in enumerate(other_ids):
                neigh_hist[row, neighbor_row] = self._history_features(
                    frame.history_global[other_id],
                    frame.history_mask[other_id],
                    agent_from_world[row],
                    focal_yaw,
                    frame.dt,
                )

        image, centerline, has_lane, initial_heading = self._rasterize(
            frame, controlled_ids, agent_from_world, context_agent_ids=active_ids
        )
        speeds = np.linalg.norm(controlled_states[:, 2:4], axis=-1).astype(np.float32)
        current_accelerations_world = np.zeros((batch_size, 2), dtype=np.float32)
        for row, agent_id in enumerate(controlled_ids):
            if frame.history_mask[agent_id, -2:].all():
                current_accelerations_world[row] = (
                    frame.history_global[agent_id, -1, 2:4]
                    - frame.history_global[agent_id, -2, 2:4]
                ) / frame.dt
        raster_from_agent = np.repeat(
            np.array(
                [
                    [self.pixels_per_meter, 0.0, self.raster_center[0] * self.raster_size],
                    [0.0, self.pixels_per_meter, self.raster_center[1] * self.raster_size],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )[None],
            batch_size,
            axis=0,
        )

        # Safe-Sim 静止目标过滤器即使处于 "on_lane" 模式也要求带形状的参考轨迹；
        # 该模式不会读取未来值，因此用当前实测状态构造一帧显式占位数据。
        current_reference = np.zeros((batch_size, 1, 8), dtype=np.float32)
        current_reference[:, 0, :2] = controlled_states[:, :2]
        current_reference[:, 0, 2] = speeds

        static_obstacles = np.asarray(frame.static_obstacles_global, dtype=np.float32)
        static_obstacle_mask = np.ones(static_obstacles.shape[0], dtype=bool)

        data = {
            "image": torch.from_numpy(image),
            "agent_hist": torch.from_numpy(agent_hist),
            "neigh_hist": torch.from_numpy(neigh_hist),
            "curr_speed": torch.from_numpy(speeds),
            # 传入上一真实帧加速度，使闭环重规划边界也能计算 jerk。
            "scenario_curr_acceleration_world": torch.from_numpy(
                current_accelerations_world
            ),
            "dt": torch.full((batch_size,), float(frame.dt), dtype=torch.float32),
            "centroid": torch.from_numpy(controlled_states[:, :2].astype(np.float32)),
            "yaw": torch.from_numpy(controlled_states[:, 4].astype(np.float32)),
            "extent": torch.from_numpy(controlled_states[:, 5:7].astype(np.float32)),
            "type": torch.from_numpy(np.argmax(frame.agent_types[controlled_ids], axis=-1).astype(np.int64)),
            "agent_fut_extent": torch.from_numpy(
                controlled_states[:, None, 5:7].astype(np.float32)
            ),
            "all_other_agents_types": torch.from_numpy(
                np.argmax(frame.agent_types[controlled_ids], axis=-1).astype(np.int64)
            ),
            "raster_from_agent": torch.from_numpy(raster_from_agent),
            "world_from_agent": torch.from_numpy(world_from_agent),
            # 每个受控车辆共享同一组全局障碍物；mask 允许空障碍物批次。
            "static_obstacles_world": torch.from_numpy(
                np.repeat(static_obstacles[None], batch_size, axis=0)
            ),
            "static_obstacle_mask": torch.from_numpy(
                np.repeat(static_obstacle_mask[None], batch_size, axis=0)
            ),
            # 所有预测行均属于当前同一个仿真场景，供多车联合损失建立配对关系。
            "scene_index": torch.zeros((batch_size,), dtype=torch.int64),
            "extras": {
                "centerline_xy": torch.from_numpy(centerline),
                "has_lane": torch.from_numpy(has_lane),
                "init_centerline_heading": torch.from_numpy(initial_heading),
                "full_fut_traj": torch.from_numpy(current_reference),
                "full_fut_valid": torch.ones((batch_size, 1), dtype=torch.float32),
            },
        }
        if frame.ego_state_global is not None:
            ego_state = np.asarray(frame.ego_state_global, dtype=np.float32)
            data["scenario_ego_state"] = torch.from_numpy(
                np.repeat(ego_state[None], batch_size, axis=0)
            )
        return SafeSimBatch(
            data=data,
            row_to_agent_id=controlled_ids,
            agent_from_world=agent_from_world,
            world_from_agent=world_from_agent,
            raster_source=self.RASTER_SOURCE,
        )

    def build_anchor_guidance(
        self,
        frame,
        safe_batch,
        attack_intent,
        prediction_horizon,
        anchor_interval_seconds,
    ):
        """将全局 LLM 锚点插值为攻击车局部坐标系下的逐帧引导目标。"""
        batch_size = len(safe_batch.row_to_agent_id)
        positions_local = np.zeros((batch_size, prediction_horizon, 2), dtype=np.float32)
        valid_mask = np.zeros((batch_size, prediction_horizon), dtype=bool)
        metadata = {
            "active": False,
            "target_id": None,
            "valid_steps": 0,
        }
        tensors = {
            "llm_anchor_positions_local": torch.from_numpy(positions_local),
            "llm_anchor_mask": torch.from_numpy(valid_mask),
        }
        if attack_intent is None:
            return tensors, metadata
        if anchor_interval_seconds <= 0:
            raise ValueError("anchor_interval_seconds must be positive")

        target_id = int(attack_intent["target_id"])
        target_rows = np.flatnonzero(safe_batch.row_to_agent_id == target_id)
        if len(target_rows) == 0:
            return tensors, metadata

        anchors_global = np.asarray(attack_intent["anchors"], dtype=np.float32)
        if anchors_global.ndim != 2 or anchors_global.shape[1] != 2 or len(anchors_global) < 2:
            raise ValueError("LLM anchors must have shape [K, 2] with at least two points")
        if not np.isfinite(anchors_global).all():
            raise ValueError("LLM anchors must contain only finite coordinates")

        source_step = int(attack_intent["source_step"])
        if source_step > frame.step:
            raise ValueError("attack intent source_step cannot be later than the current frame")
        elapsed_seconds = (frame.step - source_step) * frame.dt
        prediction_times = elapsed_seconds + (
            np.arange(prediction_horizon, dtype=np.float32) + 1.0
        ) * frame.dt
        anchor_times = np.arange(len(anchors_global), dtype=np.float32) * anchor_interval_seconds
        active_steps = prediction_times <= anchor_times[-1] + 1e-6
        if not active_steps.any():
            return tensors, metadata

        interpolated_global = np.column_stack(
            [
                np.interp(prediction_times[active_steps], anchor_times, anchors_global[:, axis])
                for axis in range(2)
            ]
        ).astype(np.float32)
        target_row = int(target_rows[0])
        positions_local[target_row, active_steps] = _transform_points(
            interpolated_global,
            safe_batch.agent_from_world[target_row],
        )
        valid_mask[target_row, active_steps] = True
        metadata.update(
            {
                "active": True,
                "target_id": target_id,
                "valid_steps": int(active_steps.sum()),
                "elapsed_seconds": float(elapsed_seconds),
            }
        )
        return tensors, metadata

    def decode(self, frame, safe_batch, positions_local, yaws_local):
        positions_local = np.asarray(positions_local, dtype=np.float32)
        yaws_local = np.asarray(yaws_local, dtype=np.float32)
        if positions_local.ndim != 3 or positions_local.shape[-1] != 2:
            raise ValueError(f"positions_local must be [A, T, 2], got {positions_local.shape}")
        if yaws_local.ndim == 2:
            yaws_local = yaws_local[..., None]
        if yaws_local.shape[:2] != positions_local.shape[:2] or yaws_local.shape[-1] != 1:
            raise ValueError(f"yaws_local must be [A, T, 1], got {yaws_local.shape}")
        if positions_local.shape[0] != len(safe_batch.row_to_agent_id):
            raise ValueError("model output batch does not match SafeSimBatch ID mapping")

        batch_size, horizon = positions_local.shape[:2]
        positions_global = np.empty_like(positions_local)
        yaws_global = np.empty_like(yaws_local)
        velocities_global = np.empty_like(positions_local)
        for row, agent_id in enumerate(safe_batch.row_to_agent_id):
            positions_global[row] = _transform_points(
                positions_local[row], safe_batch.world_from_agent[row]
            )
            focal_yaw = float(frame.states_global[agent_id, 4])
            yaws_global[row, :, 0] = _wrap_angle(yaws_local[row, :, 0] + focal_yaw)
            previous = frame.states_global[agent_id, :2].astype(np.float32)
            for step in range(horizon):
                velocities_global[row, step] = (positions_global[row, step] - previous) / frame.dt
                previous = positions_global[row, step]

        return JointTrajectory(
            source_step=frame.step,
            agent_ids=safe_batch.row_to_agent_id.copy(),
            positions_global=positions_global,
            yaws_global=yaws_global,
            velocities_global=velocities_global,
            valid_mask=np.ones((batch_size, horizon), dtype=bool),
            metadata={"raster_source": safe_batch.raster_source},
        )
