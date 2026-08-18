# adversarial_modules.py
import json
import re
import torch
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
# from diffusers import DDIMScheduler, UNet1DModel


class DiffusionTrajectoryRefiner:
    """使用插值方法将粗粒度锚点细化为平滑的物理可行轨迹"""
    def __init__(self, model_path=None, num_steps=5, device="cuda"):
        self.device = device
        self.num_steps = num_steps  # 大模型每次预测的帧数，扩散模型需要预测的未来步数

    @torch.no_grad()
    def refine_trajectory(self, initial_state, anchors):
        """
        initial_state: list/array [x, y, vx, vy, heading] 攻击车辆当前状态
        anchors: [[x1, y1], [x2, y2], ...] 大模型生成的稀疏锚点
        """
        # 使用插值方法生成平滑轨迹
        trajectory = self._interpolate_trajectory(initial_state, anchors)
        return trajectory

    def _interpolate_trajectory(self, initial_state, anchors):
        """
        使用三次样条插值生成平滑轨迹，并确保经过所有锚点。

        Args:
            initial_state: [x, y, vx, vy, heading] 起始状态
            anchors: [[x1, y1], [x2, y2], ...] 必须包含至少一个点（终点）

        Returns:
            trajectory: shape (self.num_steps, 5) 插值后的完整轨迹
        """
        dt = 0.1  # 时间步长（秒）
        num_steps = self.num_steps
        total_time = (num_steps - 1) * dt

        # 1. 构建锚点序列（包含起点）
        start_pos = np.array(initial_state[:2])
        anchor_points = np.array(anchors)  # shape (N, 2)
        # 如果 anchors 第一个点与起点重合，则直接使用，否则插入起点
        if len(anchor_points) == 0:
            # 无锚点时，仅保留起点和终点（终点使用起点？）这里根据实际情况处理
            # 但原函数要求至少2个锚点，我们假设至少有一个终点锚点
            raise ValueError("At least one anchor point is required.")

        # 检查起点是否已在锚点中（允许容差）
        if not np.allclose(anchor_points[0], start_pos, atol=1e-3):
            points = np.vstack([start_pos, anchor_points])
        else:
            points = anchor_points

        M = len(points)  # 总控制点数
        if M < 2:
            raise ValueError("Need at least two points for interpolation.")

        # 2. 分配时间节点（确保起点在 t=0，终点在 t=total_time）
        t_nodes = np.linspace(0, total_time, M)

        # 3. 三次样条插值（带边界条件：起点一阶导数 = 初始速度）
        #   对于点数 < 3，退化为线性插值
        if M >= 3:
            # 分别对 x(t), y(t) 进行插值，并指定左端导数边界
            vx0, vy0 = initial_state[2], initial_state[3]
            # 左端一阶导数给定，右端自然边界（二阶导为0）
            bc_x = ((1, vx0), (2, 0.0))
            bc_y = ((1, vy0), (2, 0.0))
            spline_x = CubicSpline(t_nodes, points[:, 0], bc_type=bc_x)
            spline_y = CubicSpline(t_nodes, points[:, 1], bc_type=bc_y)
        else:
            # 仅有两个点时使用线性插值
            spline_x = CubicSpline(t_nodes, points[:, 0], bc_type='natural')
            spline_y = CubicSpline(t_nodes, points[:, 1], bc_type='natural')

        # 4. 生成密集时间序列并计算位置
        t_dense = np.arange(num_steps) * dt
        interp_x = spline_x(t_dense)
        interp_y = spline_y(t_dense)

        # 5. 计算原始速度（中心差分）
        vx_raw = np.gradient(interp_x, dt)
        vy_raw = np.gradient(interp_y, dt)

        # 6. 速度平滑滤波（Savitzky-Golay）
        window = min(11, num_steps if num_steps % 2 == 1 else num_steps - 1)
        window = max(window, 3)
        if window >= 3:
            vx_smooth = savgol_filter(vx_raw, window_length=window, polyorder=2)
            vy_smooth = savgol_filter(vy_raw, window_length=window, polyorder=2)
        else:
            vx_smooth, vy_smooth = vx_raw, vy_raw

        # 强制起点速度与初始状态一致
        vx_smooth[0] = initial_state[2]
        vy_smooth[0] = initial_state[3]

        # 7. 计算航向角（基于平滑后的速度，并对小位移进行特殊处理）
        # 计算速度幅值
        speed = np.sqrt(vx_smooth**2 + vy_smooth**2)
        heading_raw = np.arctan2(vy_smooth, vx_smooth)

        # 处理小位移：若速度幅值小于阈值，沿用上一帧航向
        threshold = 0.01
        heading_smooth = heading_raw.copy()
        for i in range(1, num_steps):
            if speed[i] < threshold:
                heading_smooth[i] = heading_smooth[i-1]

        # 对航向进行角度展开，然后平滑（消除跳变）
        heading_unwrap = np.unwrap(heading_smooth)
        # 对展开后的角度进行平滑（使用移动平均或 Savitzky-Golay）
        if num_steps >= 5:
            window_h = min(5, num_steps if num_steps % 2 == 1 else num_steps - 1)
            if window_h >= 3:
                heading_unwrap_smooth = savgol_filter(heading_unwrap, window_length=window_h, polyorder=2)
            else:
                heading_unwrap_smooth = heading_unwrap
        else:
            heading_unwrap_smooth = heading_unwrap

        # 重新映射到 [-pi, pi]
        interp_heading = np.arctan2(np.sin(heading_unwrap_smooth), np.cos(heading_unwrap_smooth))

        # 强制起始航向与 initial_state 一致
        interp_heading[0] = initial_state[4]

        # 8. 构造完整轨迹
        trajectory = np.zeros((num_steps, 5))
        trajectory[:, 0] = interp_x
        trajectory[:, 1] = interp_y
        trajectory[:, 2] = vx_smooth
        trajectory[:, 3] = vy_smooth
        trajectory[:, 4] = interp_heading

        return trajectory

    # def _interpolate_trajectory(self, initial_state, anchors):
    #     """
    #     使用线性插值方法细化稀疏锚点,生成平滑轨迹,dt = 10
    #     initial_state: [x, y, vx, vy, heading]
    #     anchors: [[x1, y1], [x2, y2], ...]
    #     """
    #     # 确保锚点数量足够
    #     if len(anchors) < 2:
    #         raise ValueError("At least two anchors are required for interpolation.")

    #     # 提取锚点的 x, y, vx, vy, heading
    #     anchor_points = np.array(anchors)
    #     anchor_x = anchor_points[:, 0]
    #     anchor_y = anchor_points[:, 1]

    #     # 生成插值点的时间步
    #     num_anchors = len(anchors)
    #     anchor_timesteps = np.linspace(0, self.num_steps - 1, num=num_anchors)
    #     interp_timesteps = np.arange(self.num_steps)

    #     # 对 x, y, vx, vy, heading 分别进行线性插值
    #     interp_x = np.interp(interp_timesteps, anchor_timesteps, anchor_x)
    #     interp_y = np.interp(interp_timesteps, anchor_timesteps, anchor_y)

    #     # 初始化速度和航向角
    #     interp_vx = np.zeros(self.num_steps)
    #     interp_vy = np.zeros(self.num_steps)
    #     interp_heading = np.zeros(self.num_steps)

    #     # 使用初始状态的速度和航向角
    #     interp_vx[0] = initial_state[2]  # 初始 vx
    #     interp_vy[0] = initial_state[3]  # 初始 vy
    #     interp_heading[0] = initial_state[4]  # 初始航向角

    #     # 计算速度和航向角
    #     for i in range(1, self.num_steps):
    #         # 计算当前点与前一点的位移
    #         dx = interp_x[i] - interp_x[i - 1]
    #         dy = interp_y[i] - interp_y[i - 1]
    #         dt = 1  # 假设时间步长为 1

    #         # 计算速度
    #         interp_vx[i] = dx / dt
    #         interp_vy[i] = dy / dt

    #         # 计算航向角（使用 atan2 确保角度范围正确）
    #         interp_heading[i] = np.arctan2(dy, dx)


    #     # 构造完整轨迹
    #     trajectory = np.zeros((self.num_steps, 5))
    #     trajectory[:, 0] = interp_x
    #     trajectory[:, 1] = interp_y
    #     trajectory[:, 2] = interp_vx
    #     trajectory[:, 3] = interp_vy
    #     trajectory[:, 4] = interp_heading

    #     return trajectory

# class DiffusionTrajectoryRefiner:
#     """使用扩散模型将粗粒度锚点细化为平滑的物理可行轨迹"""
#     def __init__(self, model_path="path/to/diffusion_weights", device="cuda"):
#         self.device = device
#         # 假设使用1D UNet处理轨迹序列 [batch, seq_len, features]
#         self.model = UNet1DModel.from_pretrained(model_path).to(device)
#         self.scheduler = DDIMScheduler.from_pretrained(model_path)
#         self.scheduler.set_timesteps(10) # 使用DDIM加速采样
#         self.num_steps = 30 # 预测未来步数

#     @torch.no_grad()
#     def refine_trajectory(self, initial_state, anchors):
#         """
#         initial_state: list/array [x, y, vx, vy, heading] 攻击车辆当前状态
#         anchors: [[x1,y1], [x2,y2], ...] 大模型生成的稀疏锚点
#         """
#         condition = self._encode_condition(initial_state, anchors)
#         # 从纯高斯噪声开始采样 [batch, seq_len, feature_dim]
#         trajectory = torch.randn((1, self.num_steps, 5), device=self.device)
#         for t in self.scheduler.timesteps:
#             model_output = self.model(trajectory, t, condition).sample
#             trajectory = self.scheduler.step(model_output, t, trajectory).prev_sample
#         return trajectory.squeeze(0).cpu().numpy() # [num_steps, 5]

#     def _encode_condition(self, initial_state, anchors):
#         """将初始状态和锚点编码为条件张量 (需根据实际模型实现)"""
#         # 补全锚点数量
#         while len(anchors) < 3:
#             anchors.append(anchors[-1])  # 重复最后一个锚点
#         # 简化示例：拼接并扩展维度
#         cond = np.concatenate([initial_state, np.array(anchors).flatten()])
#         return torch.tensor(cond, dtype=torch.float32).unsqueeze(0).to(self.device)