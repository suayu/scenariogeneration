#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从自动驾驶数据集中提取静态道路几何信息和动态交通参与者的轨迹，
转换为 CARLA 可以加载的 OpenDRIVE 地图，并在同步模式下回放轨迹。
额外功能：
    - 无头模式启动 CARLA 服务器（不显示图形界面），适用于远程服务器。
    - 在自车（第一个生成的车辆）前方绑定 RGB 相机，逐帧保存图像。
"""

import os
import sys
import subprocess
import time
import signal
import numpy as np
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict, Optional

# 导入 CARLA Python API
try:
    import carla
except ImportError:
    print("错误：未找到 CARLA Python API，请将其添加到 PYTHONPATH。")
    sys.exit(1)


# ================================
# 1. 根据车道中心线生成 OpenDRIVE 文件
# ================================

def generate_opendrive_from_lanes(
    lane_centerlines: List[np.ndarray],
    lane_widths: List[float],
    lane_types: Optional[List[int]] = None,
    output_file: str = "output.xodr"
) -> str:
    """
    将车道中心线和宽度转换为简单的 OpenDRIVE 格式。
    每条车道作为一条独立的路段（road），每条路段只有一个车道（lane）。
    对于更复杂的多车道道路，建议后续进行后处理合并。

    参数：
        lane_centerlines: 列表，每个元素是一条车道的中心线点集 (N_i, 2)
        lane_widths: 列表，每条车道的宽度（米）
        lane_types: 可选，每条车道的类型编码（0:普通车道, 1:绿灯车道, 2:红灯车道）
        output_file: 输出的 .xodr 文件路径

    返回值：
        输出文件的路径
    """
    # 创建 OpenDRIVE 根元素
    root = ET.Element("OpenDRIVE")

    # 头部信息
    header = ET.SubElement(root, "header")
    header.set("revMajor", "1")
    header.set("revMinor", "4")
    header.set("name", "Extracted Map")
    header.set("version", "1.00")
    header.set("date", "2024-01-01T00:00:00")
    # 设置地图边界（足够大，覆盖所有道路）
    header.set("north", "100000")
    header.set("south", "-100000")
    header.set("east", "100000")
    header.set("west", "-100000")

    # 地理参考（占位符）
    geo_ref = ET.SubElement(header, "geoReference")
    geo_ref.text = "+proj=utm +zone=50 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

    # 遍历每条车道，生成对应的 road 元素
    road_id = 1
    for i, points in enumerate(lane_centerlines):
        if points.shape[0] < 2:   # 至少需要两个点才能构成一条道路
            continue

        # 根据 lane_types 确定车道类型字符串（OpenDRIVE 标准类型）
        lane_type_str = "driving"
        if lane_types is not None and lane_types[i] == 1:
            lane_type_str = "green"     # 自定义，但标准中无此类型，可保留为 driving
        elif lane_types is not None and lane_types[i] == 2:
            lane_type_str = "driving"   # 红灯车道仍是行驶车道

        road = ET.SubElement(root, "road")
        road.set("name", f"lane_{i}")
        road.set("id", str(road_id))
        road.set("junction", "-1")   # 不属于任何交叉口
        road_id += 1

        # 前后连接（无交叉口时为 -1，此处简单置空）
        link = ET.SubElement(road, "link")
        ET.SubElement(link, "predecessor")
        ET.SubElement(link, "successor")

        # 平面视图：将车道点转换为一系列的线段几何（line）
        planView = ET.SubElement(road, "planView")
        # 累积长度（s坐标），用于每个几何段的起始位置
        cumulative_s = 0.0
        for j in range(len(points)-1):
            p1 = points[j]
            p2 = points[j+1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            segment_length = np.hypot(dx, dy)
            if segment_length < 0.01:
                continue
            # 线段的朝向角（弧度）
            hdg = np.arctan2(dy, dx)
            geometry = ET.SubElement(planView, "geometry")
            geometry.set("s", str(cumulative_s))
            geometry.set("x", str(p1[0]))
            geometry.set("y", str(p1[1]))
            geometry.set("hdg", str(hdg))
            geometry.set("length", str(segment_length))
            # 线段类型为直线（line）
            line = ET.SubElement(geometry, "line")
            cumulative_s += segment_length

        # 车道段（lane section），从 s=0 开始，只有一个段
        laneSection = ET.SubElement(road, "laneSection")
        laneSection.set("s", "0.0")

        # 左侧车道（为空）
        left = ET.SubElement(laneSection, "left")
        # 右侧车道（我们放置实际车道）
        right = ET.SubElement(laneSection, "right")
        lane = ET.SubElement(right, "lane")
        lane.set("id", "1")
        lane.set("type", lane_type_str)
        lane.set("level", "false")

        # 车道宽度（恒定值，不随 s 变化）
        widthElem = ET.SubElement(lane, "width")
        widthElem.set("sOffset", "0.0")
        widthElem.set("a", str(lane_widths[i]))
        widthElem.set("b", "0.0")
        widthElem.set("c", "0.0")
        widthElem.set("d", "0.0")

        # 车道连接（无前后连接）
        laneLink = ET.SubElement(lane, "link")
        ET.SubElement(laneLink, "predecessor")
        ET.SubElement(laneLink, "successor")

        # 车道标线（白色实线）
        roadMark = ET.SubElement(lane, "roadMark")
        roadMark.set("sOffset", "0.0")
        roadMark.set("type", "solid")
        roadMark.set("weight", "standard")
        roadMark.set("color", "white")
        roadMark.set("width", "0.15")

    # 将 XML 写入文件
    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="UTF-8", xml_declaration=True)
    print(f"OpenDRIVE 文件已保存至 {output_file}")
    return output_file


# ================================
# 2. 从数据集场景中提取静态道路信息
# ================================

def extract_road_data_from_scene(
    lane_samples: np.ndarray,
    lane_types: Optional[np.ndarray] = None,
    default_lane_width: float = 3.5
) -> Tuple[List[np.ndarray], List[float], Optional[List[int]]]:
    """
    从原始车道点数据（与 plot_scene 中的格式相同）中提取车道中心线和宽度。

    参数：
        lane_samples: 形状 (N_lanes, num_points, 2/3) — 每条车道的离散点，第三维可能包含额外信息
        lane_types: 可选，形状 (N_lanes,) — 车道类型编码
        default_lane_width: 当数据中没有宽度信息时使用的默认宽度（米）

    返回值：
        centerlines: 列表，每个元素是一条车道的中心线点集 (num_points_i, 2)
        widths: 列表，每条车道的宽度（米）
        types: 列表，每条车道的类型（如果 lane_types 不为 None），否则为 None
    """
    centerlines = []
    widths = []
    types_list = None if lane_types is None else []

    for i, lane_points in enumerate(lane_samples):
        # 提取 x, y 坐标（忽略可能的第三维）
        if lane_points.shape[1] >= 2:
            points_xy = lane_points[:, :2]
        else:
            continue

        # 过滤无效点（NaN 或零值）
        valid_mask = ~np.isnan(points_xy).any(axis=1) & (np.linalg.norm(points_xy, axis=1) > 0)
        points_xy = points_xy[valid_mask]
        if points_xy.shape[0] < 2:
            continue

        centerlines.append(points_xy)
        widths.append(default_lane_width)   # 使用默认宽度
        if lane_types is not None:
            types_list.append(int(lane_types[i]))

    return centerlines, widths, types_list if lane_types is not None else None


# ================================
# 3. 提取交通参与者的轨迹
# ================================

def extract_agent_trajectories(
    agent_states_sequence: List[np.ndarray],
    agent_types_sequence: List[np.ndarray],
    agent_dim: Optional[Dict[str, int]] = None
) -> Tuple[Dict[int, List[np.ndarray]], Dict[int, Dict]]:
    """
    从按时间步组织的代理数据中，提取每个代理的轨迹和生成所需的初始信息。

    参数：
        agent_states_sequence: 列表，每个元素对应一个时间步，形状 (N_agents, state_dim)
            假设 state_dim 包含至少：[x, y, z?, cos_theta, sin_theta, length, width, active_flag?]
            默认索引：0:x, 1:y, 2:z, 3:cos_theta, 4:sin_theta, 5:length, 6:width
        agent_types_sequence: 列表，每个元素对应一个时间步，形状 (N_agents,) 或 one-hot 编码
        agent_dim: 可选，自定义状态字段索引映射

    返回值：
        trajectories: 字典 {agent_id: [(x, y, heading_rad), ...]}，每个代理的轨迹点（时间顺序）
        spawn_info: 字典 {agent_id: 包含 'type', 'length', 'width', 'initial_transform'} 用于生成
    """
    # 默认字段索引
    if agent_dim is None:
        agent_dim = {
            'x': 0, 'y': 1, 'z': 2,
            'cos_theta': 3, 'sin_theta': 4,
            'length': 5, 'width': 6
        }

    trajectories = {}
    spawn_info = {}
    num_agents = agent_states_sequence[0].shape[0]

    # 遍历每个时间步
    for t, states in enumerate(agent_states_sequence):
        # 处理代理类型（如果是 one-hot 则转为类别索引）
        types_t = agent_types_sequence[t]
        if types_t.ndim == 2:
            types_t = np.argmax(types_t, axis=1)

        for aid in range(num_agents):
            state = states[aid]
            # 如果状态数组长度大于7且最后一个字段为0，表示该代理无效（例如被过滤掉了）
            if state.shape[0] > 7 and state[-1] == 0:
                continue

            x = state[agent_dim['x']]
            y = state[agent_dim['y']]
            cos_theta = state[agent_dim['cos_theta']]
            sin_theta = state[agent_dim['sin_theta']]
            heading = np.arctan2(sin_theta, cos_theta)   # 弧度制朝向

            # 如果是第一次遇到该代理，保存初始信息
            if aid not in trajectories:
                trajectories[aid] = []
                length = float(state[agent_dim['length']])
                width = float(state[agent_dim['width']])
                agent_type = int(types_t[aid])
                # 根据类型映射 CARLA 蓝图名称
                if agent_type == 0:
                    actor_type = "vehicle"
                elif agent_type == 1:
                    actor_type = "walker.pedestrian.0001"
                else:
                    actor_type = "static.prop"
                spawn_info[aid] = {
                    'type': actor_type,
                    'length': length,
                    'width': width,
                    'initial_transform': carla.Transform(
                        carla.Location(x=x, y=y, z=0.5),          # 抬高 0.5 米避免陷入地面
                        carla.Rotation(yaw=np.degrees(heading))   # 朝向转度数
                    )
                }
            # 记录轨迹点
            trajectories[aid].append((x, y, heading))

    return trajectories, spawn_info


# ================================
# 4. CARLA 服务器管理（无头模式）
# ================================

class CarlaServerManager:
    """管理 CARLA 服务器进程，支持无头模式（不显示渲染窗口）。"""

    def __init__(self, carla_path: str, port: int = 2000, timeout_seconds: float = 10.0):
        """
        参数：
            carla_path: CARLA 服务器可执行文件的路径（例如 "./CarlaUE4.sh" 或 "CarlaUE4.exe"）
            port: CARLA 服务端口
            timeout_seconds: 等待服务器就绪的最大时间（秒）
        """
        self.carla_path = carla_path
        self.port = port
        self.timeout = timeout_seconds
        self.process = None

    def start(self) -> bool:
        """
        启动 CARLA 服务器，使用无头模式参数。
        返回值：启动是否成功（服务器成功绑定端口并响应）。
        """
        # 无头模式启动命令
        cmd = [
            self.carla_path,
            "-RenderOffScreen",      # 关闭渲染窗口
            "-carla-server",         # 以服务器模式运行
            f"-carla-port={self.port}",
            "-quality-level=Low"     # 降低画质，节省资源
        ]
        # 设置环境变量，强制无显示（避免 X11 依赖）
        env = os.environ.copy()
        env["DISPLAY"] = ""

        try:
            # 启动子进程
            # 在 Linux 下使用 setsid 创建新会话，便于后续终止整个进程组
            if os.name != 'nt':
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=env,
                    preexec_fn=os.setsid
                )
            else:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=env
                )
        except Exception as e:
            print(f"启动 CARLA 服务器失败: {e}")
            return False

        # 等待服务器就绪：尝试建立连接
        start_time = time.time()
        client = carla.Client("localhost", self.port)
        client.set_timeout(2.0)   # 短超时以便快速重试
        while time.time() - start_time < self.timeout:
            try:
                client.get_server_version()   # 如果成功，说明服务器已启动
                print(f"CARLA 服务器已就绪，端口 {self.port}")
                return True
            except Exception:
                time.sleep(0.5)
        print(f"CARLA 服务器在 {self.timeout} 秒内未就绪")
        return False

    def stop(self):
        """终止 CARLA 服务器进程。"""
        if self.process:
            if os.name == 'nt':
                self.process.terminate()
            else:
                # 终止整个进程组
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.process.wait()
            print("CARLA 服务器已停止。")


# ================================
# 5. CARLA 同步回放 + 相机录制
# ================================

def carla_replay_trajectories(
    carla_client: carla.Client,
    opendrive_file: str,
    trajectories: Dict[int, List[np.ndarray]],
    spawn_info: Dict[int, Dict],
    fps: int = 10,
    timeout: float = 2.0,
    camera_save_dir: Optional[str] = "carla_frames",
    camera_attach_to_ego: bool = True,
    camera_image_size: Tuple[int, int] = (800, 600)
) -> None:
    """
    加载 OpenDRIVE 地图，生成代理，并在同步模式下按时间步更新位置。
    如果提供 camera_save_dir，则会在自车（第一个生成的车辆）上附加 RGB 相机，并逐帧保存图像。

    参数：
        camera_save_dir: 保存图像帧的目录，不存在则会自动创建。
        camera_attach_to_ego: 是否将相机附加到自车（第一个生成的代理）上。
        camera_image_size: 相机图像的尺寸 (宽度, 高度)。
    """
    # 读取 OpenDRIVE 内容
    with open(opendrive_file, 'r') as f:
        opendrive_content = f.read()

    carla_client.set_timeout(timeout)
    world = carla_client.load_opendrive(opendrive_content)

    # 设置同步模式
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / fps   # 每帧时间间隔
    world.apply_settings(settings)

    # 生成所有代理
    blueprints = world.get_blueprint_library()
    actors = []        # 存储 (agent_id, actor) 对
    ego_actor = None   # 第一个生成的车辆作为自车

    for aid, info in spawn_info.items():
        # 选择蓝图
        if info['type'] == 'vehicle':
            bp = blueprints.filter('vehicle.*')[0]   # 使用第一个可用的车辆蓝图
        elif info['type'] == 'walker.pedestrian.0001':
            bp = blueprints.filter('walker.pedestrian.0001')[0]
        else:
            continue

        transform = info['initial_transform']
        actor = world.spawn_actor(bp, transform)
        if actor:
            actors.append((aid, actor))
            if ego_actor is None:
                ego_actor = actor
            print(f"生成代理 {aid} ({info['type']}) 于 {transform.location}")
        else:
            print(f"生成代理 {aid} 失败，可能位置重叠或超出地图边界")

    if not actors:
        print("没有成功生成任何代理，退出回放。")
        return

    # ---------- 相机设置 ----------
    camera = None
    if camera_save_dir and ego_actor is not None and camera_attach_to_ego:
        os.makedirs(camera_save_dir, exist_ok=True)
        # 获取 RGB 相机蓝图
        camera_bp = blueprints.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(camera_image_size[0]))
        camera_bp.set_attribute('image_size_y', str(camera_image_size[1]))
        camera_bp.set_attribute('fov', '90')          # 视野90度
        # 相机安装位置：自车前保险杠上方，向前看
        spawn_transform = carla.Transform(
            carla.Location(x=2.0, y=0.0, z=1.5),      # 车前2米，高1.5米
            carla.Rotation(pitch=-5.0, yaw=0.0, roll=0.0)  # 略微俯视
        )
        camera = world.spawn_actor(camera_bp, spawn_transform, attach_to=ego_actor)
        print(f"RGB 相机已附加到自车，图像将保存至 {camera_save_dir}")

    # 帧计数器，用于图像命名
    frame_counter = 0

    def save_camera_image(image):
        """相机回调函数：将收到的图像保存为 PNG 文件"""
        nonlocal frame_counter
        filename = os.path.join(camera_save_dir, f"frame_{frame_counter:04d}.png")
        image.save_to_disk(filename)
        # 若需要显示进度，可取消下面行的注释
        # print(f"已保存 {filename}")

    if camera:
        camera.listen(lambda img: save_camera_image(img))

    # ---------- 回放主循环 ----------
    # 获取轨迹的帧数（所有代理应具有相同长度）
    num_frames = len(next(iter(trajectories.values())))
    spectator = world.get_spectator()   # 用于控制观察视角

    try:
        for frame in range(num_frames):
            frame_counter = frame   # 更新帧计数器，用于相机文件名

            # 更新每个代理的位置和朝向
            for aid, actor in actors:
                if frame < len(trajectories[aid]):
                    x, y, heading_rad = trajectories[aid][frame]
                    yaw = np.degrees(heading_rad)
                    location = carla.Location(x=x, y=y, z=0.5)
                    rotation = carla.Rotation(yaw=yaw, pitch=0, roll=0)
                    actor.set_transform(carla.Transform(location, rotation))

            # 可选：调整观察者视角，跟随自车（便于观察）
            if ego_actor:
                ego_transform = ego_actor.get_transform()
                spectator.set_transform(carla.Transform(
                    ego_transform.location + carla.Location(z=15),   # 俯视视角，高度15米
                    carla.Rotation(pitch=-30, yaw=ego_transform.rotation.yaw)
                ))

            # 触发仿真 tick，推进物理、传感器等（相机回调会在此时执行）
            world.tick()

    finally:
        # ---------- 清理资源 ----------
        if camera:
            camera.stop()      # 停止相机监听
            camera.destroy()   # 销毁相机
        for _, actor in actors:
            actor.destroy()    # 销毁所有代理
        # 恢复为异步模式，方便后续可能的操作
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print(f"回放结束。共 {num_frames} 帧，相机图像已保存至 {camera_save_dir}。")


# ================================
# 6. 完整流程：启动服务器 → 生成地图 → 回放 → 关闭服务器
# ================================

def run_carla_replay(
    lane_samples: np.ndarray,
    lane_types: Optional[np.ndarray],
    agent_states_sequence: List[np.ndarray],
    agent_types_sequence: List[np.ndarray],
    output_xodr: str = "extracted_map.xodr",
    carla_server_path: str = "./CarlaUE4.sh",
    carla_port: int = 2000,
    fps: int = 10,
    camera_save_dir: str = "carla_frames"
):
    """
    完整流程：启动无头 CARLA 服务器 → 提取道路数据 → 生成 OpenDRIVE 地图 → 回放轨迹 → 录制相机图像 → 关闭服务器。

    参数：
        lane_samples: 车道点数据，形状 (N_lanes, num_points, 2/3)
        lane_types: 可选，车道类型
        agent_states_sequence: 按时间步排列的代理状态列表
        agent_types_sequence: 按时间步排列的代理类型列表
        output_xodr: 生成的 OpenDRIVE 文件保存路径
        carla_server_path: CARLA 服务器可执行文件路径
        carla_port: CARLA 服务端口
        fps: 仿真帧率（必须与轨迹采样率一致）
        camera_save_dir: 相机图像保存目录
    """
    # 1. 提取道路几何信息
    centerlines, widths, types = extract_road_data_from_scene(
        lane_samples, lane_types, default_lane_width=3.5
    )

    # 2. 生成 OpenDRIVE 地图文件
    opendrive_path = generate_opendrive_from_lanes(
        centerlines, widths, types, output_xodr
    )

    # 3. 提取代理轨迹和生成信息
    trajectories, spawn_info = extract_agent_trajectories(
        agent_states_sequence, agent_types_sequence
    )

    # 4. 启动 CARLA 服务器（无头模式）
    server_manager = CarlaServerManager(carla_server_path, port=carla_port)
    if not server_manager.start():
        print("启动 CARLA 服务器失败，程序退出。")
        return

    # 等待服务器完全初始化（可选，增加稳定性）
    time.sleep(2)

    # 5. 连接客户端并执行回放
    client = carla.Client("localhost", carla_port)
    client.set_timeout(10.0)
    try:
        carla_replay_trajectories(
            client, opendrive_path, trajectories, spawn_info,
            fps=fps, camera_save_dir=camera_save_dir
        )
    except Exception as e:
        print(f"回放过程中发生错误: {e}")
    finally:
        # 6. 关闭 CARLA 服务器
        server_manager.stop()


# ================================
# 示例用法（包含模拟数据）
# ================================

if __name__ == "__main__":
    # 创建模拟的车道数据：3条直线车道，间距3.5米
    dummy_lanes = []
    for i in range(3):
        x = np.linspace(-30, 30, 20)               # 从-30到30的20个点
        y = np.ones_like(x) * (i * 3.5 - 3.5)      # 车道横向偏移
        points = np.stack([x, y], axis=-1)
        dummy_lanes.append(points)
    lane_samples = np.array(dummy_lanes)   # (3, 20, 2)
    lane_types = np.array([0, 0, 0])       # 所有车道都是普通车道

    # 模拟代理轨迹：10个时间步，2个代理（车辆和行人）
    T = 10
    num_agents = 2
    agent_states_seq = []
    agent_types_seq = []
    for t in range(T):
        states = np.zeros((num_agents, 8))   # 最后一列为 active 标志
        # 代理0：车辆，沿 x 轴正向移动
        states[0, 0] = 0.0 + t * 0.5
        states[0, 1] = 0.0
        states[0, 3] = np.cos(0.0)
        states[0, 4] = np.sin(0.0)
        states[0, 5] = 4.5    # 长度
        states[0, 6] = 1.8    # 宽度
        states[0, 7] = 1      # 有效
        # 代理1：行人，静止在某点
        states[1, 0] = -10.0
        states[1, 1] = 3.0
        states[1, 3] = 1.0
        states[1, 4] = 0.0
        states[1, 5] = 0.5
        states[1, 6] = 0.5
        states[1, 7] = 1
        agent_states_seq.append(states)
        agent_types_seq.append(np.array([0, 1]))   # 0:车辆, 1:行人

    # 运行完整流程（请根据实际 CARLA 安装路径修改 carla_server_path）
    run_carla_replay(
        lane_samples, lane_types,
        agent_states_seq, agent_types_seq,
        carla_server_path="./CarlaUE4.sh",   # 请替换为你的 CARLA 可执行文件路径
        carla_port=2000,
        fps=5,                               # 低帧率用于演示
        camera_save_dir="output_frames"
    )