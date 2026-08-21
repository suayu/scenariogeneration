import os
import json
import time
import subprocess
import numpy as np
import carla
from typing import Dict, List
import xml.etree.ElementTree as ET
from contextlib import closing
import socket

def yaw_to_carla_yaw(yaw: float) -> float:
    """
    将仿真数据中的航向角转换为 CARLA 的航向角。
    仿真数据中的航向角是以弧度表示的，# 且 0 弧度指向正 x 轴，顺时针为正方向。
    CARLA 中的航向角是以度数表示的，# 且 0 度指向正 x 轴，逆时针为正方向。
    """
    carla_yaw = -np.degrees(yaw)  # 转换为度数并取负号
    return carla_yaw

def is_port_open(port, host="localhost", timeout=1):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0

def start_carla_simulator(carla_path: str, port: int = 2000, quality="Low",offscreen=True):
    """
    以無頭模式啟動 CARLA 模擬器。

    參數:
        carla_path: CARLA 模擬器的啟動腳本路徑，例如 "/path/to/CarlaUE4.sh"
        port: 模擬器運行的端口，默認為 2000
    """
    print("正在啟動 CARLA 模擬器（無頭模式）...")
    cmd = [
        carla_path,
        f"-carla-rpc-port={port}",
        f"-quality-level={quality}",
        "-nosound"
    ]
    if offscreen:
        cmd.append("-RenderOffScreen")

    # 將輸出重定向到日誌文件，避免 PIPE 阻塞導致死鎖
    log_file = open("carla_simulator.log", "w")
    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid  # 允许 killpg
    )

    print("[INFO] Waiting for CARLA simulator to start...")
    # 最多等待 120 秒
    for _ in range(120):
        if is_port_open(port):
            print(f"✅ CARLA is ready on port {port}")
            return process
        time.sleep(1)

    print("❌ CARLA simulator failed to start within timeout.")
    return process

def load_simulation_data(initial_data_path: str, step_data_path: str):
    """
    加载仿真数据，包括初始数据和时间步数据。

    参数:
        initial_data_path: 初始数据文件路径 (initial_data.json)
        step_data_path: 时间步数据文件路径 (step_data.json)

    返回:
        initial_data: 包含自车和其他交通参与者初始状态的字典
        step_data: 包含每一帧状态的字典
    """
    with open(initial_data_path, 'r') as f:
        initial_data = json.load(f)
    with open(step_data_path, 'r') as f:
        step_data = json.load(f)
    return initial_data, step_data

def draw_road_network(world, data, z=0.5, color=None, life_time=0):
    """
    Draw road network from a JSON file in CARLA using the debug module.

    Args:
        client (carla.Client): CARLA client instance.
        json_file (str): Path to the JSON file containing road network data.
        z (float): Z-coordinate for the road points (default is 0.0).
        color (carla.Color): Color of the debug lines (default is green).
        life_time (float): Lifetime of the debug lines in seconds (default is 0.1).
    """
    if color is None:
        color = carla.Color(0, 0, 255)  # Default color is red

    # Extract road network
    road_network = data.get("road_network", [])

    # Draw each road boundary
    for road in road_network:
        for i in range(len(road) - 1):
            start = road[i]
            end = road[i + 1]
            start_location = carla.Location(x=start[0], y=-start[1], z=z)
            end_location = carla.Location(x=end[0], y=-end[1], z=z)
            world.debug.draw_line(start_location, end_location, thickness=0.1, color=color, life_time=life_time)

def replay_simulation_in_carla(
    client: carla.Client,
    opendrive_file: str,
    initial_data: Dict,
    step_data: List[Dict],
    fps: int = 10,
    output_dir: str = "./output_images"
):
    """
    在 CARLA 中回放仿真数据，并添加相机功能。

    参数:
        client: CARLA 客户端
        opendrive_file: OpenDRIVE 文件路径
        initial_data: 初始数据字典
        step_data: 时间步数据列表
        fps: 仿真帧率
        output_dir: 保存 RGB 图像的目录
    """

    # 提高生成地图的超时时间
    client.set_timeout(60.0)

    if not os.path.isfile(opendrive_file):
        raise FileNotFoundError(f"Replay... OpenDRIVE file not found: {opendrive_file}")

    with open(opendrive_file, "r", encoding="utf-8") as f:
        opendrive_content = f.read()

    print("Replay... Generating OpenDRIVE world...")

    parameters = carla.OpendriveGenerationParameters(
        vertex_distance=5.0,
        max_road_length=50.0,
        wall_height=0.0,
        additional_width=0.0,
        smooth_junctions=True,
        enable_mesh_visibility=False,
        enable_pedestrian_navigation=True
    )

    try:
        # 恢复生成 OpenDRIVE World 的功能
        # world = client.generate_opendrive_world(
        #     opendrive_content,
        #     parameters
        # )
        world = client.get_world()  # 直接获取当前世界，假设之前已经生成了地图
    except RuntimeError as e:
        print("Replay... Failed to generate OpenDRIVE world:", e)
        print("Possible causes: Invalid xodr geometry / CARLA server crashed / Not enough GPU memory")
        world = client.get_world()

    print("[INFO] OpenDRIVE world generated successfully")
    time.sleep(2)

    # 设置同步模式
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / fps
    world.apply_settings(settings)
    print("[INFO] Synchronous mode enabled")

    draw_road_network(world, initial_data)

    carla_map = world.get_map()
    print(f"[INFO] Loaded map: {carla_map.name}")

    blueprints = world.get_blueprint_library()
    actors = []
    actor_map = {} # 用于映射 ID 和 Actor，防止索引错乱

    # 生成自车
    ego_data = initial_data["ego_vehicle"]["initial_state"]  # [x, y, vx, vy, heading, length, width, 1]
    ego_bp = blueprints.filter('vehicle.tesla.model3')[0]

    # 稍微抬高 Z 轴以避免与地面初始碰撞
    ego_transform = carla.Transform(
        carla.Location(x=ego_data[0], y=-ego_data[1], z=0.5),
        carla.Rotation(yaw=np.degrees(-ego_data[4]))
    )

    # 使用 try_spawn_actor 替代 spawn_actor 防止碰撞报错中断

    print("actors: ", world.get_actors())

    ego_actor = world.try_spawn_actor(ego_bp, ego_transform)
    print(f"[INFO] Attempting to spawn ego vehicle {ego_actor} at: {ego_transform}")
    # if ego_actor is None:
    #     print("[WARN] Ego vehicle spawn failed at data location, trying map spawn points...")
    #     for sp in carla_map.get_spawn_points():
    #         ego_actor = world.try_spawn_actor(ego_bp, sp)
    #         if ego_actor is not None:
    #             ego_transform = sp
    #             break

    if ego_actor is None:
        raise RuntimeError("Failed to spawn ego vehicle due to collision.")
    else:
        ego_actor.set_simulate_physics(False)  # 禁用物理以避免碰撞影响

    actors.append(ego_actor)
    actor_map["ego"] = ego_actor

    # 清空 output_dir
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(output_dir, exist_ok=True)

    # 创建相机并附加到自车
    camera_bp = blueprints.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", "800")
    camera_bp.set_attribute("image_size_y", "600")
    camera_bp.set_attribute("fov", "90")
    camera_transform = carla.Transform(
        carla.Location(x=0, y=0, z=35),
        carla.Rotation(pitch=-90)
    )
    camera = world.try_spawn_actor(camera_bp, camera_transform, attach_to=ego_actor)

    os.makedirs(output_dir, exist_ok=True)
    def save_image(image):
        image.save_to_disk(f"{output_dir}/frame_{image.frame:06d}.png")
    camera.listen(save_image)

    # 生成其他交通参与者
    for agent_id, agent_data in initial_data["agents"].items():
        agent_type = agent_data["agent_type"]  # 获取 agent 的类型
        agent_state = agent_data["initial_state"][0]  # 获取 agent 的初始状态

        # 根据 agent 类型选择蓝图
        if agent_type == 1:  # vehicle
            agent_bp = blueprints.filter("vehicle.*")[0]
        elif agent_type == 2:  # pedestrian
            agent_bp = blueprints.filter("walker.pedestrian.*")[0]
        elif agent_type == 3:  # cyclist
            agent_bp = blueprints.filter("vehicle.bh.crossbike")[0]
        else:
            print(f"[WARNING] Unknown agent type {agent_type} for agent {agent_id}. Skipping...")
            continue

        # 设置 agent 的初始位置和朝向
        agent_transform = carla.Transform(
            carla.Location(x=agent_state[0], y=-agent_state[1], z=0.5 if agent_type != 2 else 0.0),
            carla.Rotation(yaw=np.degrees(-agent_state[4]))
        )

        print(f"[INFO] Spawning agent {agent_id} of type {agent_type} at: {agent_transform}")
        agent_actor = world.try_spawn_actor(agent_bp, agent_transform)
        actors.append(agent_actor)
        actor_map[agent_id] = agent_actor

    # 回放时间步数据
    try:
        print(type(step_data), " ", len(step_data))
        frames = len(step_data["ego_vehicle"])
        print("frames: ", frames)
        for frame_idx in range(frames):
            # 更新自车状态
            print(f"[INFO] Frame {frame_idx}/{frames}...")
            ego_state = step_data["ego_vehicle"][frame_idx]  # [x, y, vx, vy, heading, length, width, 1]
            ego_transform = carla.Transform(
                carla.Location(x=ego_state[0], y=-ego_state[1], z=0.5),
                carla.Rotation(yaw=np.degrees(-ego_state[4]))
            )
            actor_map["ego"].set_transform(ego_transform)
            world.get_spectator().set_transform(carla.Transform(
                carla.Location(x=ego_state[0], y=-ego_state[1], z=35),
                carla.Rotation(pitch=-90)
            ))

            # 更新其他交通参与者状态
            for agent_id, agent_state in step_data.items():
                if agent_id in actor_map and agent_id.startswith("agent"):
                    frame_data = agent_state[frame_idx]  # [x, y, vx, vy, heading, length, width, 1]

                    agent_transform = carla.Transform(
                        carla.Location(x=frame_data[0], y=-frame_data[1], z=0.5),
                        carla.Rotation(yaw=np.degrees(-frame_data[4]))
                    )
                    actor_map[agent_id].set_transform(agent_transform)

            time.sleep(0.5)
            world.tick()

    finally:
        # 清理资源
        for actor in actors:
            actor.destroy()
        if camera.is_alive:
            camera.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("仿真回放完成")

def generate_opendrive_file(initial_data: Dict, opendrive_output: str) -> str:
    """
    根據 scenario_0000_initial.json 的路網數據生成 OpenDrive 文件。

    參數:
        initial_data: 包含路網數據的字典
        opendrive_output: 生成的 OpenDrive 文件路徑

    返回:
        opendrive_output: 生成的 OpenDrive 文件路徑
    """
    road_network = initial_data.get("road_network", [])

    # 創建 OpenDrive 根元素
    root = ET.Element("OpenDRIVE")
    header = ET.SubElement(root, "header", attrib={
        "revMajor": "1",
        "revMinor": "4",
        "name": "GeneratedMap",
        "version": "1.00",
        "date": "2026-06-15",
        "north": "0",
        "south": "0",
        "east": "0",
        "west": "0"
    })

    # 遍歷路網數據，創建道路
    for road_id, road_points in enumerate(road_network):
        road = ET.SubElement(root, "road", attrib={
            "name": f"Road_{road_id}",
            "length": str(len(road_points)),
            "id": str(road_id),
            "junction": "-1"
        })
        plan_view = ET.SubElement(road, "planView")

        # 添加幾何信息
        for idx, point in enumerate(road_points):
            x, y = point[0], point[1] if len(point) > 1 else 0.0
            geometry = ET.SubElement(plan_view, "geometry", attrib={
                "s": str(idx),
                "x": str(x),
                "y": str(y),
                "hdg": "0.0",
                "length": "1.0"  # 假設每段長度為 1.0
            })
            ET.SubElement(geometry, "line")

    # 保存生成的 OpenDrive 文件
    tree = ET.ElementTree(root)
    tree.write(opendrive_output, encoding="UTF-8", xml_declaration=True)

    print(f"OpenDrive 地圖已生成並保存到 {opendrive_output}")
    return opendrive_output


def load_opendrive_file(opendrive_file: str) -> str:
    """
    讀取 OpenDrive 文件內容。

    參數:
        opendrive_file: OpenDrive 文件路徑

    返回:
        opendrive_content: OpenDrive 文件內容
    """
    if not os.path.isfile(opendrive_file):
        raise FileNotFoundError(f"OpenDRIVE file not found: {opendrive_file}")

    with open(opendrive_file, "r", encoding="utf-8") as f:
        opendrive_content = f.read()
    print(f"OpenDrive 文件已讀取: {opendrive_file}")
    return opendrive_content

if __name__ == "__main__":
    # 示例用法
    initial_data_path = "./carla_maps/initial_data/scenario_0000_initial.json"
    step_data_path = "./carla_maps/step_data/scenario_0000_all_steps.json"
    # opendrive_output = "output.xodr"
    opendrive_file_path = "minimal.xodr"
    output_dir = "./output_images"
    carla_path = "./Carla/CarlaUE4.sh"
    generate_new_opendrive = False  # 開關參數，控制是否生成新文件

    # 1. 啟動 CARLA 仿真器（内部已增加端口轮询等待，无需额外长 sleep）
    carla_process = start_carla_simulator(carla_path)
    print("[INFO] CARLA simulator started successfully.")

    try:
        # 2. 连接 CARLA 客户端
        client = carla.Client("localhost", 2000)
        client.set_timeout(20.0)
        print("[INFO] Connected to CARLA simulator.")

        # 3. 加载仿真数据
        initial_data, step_data = load_simulation_data(initial_data_path, step_data_path)
        print("[INFO] Simulation data loaded finished.")

        # 4. 生成或讀取 OpenDRIVE 文件
        if generate_new_opendrive:
            opendrive_file_path = generate_opendrive_file(initial_data, opendrive_file_path)

        opendrive_content = load_opendrive_file(opendrive_file_path)
        print("[INFO] OpenDRIVE file loaded finished.")

        # 5. 回放仿真
        print("Starting simulation replay in CARLA...")
        replay_simulation_in_carla(client, opendrive_file_path, initial_data, step_data, fps=10, output_dir=output_dir)

    finally:
        # 確保在仿真結束後關閉 CARLA 仿真器
        if carla_process:
            print("[INFO] Shutting down CARLA simulator...")
            carla_process.terminate()  # 終止 CARLA 進程
            carla_process.wait()  # 等待進程完全退出
            print("[INFO] CARLA simulator shut down successfully.")




    # 1.檢查carla無法正常啓動問題
    # 2.檢查地圖生成模塊
