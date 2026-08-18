import carla
import time
import os
import subprocess

def start_carla_simulator(carla_path: str, port: int = 2000):
    """
    啟動 CARLA 模擬器（無頭模式）。

    參數:
        carla_path: CARLA 模擬器的啟動腳本路徑，例如 "./CarlaUE4.sh"
        port: 模擬器運行的端口，默認為 2000
    """
    print("[INFO] Starting CARLA simulator...")
    cmd = [
        carla_path,
        "-RenderOffScreen",  # 無頭模式
        f"-carla-rpc-port={port}",  # 指定端口
        "-quality-level=Low"  # 降低圖形質量
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("[INFO] CARLA simulator started.")
    return process

def start_carla_client(port: int = 2000):
    """
    連接到 CARLA 模擬器，並返回客戶端對象。
    """
    client = carla.Client("localhost", port)
    client.set_timeout(20.0)  # 設置超時時間為 20 秒
    print("[INFO] Connected to CARLA simulator.")
    return client

def setup_camera(world, vehicle, output_dir):
    """
    在指定的車輛上創建一個相機，並設置保存圖像的回調函數。
    """
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", "800")
    camera_bp.set_attribute("image_size_y", "600")
    camera_bp.set_attribute("fov", "90")

    # 創建相機並附加到車輛
    camera_transform = carla.Transform(
        carla.Location(x=0, y=0, z=2.5),  # 相機位於車輛上方
        carla.Rotation(pitch=0)
    )
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)

    # 定義回調函數保存圖像
    def save_image(image):
        image.save_to_disk(f"{output_dir}/test_image.png")
        print("[INFO] Image saved to test_image.png")

    # 綁定回調函數
    camera.listen(save_image)
    return camera

def main():
    # 啟動 CARLA 模擬器
    carla_path = "./Carla/CarlaUE4.sh"  # 修改為 CARLA 的實際啟動腳本路徑
    carla_process = start_carla_simulator(carla_path)

    try:
        # 等待模擬器完全啟動
        print("[INFO] Waiting for CARLA simulator to fully start...")
        time.sleep(10)

        # 連接到 CARLA 模擬器
        client = start_carla_client()

        # 獲取世界對象
        world = client.get_world()

        # 獲取藍圖庫並生成一輛車輛
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter("vehicle.*")[0]
        spawn_point = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)

        print("[INFO] Vehicle spawned.")

        # 創建相機並拍攝照片
        output_dir = "./output_images"
        camera = setup_camera(world, vehicle, output_dir)

        # 等待一段時間以確保圖像保存
        time.sleep(5)

        # 清理資源
        print("[INFO] Cleaning up...")
        camera.destroy()
        vehicle.destroy()
        print("[INFO] Test completed.")

    finally:
        # 關閉 CARLA 模擬器
        print("[INFO] Shutting down CARLA simulator...")
        carla_process.terminate()
        carla_process.wait()
        print("[INFO] CARLA simulator shut down.")

if __name__ == "__main__":
    main()