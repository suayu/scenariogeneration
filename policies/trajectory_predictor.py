import torch
import numpy as np
from tbsim.evaluation.policy_composers import get_policy_composer
from tbsim.configs.eval_config import EvaluationConfig
from tbsim.policies.wrappers import RolloutWrapper
from tbsim.utils.batch_utils import set_global_batch_type, batch_utils
from tbsim.utils.tensor_utils import to_torch

def preprocess_simulator_state(simulator_state, config):
    """
    将仿真器状态转换为模型输入格式

    Args:
        simulator_state: 仿真器提供的状态字典，包含：
            - agents: List[Dict]，每个智能体包含 id, x, y, yaw, speed, length, width
            - history: Dict[int, List]，每个智能体的历史轨迹 (最近N帧)
            - map_data: 可选，地图信息
        config: 模型配置，包含历史帧数、批量大小等

    Returns:
        batch: 模型输入字典
    """
    # 1. 设置批量类型（根据数据集选择）
    set_global_batch_type("trajdata")  # 或 "l5kit"

    # 2. 提取智能体信息
    agent_ids = [a['id'] for a in simulator_state['agents']]
    num_agents = len(agent_ids)
    T_history = config.get('history_frames', 20)  # 默认2秒@10Hz

    # 3. 构建历史张量
    history_pos = np.zeros((num_agents, T_history, 2), dtype=np.float32)
    history_yaws = np.zeros((num_agents, T_history), dtype=np.float32)
    history_vels = np.zeros((num_agents, T_history), dtype=np.float32)
    history_avails = np.ones((num_agents, T_history), dtype=bool)

    for i, agent in enumerate(simulator_state['agents']):
        # 获取该智能体的历史轨迹（从最近到最远）
        hist = simulator_state['history'].get(agent['id'], [])
        hist = hist[-T_history:]  # 取最近T_history帧

        # 填充历史数据（不足则补零并标记无效）
        for t, frame in enumerate(hist):
            if t >= T_history:
                break
            history_pos[i, t] = [frame['x'], frame['y']]
            history_yaws[i, t] = frame['yaw']
            history_vels[i, t] = frame['speed']
        if len(hist) < T_history:
            history_avails[i, len(hist):] = False

    # 4. 构建当前状态
    curr_speed = np.array([a['speed'] for a in simulator_state['agents']], dtype=np.float32)
    extent = np.array([[a['length'], a['width'], 2.0] for a in simulator_state['agents']], dtype=np.float32)

    # 5. 组装batch字典
    batch = {
        'history_positions': torch.from_numpy(history_pos).unsqueeze(0),  # [1, A, T, 2]
        'history_yaws': torch.from_numpy(history_yaws).unsqueeze(0),      # [1, A, T]
        'history_vels': torch.from_numpy(history_vels).unsqueeze(0),      # [1, A, T]
        'history_avails': torch.from_numpy(history_avails).unsqueeze(0),  # [1, A, T]
        'curr_speed': torch.from_numpy(curr_speed).unsqueeze(0),          # [1, A]
        'extent': torch.from_numpy(extent).unsqueeze(0),                  # [1, A, 3]
        'agent_type': torch.zeros(num_agents, dtype=torch.long).unsqueeze(0),  # [1, A]
    }

    # 6. 可选：添加栅格化地图
    if config.get('use_raster', False):
        raster = render_raster_map(simulator_state['map_data'], agent_positions)
        batch['raster_map'] = torch.from_numpy(raster).unsqueeze(0)  # [1, C, H, W]

    return batch

class TrajectoryPredictor:
    """封装TBSIM轨迹预测模型的推理接口"""

    def __init__(self, config_path, ckpt_path, device='cuda:0'):
        """
        Args:
            config_path: 评估配置文件路径 (JSON)
            ckpt_path: 模型checkpoint路径或YAML
            device: 推理设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 1. 加载配置
        self.eval_cfg = EvaluationConfig()
        if config_path:
            import json
            with open(config_path, 'r') as f:
                external_cfg = json.load(f)
            self.eval_cfg.update(**external_cfg)

        # 2. 设置checkpoint路径
        self.eval_cfg.ckpt.policy.ckpt_dir = ckpt_path
        self.eval_cfg.ckpt.policy.ckpt_key = 'best'  # 或指定具体迭代

        # 3. 构建策略（自动加载模型权重）
        from tbsim.evaluation.policy_composers import BC  # 或其他策略类
        composer = BC(self.eval_cfg, self.device)
        self.policy, self.exp_config = composer.get_policy()
        self.policy.eval()  # 切换到评估模式

        # 4. 可选：包装为RolloutWrapper（支持多智能体）
        self.rollout_policy = RolloutWrapper(agents_policy=self.policy)

        print(f"Model loaded successfully on {self.device}")

    def predict(self, simulator_state, num_modes=1):
        """
        单步轨迹预测

        Args:
            simulator_state: 仿真器状态(见preprocess_simulator_state)
            num_modes: 返回的模态数量

        Returns:
            predictions: Dict,包含预测轨迹和置信度
        """
        with torch.no_grad():
            # 1. 预处理
            batch = preprocess_simulator_state(simulator_state, self.exp_config)

            # 2. 移至设备
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # 3. 模型推理
            # 注意：具体调用方式取决于策略类型，以下是通用模式
            if hasattr(self.policy, 'forward'):
                outputs = self.policy(batch)
            elif hasattr(self.policy, 'predict'):
                outputs = self.policy.predict(batch)
            else:
                raise AttributeError("Policy has no forward or predict method")

            # 4. 解析输出
            # 假设输出包含 trajectories 和 probs
            # 形状: trajectories [B, A, M, T_future, 2], probs [B, A, M]
            trajectories = outputs.get('trajectories', outputs.get('action'))
            probs = outputs.get('probs', None)

            # 5. 取第一个batch和指定模态数
            trajectories = trajectories[0].cpu().numpy()  # [A, M, T, 2]
            if probs is not None:
                probs = probs[0].cpu().numpy()  # [A, M]
            else:
                # 若无概率，均匀分配
                probs = np.ones(trajectories.shape[:2]) / trajectories.shape[1]

            # 6. 按概率排序，取top-k
            if num_modes < trajectories.shape[1]:
                top_indices = np.argsort(probs, axis=1)[:, -num_modes:]
                # 简化：返回所有模态
                # 实际使用时可按需筛选

            return {
                'trajectories': trajectories,  # [A, M, T, 2]
                'probabilities': probs,        # [A, M]
                'agent_ids': [a['id'] for a in simulator_state['agents']]
            }
