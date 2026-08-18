import torch
from tbsim.configs.eval_config import EvaluationConfig
from tbsim.configs.guidance_config import GuidanceConfig
from tbsim.evaluation.policy_composers import Diffusion
from tbsim.configs.base import Dict

class DiffusionModelWrapper:
    """封装 Diffusion 策略组合器，供外部仿真器调用"""

    def __init__(self, config_path, ckpt_path, device='cuda:0'):
        # 1. 构建评估配置
        self.eval_cfg = EvaluationConfig()
        self.eval_cfg.ckpt.planner.ckpt_dir = ckpt_path
        self.eval_cfg.ckpt.planner.ckpt_key = 'best'
        self.eval_cfg.guidance = False  # 默认关闭引导

        self.device = torch.device(device)

        # 2. 创建引导配置（可选）
        self.guide_config = GuidanceConfig()

        # 3. 实例化 Diffusion 组合器
        self.composer = Diffusion(
            eval_config=self.eval_cfg,
            device=self.device,
            ckpt_root_dir=self.eval_cfg.ckpt_root_dir
        )

        # 4. 加载模型
        self.policy, self.exp_config = self.composer.get_policy()
        self.policy.eval()

    def predict(self, obs_dict, sample=True, num_samples=5):
        """
        执行轨迹预测

        Args:
            obs_dict: 观测字典（见数据适配章节）
            sample: True=多模态扩散采样False=确定性预测
            num_samples: 采样数量（仅当 sample=True 时有效）

        Returns:
            action: Action 对象，包含 positions 和 yaws
            info: 额外信息（包含所有采样样本等）
        """
        with torch.no_grad():
            # 将观测移至设备
            obs_torch = {k: v.to(self.device) if torch.is_tensor(v) else v
                         for k, v in obs_dict.items()}
            # 调用模型
            action, info = self.policy.get_action(
                obs_torch,
                sample=sample,
                # 可传入 plan_samples 实现目标条件生成
            )
        return action, info