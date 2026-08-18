import os
import hydra
from omegaconf import OmegaConf
from cfgs.config import CONFIG_PATH

# 导入项目中的工具函数
from utils.train_helpers import cache_latent_stats, set_latent_stats

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    """
    独立脚本：仅生成 Autoencoder 的潜变量统计文件 (latent_stats.pkl)
    使用方法: python gen_latent_stats.py dataset_name=waymo model_name=ldm ae.train.run_name=scenario_dreamer_autoencoder_waymo
    """
    dataset_name = cfg.dataset_name.name

    # 1. 构建 LDM 配置（因为 cache_latent_stats 需要 LDM 的 dataset 配置）
    if cfg.model_name == 'ldm':
        cfg_ae = cfg.ae
        cfg = cfg.ldm
        OmegaConf.set_struct(cfg, False)
        OmegaConf.set_struct(cfg_ae, False)
        cfg.dataset_name = dataset_name
        cfg_ae.dataset_name = dataset_name
        OmegaConf.set_struct(cfg, True)
        OmegaConf.set_struct(cfg_ae, True)
    else:
        raise ValueError("请使用 model_name=ldm 来加载正确的数据集配置")

    # 2. 核心调用：生成 latent_stats.pkl
    print(f"开始生成潜变量统计文件，输出路径：{cfg.dataset.latent_stats_path}")
    if not os.path.exists(cfg.dataset.latent_stats_path):
        cache_latent_stats(cfg)  # 调用函数生成 .pkl
        print("✅ 潜变量统计文件生成成功！")
    else:
        print("⚠️  文件已存在，跳过生成。")

    # 3. 验证：尝试加载（可选）
    cfg = set_latent_stats(cfg)
    print("✅ 统计文件加载验证通过。")

if __name__ == '__main__':
    main()


    """
    python gen_latent_stats.py \
  dataset_name=waymo \
  model_name=ldm \
  ae.train.run_name=scenario_dreamer_autoencoder_waymo
    """