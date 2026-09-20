"""独立后台冒烟入口：冻结运行参数并保存退出状态，不自动跳过验收。"""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home2/zhaoyx/scenario-dreamer')
OUT = ROOT / 'experiments/riskweaver_target_smoke_20260911_r6'


def main():
    # 专用目录拒绝覆盖，保留失败现场。
    OUT.mkdir(exist_ok=False)
    env = dict(os.environ, PROJECT_ROOT=str(ROOT), SCRATCH_ROOT=str(ROOT),
               DATASET_ROOT=str(ROOT/'metadata'), PYTHONPATH=f'{ROOT}:{ROOT}/safe-sim:{ROOT}/safe-sim/trajdata/src',
               MPLBACKEND='Agg', PYTHONUNBUFFERED='1', HYDRA_FULL_ERROR='1',
               RISKWEAVER_DIAGNOSTIC_DIR=str(OUT/'diagnostics'))
    overrides = ['sim.seed=42','sim.evaluation.max_scenarios=1','sim.steps=400',
        f'+sim.attack_request_file={ROOT}/tests/riskweaver_joint_instructions.txt',
        'sim.llm.provider=qwen','sim.llm.attack_mode=joint','sim.llm.obstacles.enabled=true',
        'sim.traffic_model.guidance.mode=llm_joint',
        'sim.traffic_model.iterative_adversarial.profile_enabled=true',
        'sim.traffic_model.iterative_adversarial.escalation_enabled=true',
        'sim.traffic_model.iterative_adversarial.full_method_enabled=true',
        'sim.evaluation.difficulty_control.mode=target',
        'sim.evaluation.difficulty_control.target_difficulty=0.50',
        'sim.evaluation.difficulty_control.tolerance=0.10',
        'sim.evaluation.difficulty_control.smoke_acceptance=true',
        'sim.visualize=true','sim.lightweight=true',
        'sim.visualization.show_diffusion_candidates=true',
        f'sim.movie_path={OUT}/movies',f'sim.scenario_data_output_path={OUT}/carla',
        f'hydra.run.dir={OUT}/hydra']
    cmd = ['bash','scripts/run_with_llm_provider.sh','qwen',sys.executable,'run_simulation.py',*overrides]
    (OUT/'launch.json').write_text(json.dumps({'command':cmd,'pid':os.getpid()},indent=2),encoding='utf-8')
    # 配置仅含非敏感运行参数；统一密钥文件由现有启动器加载，不复制或打印。
    with (OUT/'resolved_config.yaml').open('w') as snapshot:
        subprocess.run(cmd+['--cfg','job','--resolve'],cwd=ROOT,env=env,stdout=snapshot,check=True)
    with (OUT/'run.log').open('w') as log:
        code = subprocess.call(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    (OUT/'exit_status.json').write_text(json.dumps({'exit_code':code}),encoding='utf-8')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
