"""在原失败输入上验证五分钟优化截断，保存候选但不强行执行。"""
import contextlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('PROJECT_ROOT',str(ROOT));os.environ.setdefault('SCRATCH_ROOT',str(ROOT))
os.environ.setdefault('DATASET_ROOT',str(ROOT/'metadata'))


def main():
    import numpy as np
    from hydra import compose,initialize
    from simulator import Simulator
    from policies.idm_policy import IDMPolicy
    from policies.strive_adapter import StriveController
    from policies.joint_safety import safe_joint_candidates
    output=Path(sys.argv[1]);output.mkdir(exist_ok=False)
    with (output/'diagnostic.log').open('w') as log,contextlib.redirect_stdout(log):
        with initialize(version_base=None,config_path='../../cfgs'):
            cfg=compose(config_name='config',overrides=['sim.traffic_model.backend=external_joint',
                'sim.visualize=false','sim.policy=idm','sim.seed=1000045','sim.llm.obstacles.enabled=false'])
        env=Simulator(cfg)
        old=ROOT/'experiments/e1_five_arm_idm_100_modelpool_20261012_r4/frozen_100_scenes.json'
        env.test_files=[json.loads(old.read_text())['scenario_files'][0]]
        obs=env.reset(0);policy=IDMPolicy(cfg,env);policy.reset(obs)
        controller=StriveController('/home2/zhaoyx/STRIVE-riskweaver',
            '/home2/zhaoyx/STRIVE-riskweaver/.venv/bin/python',
            ROOT/'checkpoints/strive_official_20261008/traffic_model.pth',output/'planning')
        start=time.monotonic();joint=controller.predict(env,policy)
        states=env.data_dict['agent'][-1][joint.agent_ids]
        safe=safe_joint_candidates(joint.positions_global[:,None],joint.yaws_global[:,None],
            np.broadcast_to(np.eye(3),(len(states),3,3)),states[:,5:7])
        result={'status':'passed','diagnostic_only':True,'real_executed_frames':0,
            'wall_seconds':time.monotonic()-start,'background_count':len(states),
            'background_joint_geometry_safe':bool(safe[0]),'audit':joint.metadata}
        (output/'candidate.json').write_text(json.dumps({'route':env.scenario_dict['route'].tolist(),
            'ego':env.ego_state.tolist(),'agents':states.tolist(),'positions':joint.positions_global.tolist(),
            'yaws':joint.yaws_global.tolist()}))
        (output/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()
