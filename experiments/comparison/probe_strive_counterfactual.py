"""真实统一仿真器上的副本隔离诊断；CV输入不属于STRIVE模型实验。"""
import contextlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('PROJECT_ROOT',str(ROOT));os.environ.setdefault('SCRATCH_ROOT',str(ROOT))
os.environ.setdefault('DATASET_ROOT',str(ROOT/'metadata'))
import numpy as np
from hydra import compose,initialize
from policies.idm_policy import IDMPolicy
from policies.strive_adapter import counterfactual_rollout,make_request
from policies.traffic_types import JointTrajectory
from simulator import Simulator

out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
with (out/'diagnostic.log').open('w') as log,contextlib.redirect_stdout(log):
    with initialize(version_base=None,config_path='../../cfgs'):
        cfg=compose(config_name='config',overrides=['sim.traffic_model.backend=external_joint','sim.visualize=false',
             'sim.policy=idm','sim.seed=42','sim.llm.obstacles.enabled=false'])
    env=Simulator(cfg)
    manifest=json.loads((ROOT/'experiments/scenario_dreamer_ldm_expand250_20260928_r1/riskweaver_frozen_250_scenes.json').read_text())
    env.test_files=manifest['scenario_files']
    obs=env.reset(0);policy=IDMPolicy(cfg,env);policy.reset(obs)
    frame=env.get_scenario_frame(16)
    ego=np.zeros((16,8),np.float32);mask=np.zeros(16,bool)
    ego[-1]=frame.ego_state_global;mask[-1]=True
    request=make_request(frame,ego,mask)
    ids=np.flatnonzero(env.agent_active)
    states=env.data_dict['agent'][-1][ids].copy()
    positions=states[:,None,:2]+states[:,None,2:4]*(np.arange(1,11)*env.dt)[None,:,None]
    yaws=np.broadcast_to(states[:,None,4:5],(len(ids),10,1)).copy()
    velocity=np.broadcast_to(states[:,None,2:4],(len(ids),10,2)).copy()
    joint=JointTrajectory(env.current_step,ids,positions,yaws,velocity,np.ones((len(ids),10),bool))
    before=(env.current_step,env.ego_state.copy(),env.data_dict['agent'][-1].copy(),len(env.data_dict['agent']),policy.t)
    predicted=counterfactual_rollout(env,policy,joint)
    assert env.current_step==before[0] and len(env.data_dict['agent'])==before[3] and policy.t==before[4]
    np.testing.assert_array_equal(env.ego_state,before[1])
    np.testing.assert_array_equal(env.data_dict['agent'][-1],before[2])
    assert predicted.shape==(10,4) and np.isfinite(predicted).all()
    result={'status':'passed','diagnostic_only':True,'pretrained_weights_loaded':False,
       'counterfactual_frames':10,'real_executed_frames':0,'original_state_unchanged':True,
       'background_count':len(ids),'input_past_shape':list(request['past'].shape),
       'first_ego_prediction':predicted[0].tolist()}
    (out/'result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
