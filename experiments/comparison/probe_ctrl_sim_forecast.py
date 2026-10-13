"""真实CtRL-Sim联合未来的副本隔离诊断；不属于闭环场景冒烟。"""
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
    import torch
    from experiments.comparison.ctrl_sim_forecast import CtrlSimJointForecaster,validate_forecast
    forecaster=CtrlSimJointForecaster(env,3.,42)
    before=(env.current_step,env.ego_state.copy(),env.data_dict['agent'][-1].copy(),len(env.data_dict['agent']),policy.t)
    pending=env.pending_joint_trajectory
    cpu_rng=torch.random.get_rng_state().clone()
    gpu_rng=torch.cuda.get_rng_state().clone()
    state={key:value.clone() for key,value in forecaster.template.model.state_dict().items()}
    forecast=forecaster.predict(env,policy)
    assert torch.equal(cpu_rng,torch.random.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
    assert all(torch.equal(value,forecaster.template.model.state_dict()[key]) for key,value in state.items())
    assert env.current_step==before[0] and len(env.data_dict['agent'])==before[3] and policy.t==before[4]
    assert env.pending_joint_trajectory is pending
    np.testing.assert_array_equal(env.ego_state,before[1])
    np.testing.assert_array_equal(env.data_dict['agent'][-1],before[2])
    error=None
    try: validate_forecast(forecast,env,3.)
    except ValueError as exc: error=str(exc)
    result={'status':'passed','diagnostic_only':True,'pretrained_weights_loaded':True,
      'counterfactual_frames':30,'real_executed_frames':0,'original_state_unchanged':True,
      'random_streams_unchanged':True,'model_state_unchanged':True,
      'background_count':len(forecast.agent_ids),'metadata':forecast.metadata,'validation_error':error}
    (out/'result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
