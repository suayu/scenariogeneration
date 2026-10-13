"""两个固定场景的同输入计算器对照，另验证Scenario Dreamer隔离预测与D。"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('PROJECT_ROOT',str(ROOT))
os.environ.setdefault('SCRATCH_ROOT',str(ROOT))
os.environ.setdefault('DATASET_ROOT',str(ROOT/'metadata'))


def main():
    import numpy as np
    import torch
    from hydra import compose,initialize
    from policies.idm_policy import IDMPolicy
    from policies.commonroad_reachability import CommonRoadReachability
    from policies.risk_metrics import RouteReachability
    from experiments.comparison.ctrl_sim_forecast import CtrlSimJointForecaster,validate_forecast
    from experiments.comparison.formal_metrics import FormalRiskObserver
    from experiments.comparison.e1_protocol import group_overrides
    from simulator import Simulator
    from experiments.comparison.e1_metrics import ExecutionMetrics
    from policies.evaluation_trace import AttemptTrace
    from utils.viz import generate_video
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    source=json.loads(args.manifest.read_text())
    # 在看到计算结果之前，固定使用清单头两个场景；不依结果替换。
    files=source['scenario_files'][:2]
    rows=[]
    with (args.output_dir/'diagnostic.log').open('w') as log,contextlib.redirect_stdout(log):
        with initialize(version_base=None,config_path='../../cfgs'):
            overrides=group_overrides('scenario_dreamer')
            overrides += ['sim.visualize=true',f'sim.movie_path={args.output_dir / "movies"}',
                          f'sim.scenario_data_output_path={args.output_dir / "carla"}']
            cfg=compose(config_name='config',overrides=overrides)
        env=Simulator(cfg);env.test_files=files
        grid=RouteReachability(cfg.sim.evaluation.reachability)
        common=CommonRoadReachability(cfg.sim.evaluation.reachability)
        for index,path in enumerate(files):
            obs=env.reset(index);policy=IDMPolicy(cfg,env);policy.reset(obs)
            forecaster=CtrlSimJointForecaster(env,3.,42,tracking='persistent')
            observer=FormalRiskObserver(cfg.sim.evaluation)
            execution=ExecutionMetrics(env)
            movie=args.output_dir/'movies'/f'scene_{index}';movie.mkdir(parents=True)
            trace=AttemptTrace(movie);trace.state(env,initial=True)
            env.render_state(name=f'scene_{index}',movie_path=str(movie))
            case={'scene':path,'scene_index':source.get('original_scenario_indices',[0,1])[index],
                  'measurements':[],'diagnostic_only':True}
            # 只在0/10/20帧进行同预测评估，降低成本；不是完整场景效果实验。
            for step in range(21):
                if step in (0,10,20):
                    observer.begin_frame(env,expect_joint_prediction=True)
                    before=(env.current_step,env.ego_state.copy(),env.data_dict['agent'][-1].copy(),policy.t)
                    rng=torch.random.get_rng_state().clone()
                    forecast=forecaster.predict(env,policy)
                    forecast_error=None
                    try:validate_forecast(forecast,env,3.)
                    except ValueError as error:forecast_error=str(error)
                    assert env.current_step==before[0] and policy.t==before[3]
                    np.testing.assert_array_equal(env.ego_state,before[1])
                    np.testing.assert_array_equal(env.data_dict['agent'][-1],before[2])
                    assert torch.equal(rng,torch.random.get_rng_state())
                    env.pending_joint_trajectory=forecast
                    measurement={'source_step':step,'prediction_metadata':forecast.metadata,
                                 'original_state_and_random_stream_unchanged':True,
                                 'forecast_validation_error':forecast_error,
                                 'calculator_input':'joint_forecast' if forecast_error is None else 'CV_reference_diagnostic_only'}
                    for name,backend in [('route_grid',grid),('commonroad',common)]:
                        start=time.monotonic()
                        result=backend.compute(env,forecast_error is None,env.get_static_obstacles())
                        measurement[name]={**result,'evaluation_wall_seconds':time.monotonic()-start}
                    trace.append({'kind':'evaluation_forecast','step':step,'metadata':forecast.metadata,
                                  'positions':forecast.positions_global.tolist()})
                    observer.finish_prediction(env,forecast,'ctrl_sim_rollout')
                    # 清除评估未来；实际执行仍由原CtRL-Sim产生动作。
                    env.pending_joint_trajectory=None
                    obs,done,info=env.step(policy.act(obs),None,None)
                    execution.observe(env,info)
                    record=observer.finish_transition(info,execution.last_tca)
                    measurement['D_window']=record
                    case['measurements'].append(measurement)
                    trace.state(env,info)
                    env.render_state(name=f'scene_{index}',movie_path=str(movie))
                    if done:break
                else:
                    obs,done,info=env.step(policy.act(obs),None,None)
                    execution.observe(env,info)
                    trace.state(env,info)
                    env.render_state(name=f'scene_{index}',movie_path=str(movie))
                    if done:break
            case['video']=generate_video(name=f'scene_{index}',output_dir=str(movie),delete_images=False)
            case['actual_steps']=env.current_step
            rows.append(case)
            (args.output_dir/'results.partial.json').write_text(json.dumps(rows,indent=2))
        common.close()
    (args.output_dir/'results.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps({'scenes':len(rows),'measurements':sum(len(x['measurements']) for x in rows)}))


if __name__=='__main__':
    main()
