"""冻结清单下的STRIVE闭环适配运行器；真实权重冒烟前不得作为正式E1结果。"""
import argparse
import contextlib
import copy
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

import numpy as np
from hydra import compose,initialize
from omegaconf import OmegaConf
from experiments.comparison.formal_metrics import FormalRiskObserver
from experiments.comparison.run_passive_baseline import validate_manifest,sanitize_nonfinite,aggregate
from policies.difficulty_control import ability_score
from policies.evaluation_trace import AttemptTrace
from policies.idm_policy import IDMPolicy
from policies.joint_safety import NoSafeJointCandidate,safe_joint_candidates,validate_execution_prefix
from policies.route_progress import normalized_route_progress
from policies.strive_adapter import StriveController
from policies.traffic_types import JointTrajectory
from run_simulation import merge_episode_info
from simulator import Simulator
from tests.run_full_adversarial_evaluation import min_center_distance_and_ttc,set_seed
from utils.viz import generate_video


class BaselineRuntimeFailure(RuntimeError):
    """保存失败场景结果后仍使进程失败，不能把服务异常当正常无攻击。"""
    def __init__(self, result, error):
        super().__init__(str(error))
        self.result = result


def empty_joint_prediction(source_step, horizon):
    """没有活跃背景车辆时仅生成显式空联合状态，不冒充模型调用或攻击。"""
    return JointTrajectory(source_step, np.empty(0, dtype=np.int64),
        np.empty((0, horizon, 2), dtype=np.float32), np.empty((0, horizon, 1), dtype=np.float32),
        np.empty((0, horizon, 2), dtype=np.float32), np.empty((0, horizon), dtype=bool),
        {'no_attack_reason': 'no_active_background_vehicles', 'attack_row': 0, 'model_invoked': False})


def slice_joint(joint,source_step,offset):
    """缓存仅在剩余时域内移动，不给末帧外推或冒充新的模型预测。"""
    if offset < 0 or offset >= joint.positions_global.shape[1]:
        raise ValueError('STRIVE cache exhausted')
    return JointTrajectory(source_step,joint.agent_ids.copy(),joint.positions_global[:,offset:].copy(),
        joint.yaws_global[:,offset:].copy(),joint.velocities_global[:,offset:].copy(),
        joint.valid_mask[:,offset:].copy(),{**joint.metadata,'cached_from_step':joint.source_step,'cache_offset':offset})


def run_episode(env,policy,index,steps,out,args):
    obs=env.reset(index)
    if hasattr(policy,'reset'):
        policy.reset(obs)
    out.mkdir(parents=True,exist_ok=False)
    trace=AttemptTrace(out)
    observer=FormalRiskObserver(env.cfg.sim.evaluation)
    trace.state(env,initial=True)
    terminal={'collision':False,'off_route':False,'completed':False,'progress':0.0}
    controller=StriveController(args.strive_root,args.strive_python,args.checkpoint,out/'planning',
                               args.init_iters,args.adv_iters,args.timeout)
    joint=None
    ttc=float('inf')
    count=0
    failure=None
    runtime_error=None
    plans=0
    attack_frames=0
    official_successful_plans=0
    collision_speed=None
    start=time.monotonic()
    for _ in range(steps):
        observer.begin_frame(env,expect_joint_prediction=True)
        try:
            offset=0 if joint is None else env.current_step-joint.source_step
            active=set(np.flatnonzero(env.agent_active))
            if joint is None or offset>=args.replan_steps or offset>=joint.positions_global.shape[1] or active!=set(joint.agent_ids):
                joint=(controller.predict(env,policy) if active
                       else empty_joint_prediction(env.current_step, int(round(6.0 / env.dt))))
                current=env.data_dict['agent'][-1][joint.agent_ids]
                safe=safe_joint_candidates(joint.positions_global[:,None],joint.yaws_global[:,None],
                    np.broadcast_to(np.eye(3),(len(current),3,3)),current[:,5:7])
                if not safe[0]:
                    raise NoSafeJointCandidate('STRIVE optimized background OBB collision or invalid geometry')
                offset=0
            executing=slice_joint(joint,env.current_step,offset)
            validate_execution_prefix(env.data_dict['agent'][-1],executing,env.get_scenario_frame().static_obstacles_global)
            env.inject_joint_trajectory(executing)
            if offset == 0 and len(joint.agent_ids):
                plans += 1
                official_successful_plans += int(bool(joint.metadata.get('official_adv_success')))
            observer.finish_prediction(env)
            trace.append({'kind':'prediction','step':env.current_step,'method':'strive','joint':{
                'agent_ids':executing.agent_ids.tolist(),'positions':executing.positions_global.tolist(),
                'yaws':executing.yaws_global.tolist(),'valid':executing.valid_mask.tolist(),'metadata':executing.metadata}})
            env.render_state(name=f'scenario_{index:03d}',movie_path=str(out))
            obs,terminated,info=env.step(policy.act(obs),None,None)
            terminal=merge_episode_info(terminal,info)
            _,current_ttc=min_center_distance_and_ttc(env.ego_state,env.data_dict['agent'][-1],env.agent_active)
            ttc=min(ttc,current_ttc)
            observer.finish_transition(info,current_ttc)
            trace.state(env,info)
            count+=1
            attack_frames += int(joint.metadata.get("attack_row", 0) > 0)
            if terminal['collision']:
                collision_speed=float(np.linalg.norm(env.ego_state[2:4]))
            if terminated:
                break
        except NoSafeJointCandidate as error:
            failure=str(error)
            trace.append({'kind':'generation_failure','step':env.current_step,'reason':failure})
            break
        except Exception as error:
            # 保留真实执行帧、初始/终止图和单场JSON，再向上报告失败。
            runtime_error = error
            trace.append({'kind':'runtime_failure','step':env.current_step,
                          'error_type':type(error).__name__,'reason':str(error)})
            break
    env.render_state(name=f'scenario_{index:03d}',movie_path=str(out))
    video=generate_video(name=f'scenario_{index:03d}',output_dir=str(out),delete_images=False)
    danger=observer.result()
    ea=[float(v) for v in observer.risk.ea_values if np.isfinite(v)]
    progress=normalized_route_progress(env.ego_state[:2],env.scenario_dict['route'])
    if progress is None:
        raise ValueError('degenerate route')
    score=ability_score(danger['danger'],terminal['completed'] and not terminal['collision'] and not terminal['off_route'],
                        terminal['collision'],terminal['off_route'],progress,.5)
    areas=[r['reachability'] for r in danger['records'] if r['reachability'] is not None]
    result={'scenario_index':index,'scenario_source':env.test_files[index],
       'status':'runtime_failure' if runtime_error else ('generation_failure' if failure else 'completed'),
       'generation_failure':bool(failure),'runtime_failure':runtime_error is not None,
       'runtime_failure_reason':str(runtime_error) if runtime_error else None,
       'metric_protocol':'ea_empty_set_zero_v2','attack_plan_count':plans,
       'attack_executed_frames':attack_frames,'official_adv_successful_plan_count':official_successful_plans,
       'generation_failure_reason':failure,'executed_steps':count,'collision':bool(terminal['collision']) if count else None,
       'off_route':bool(terminal['off_route']) if count else None,'completed':bool(terminal['completed']) if count else None,'progress':float(progress),
       'min_ttc_s':ttc,'ego_collision_speed_mps':collision_speed,
       'max_evasive_acceleration_mps2':max(ea) if ea else None,
       'mean_evasive_acceleration_mps2':float(np.mean(ea)) if ea else None,'ea_valid_frame_count':len(ea),
       'scenario_danger_score':danger['danger'],'danger_valid':danger['valid'],'danger_windows':danger['records'],
       'danger_valid_frame_count':danger['valid_frame_count'],'danger_total_frame_count':danger['total_frame_count'],
       'danger_coverage':danger['coverage'],'autonomous_driving_ability_score':score,'ability_valid':score is not None,
       'mean_reachable_area_m2':float(np.mean([a['dangerous_area_m2'] for a in areas])) if areas else None,
       'area_basis':'route_dynamics_reachable_terminal_area_not_map_area',
       'feasibility':trace.summary(),'video_path':str(video),'runtime_seconds':time.monotonic()-start,
       'fidelity':'official_STRIVE_optimizer_adapted_Waymo_map_and_CV_initialization','formal_ready':False}
    result=sanitize_nonfinite(result)
    (out/'attempt_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    if runtime_error is not None:
        raise BaselineRuntimeFailure(result, runtime_error) from runtime_error
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--strive-root',type=Path,default=Path('/home2/zhaoyx/STRIVE-riskweaver'))
    parser.add_argument('--strive-python',default='/home2/zhaoyx/STRIVE-riskweaver/.venv/bin/python')
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=400)
    parser.add_argument('--indices',default='0')
    parser.add_argument('--policy',choices=['idm','rl'],default='idm')
    parser.add_argument('--init-iters',type=int,default=75)
    parser.add_argument('--adv-iters',type=int,default=300)
    parser.add_argument('--replan-steps',type=int,default=10)
    parser.add_argument('--timeout',type=float,default=1800)
    args=parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f'Official STRIVE checkpoint missing: {args.checkpoint}')
    if not 1<=args.steps<=400 or args.replan_steps<1:
        raise ValueError('invalid frame/replanning budget')
    manifest,files=validate_manifest(args.manifest)
    indices=list(range(len(files))) if args.indices=='all' else [int(i) for i in args.indices.split(',')]
    if len(set(indices))!=len(indices) or any(i<0 or i>=len(files) for i in indices):
        raise ValueError('invalid/duplicate scene indices')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    overrides=['sim.seed=42',f'sim.policy={args.policy}',f'sim.steps={args.steps}','sim.visualize=true',
        'sim.lightweight=true','sim.traffic_model.backend=external_joint','sim.llm.obstacles.enabled=false']
    with initialize(version_base=None,config_path='../../cfgs'):
        cfg=compose(config_name='config',overrides=overrides)
    (args.output_dir/'resolved_config.yaml').write_text(OmegaConf.to_yaml(cfg,resolve=True),encoding='utf-8')
    (args.output_dir/'run_arguments.json').write_text(json.dumps(vars(args),default=str,indent=2),encoding='utf-8')
    set_seed(42)
    env=Simulator(cfg)
    env.test_files=[str(f) for f in files]
    if args.policy=='idm':
        policy=IDMPolicy(cfg,env)
    else:
        from policies.rl_policy import RLPolicy
        policy=RLPolicy(cfg.sim)
    records=[]
    with (args.output_dir/'run.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log):
        try:
            for index in indices:
                try:
                    record = run_episode(env,policy,index,args.steps,args.output_dir/'movies'/f'scenario_{index:03d}',args)
                except BaselineRuntimeFailure as error:
                    records.append(error.result)
                    report={'method':'strive','formal_ready':False,'aggregate':aggregate(records),'episodes':records}
                    (args.output_dir/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
                    raise
                records.append(record)
                report={'method':'strive','formal_ready':False,'aggregate':aggregate(records),'episodes':records}
                (args.output_dir/'results.partial.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
            (args.output_dir/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
            (args.output_dir/'exit_status.json').write_text(json.dumps({'exit_code':0,'formal_ready':False}),encoding='utf-8')
        except Exception as error:
            (args.output_dir/'exit_status.json').write_text(json.dumps({'exit_code':1,'error':str(error),'formal_ready':False}),encoding='utf-8')
            raise


if __name__=='__main__':
    main()
