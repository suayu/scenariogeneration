"""五组共用闭环运行器：保持各方法执行，统一状态、风险观察器和失败落盘。"""
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('PROJECT_ROOT',str(ROOT))
os.environ.setdefault('SCRATCH_ROOT',str(ROOT))
os.environ.setdefault('DATASET_ROOT',str(ROOT/'metadata'))

import numpy as np
from hydra import compose,initialize
from omegaconf import OmegaConf
from experiments.comparison.e1_protocol import GROUPS,PROFILE_GROUPS,group_overrides,dump_json,protocol_snapshot
from experiments.comparison.e1_metrics import ExecutionMetrics,aggregate_attempts,trajectory_diagnostics,geometric_lane_area
from experiments.comparison.formal_metrics import FormalRiskObserver
from experiments.comparison.run_passive_baseline import validate_manifest,sanitize_nonfinite
from experiments.comparison.run_strive_baseline import empty_joint_prediction,slice_joint
from policies.difficulty_control import ability_score
from policies.evaluation_trace import AttemptTrace
from policies.idm_policy import IDMPolicy
from policies.joint_safety import NoSafeJointCandidate,safe_joint_candidates,validate_execution_prefix
from policies.route_progress import normalized_route_progress
from policies.strive_adapter import StriveController
from run_simulation import merge_episode_info
from scenario_generator import AdversarialScenarioGenerator
from simulator import Simulator
from tests.run_full_adversarial_evaluation import set_seed
from utils.viz import generate_video


INSTRUCTION=['通过背景交通参与者的切入、前向压迫、合流冲突等攻击性行为，挑战自车驾驶策略。仅选择离散攻击策略、目标及稀疏锚点，禁止放置静态障碍物。']


def prediction_record(joint):
    """保存实际执行的未来，与代理评估未来明确区分。"""
    if joint is None:
        return None
    return {'source_step':joint.source_step,'agent_ids':joint.agent_ids.tolist(),
        'positions_global':joint.positions_global.tolist(),'yaws_global':joint.yaws_global.tolist(),
        'velocities_global':joint.velocities_global.tolist(),'valid_mask':joint.valid_mask.tolist(),
        'metadata':joint.metadata}


def run_episode(env,policy,index,global_index,args,forecaster=None):
    """所有组使用同一实际状态指标；算法失败保留已发生的执行和碰撞。"""
    out=args.output_dir/'movies'/f'scenario_{global_index:03d}'
    out.mkdir(parents=True,exist_ok=False)
    effective_seed=(args.seed+1000003*global_index)%(2**31-1)
    set_seed(effective_seed)
    obs=env.reset(index)
    policy.reset(obs)
    trace=AttemptTrace(out)
    trace.state(env,initial=True)
    execution=ExecutionMetrics(env)
    geometric=geometric_lane_area(env)
    observer=FormalRiskObserver(env.cfg.sim.evaluation)
    observer.reset()
    terminal={'collision':False,'off_route':False,'completed':False,'progress':0.}
    generator=None
    if args.group in PROFILE_GROUPS:
        # 不让跨场景的模块全局记忆随 GPU 分片顺序变化。
        import scenario_generator
        scenario_generator._POLICY_ATTACK_MEMORY.clear()
        generator=AdversarialScenarioGenerator(env.cfg,INSTRUCTION,llm_planner=env.llm_planner)
        generator.reset_episode_stats()
        generator.query_trace=trace
        if generator.profile_pipeline is not None:
            generator.profile_pipeline.begin_attempt(env,out)
    gate=None
    if generator is not None:
        from policies.ego_profile import JointTrajectorySafetyGate
        gate=JointTrajectorySafetyGate(env.cfg.sim)
    controller=None
    if args.group=='strive':
        controller=StriveController(args.strive_root,args.strive_python,args.checkpoint,out/'planning',
            args.init_iters,args.adv_iters,args.timeout)
    cache=None
    count=plans=attack_frames=prediction_service_failures=0
    failure=runtime_failure=None
    started=time.monotonic()
    env.render_state(name=f'scenario_{global_index:03d}',movie_path=str(out))
    for _ in range(args.steps):
        expect=forecaster is not None or args.group!='scenario_dreamer'
        try:
            observer.begin_frame(env,expect_joint_prediction=expect)
            if generator is not None:
                if (args.stop_file and args.stop_file.exists()):
                    raise RuntimeError('shared_planner_service_stop')
                generator.step(env,env.current_step)
                if generator._episode_planner_service_failures:
                    raise RuntimeError('planner_service_failure')
            if controller is not None:
                offset=0 if cache is None else env.current_step-cache.source_step
                active=set(np.flatnonzero(env.agent_active))
                if cache is not None and controller.remaining_optimization_seconds<=0:
                    # 预算耗尽后只复用已完成的同一未来，不新增外推或无优化替代轨迹。
                    if offset>=cache.positions_global.shape[1] or active!=set(cache.agent_ids):
                        raise NoSafeJointCandidate('strive_budget_exhausted_and_cached_future_unavailable')
                replan_due=(cache is None or active!=set(cache.agent_ids) or
                            (offset>=args.replan_steps and controller.remaining_optimization_seconds>0))
                if replan_due:
                    # 回调的调试输出不阻塞 IPC，但真实自车策略和优化预算保持不变。
                    with contextlib.redirect_stdout(io.StringIO()):
                        cache=controller.predict(env,policy) if active else empty_joint_prediction(env.current_step,int(round(6/env.dt)))
                    offset=0
                    current=env.data_dict['agent'][-1][cache.agent_ids]
                    safe=safe_joint_candidates(cache.positions_global[:,None],cache.yaws_global[:,None],
                        np.broadcast_to(np.eye(3),(len(current),3,3)),current[:,5:7])
                    if not safe[0]:
                        raise NoSafeJointCandidate('STRIVE optimized background OBB collision or invalid geometry')
                    plans+=int(bool(active))
                joint=slice_joint(cache,env.current_step,offset)
                env.inject_joint_trajectory(joint)
            else:
                joint=env.prepare_background_traffic()
            trace.append({'kind':'proposal_prediction','step':int(env.current_step),'joint':prediction_record(joint)})
            if joint is not None:
                trace.append({'kind':'trajectory_diagnostics','step':int(env.current_step),
                    'metrics':trajectory_diagnostics(env.data_dict['agent'][-1],joint,env.dt)})
            if generator is not None and env.attack_intent is not None:
                from policies.trajectory_projection import project_joint
                corrected=project_joint(joint,np.asarray(env.data_dict['agent'][-1],float),env.dt,
                    gate.dynamic_limits,env.cfg.sim.traffic_model.dynamics_projection,
                    getattr(env,'previously_attacked_agent_ids',()))
                env.inject_joint_trajectory(corrected)
                joint=corrected
                trace.append({'kind':'dynamics_projection','step':int(env.current_step),'audit':joint.metadata['dynamics_projection']})
                gate.validate_joint(env)
            if joint is not None:
                validate_execution_prefix(env.data_dict['agent'][-1],joint,env.get_scenario_frame().static_obstacles_global)
            # 修正并通过门控后的联合轨迹才是执行未来，原始候选单独保留。
            trace.append({'kind':'execution_prediction','step':int(env.current_step),'joint':prediction_record(joint)})
            if forecaster is not None:
                try:
                    forecast=forecaster.predict(env,policy)
                except Exception as error:
                    prediction_service_failures+=1
                    observer.finish_prediction(env,None,'ctrl_sim_rollout','forecast_service_failure:'+type(error).__name__)
                    raise RuntimeError('forecast_service_failure:'+type(error).__name__) from error
                observer.finish_prediction(env,forecast,'ctrl_sim_rollout')
                trace.append({'kind':'evaluation_forecast','step':int(env.current_step),
                    'joint':prediction_record(forecast),'validation_error':getattr(observer,'_prediction_error',None)})
            else:
                observer.finish_prediction(env,joint,'e1_native')
            if generator is not None:
                # 画像关/开均保持同一 RiskWeaver 正理论可达区域要求；不要求规避见证。
                if env.attack_intent is not None:
                    if not generator._reachability_query_pending:
                        generator.risk_metrics.begin_attack(env)
                    event=generator.evaluate_reachability(env)
                    if not event or not np.isfinite(event['difficulty']) or not event['dangerous_solvable']:
                        raise NoSafeJointCandidate('riskweaver_zero_or_invalid_theoretical_reachable_area')
                else:
                    generator.evaluate_reachability(env)
                generator.set_diffusion_trajectory(env.get_attack_target_prediction())
                trace.prediction(env,generator,time.monotonic()-started)
            env.render_state(name=f'scenario_{global_index:03d}',movie_path=str(out))
            anchors,refined=(generator.get_anchors_and_trajectory() if generator is not None else (None,None))
            before=env.pending_joint_trajectory
            obs,terminated,info=env.step(policy.act(obs),anchors,refined)
            terminal=merge_episode_info(terminal,info)
            execution.observe(env,info)
            # 历史 D 公式输入保留为 TCA，新增 OBB TTC 只作为独立公共指标。
            tca=execution.last_tca
            observer.finish_transition(info,tca)
            trace.state(env,info)
            if generator is not None:
                audit=trace.attack_execution(env,before)
                generator.evaluate_reaction(env,info)
                if generator.profile_pipeline is not None:
                    generator.profile_pipeline.executed(env,audit,info,trace)
            elif before is not None:
                attack_frames+=int(bool(before.metadata.get('attack_row',0)>0) if controller is not None
                                  else bool(before.metadata.get('adversarial_guidance_active')))
            count+=1
            if terminated:
                break
        except NoSafeJointCandidate as error:
            failure=str(error)
            trace.append({'kind':'generation_failure','step':int(env.current_step),'reason':failure})
            break
        except Exception as error:
            runtime_failure=type(error).__name__+': '+str(error)
            trace.append({'kind':'runtime_failure','step':int(env.current_step),'reason':runtime_failure})
            break
    env.render_state(name=f'scenario_{global_index:03d}',movie_path=str(out))
    video=generate_video(name=f'scenario_{global_index:03d}',output_dir=str(out),delete_images=False)
    danger=observer.result()
    # 只统计实际完成转移的帧，拒绝候选的预计算 EA 不进入结果。
    ea=[float(w['ea_mps2']) for w in danger['records'] if w['ea_mps2'] is not None]
    progress=normalized_route_progress(env.ego_state[:2],env.scenario_dict['route'])
    areas=[w['reachability'] for w in danger['records'] if w['reachability'] is not None]
    result={'group':args.group,'scenario_index':global_index,'local_scenario_index':index,
        'scenario_source':env.test_files[index],'seed':effective_seed,'executed_steps':count,
        'status':'runtime_failure' if runtime_failure else 'generation_failure' if failure else 'completed',
        'generation_failure':bool(failure),'generation_failure_reason':failure,
        'runtime_failure':bool(runtime_failure),'runtime_failure_reason':runtime_failure,
        'collision':bool(terminal['collision']) if count else None,
        'off_route':bool(terminal['off_route']) if count else None,
        'completed':bool(terminal['completed']) if count else None,'progress':progress,
        'max_evasive_acceleration_mps2':max(ea) if ea else None,
        'mean_evasive_acceleration_mps2':float(np.mean(ea)) if ea else None,
        'ea_valid_frame_count':len(ea),'scenario_danger_score':danger['danger'],
        'danger_valid':danger['valid'],'danger_valid_frame_count':danger['valid_frame_count'],
        'danger_total_frame_count':danger['total_frame_count'],'danger_coverage':danger['coverage'],
        'danger_windows':danger['records'],'danger_ttc_input_basis':'legacy_TCA_unchanged_definition',
        'reaction_margin':__import__('experiments.comparison.reaction_margin',fromlist=['summarize_attacks']).summarize_attacks(danger['records']),
        'reachability_source':'ctrl_sim_surrogate' if forecaster is not None and args.group!='scenario_dreamer' else
                              'ctrl_sim_rollout' if forecaster is not None else 'native',
        'prediction_service_failure_count':prediction_service_failures,
        'prediction_invalid_frame_count':sum(bool(w.get('prediction_error')) for w in danger['records']),
        'mean_reachable_area_m2':float(np.mean([w['dangerous_area_m2'] for w in areas])) if areas else None,
        'reachable_area_basis':'route_dynamics_reachable_terminal_cells',
        'attack_plan_count':plans,'attack_executed_frames':attack_frames,
        'obstacle_plan_count':0,'planner_service_failure_count':0,'feasibility':trace.summary(),
        'video_path':str(video),'runtime_seconds':time.monotonic()-started,**geometric,**execution.result()}
    result['autonomous_driving_ability_score']=ability_score(danger['danger'],bool(terminal['completed']) and not terminal['collision'],
        terminal['collision'],terminal['off_route'],progress,.5)
    if generator is not None:
        generator.finalize_episode_stats()
        generator.finalize_episode_difficulty()
        result.update(attack_plan_count=generator._episode_attack_plan_count,
            attack_executed_frames=trace.attack_executed_frames,
            obstacle_plan_count=generator._episode_obstacle_plan_count,
            planner_service_failure_count=generator._episode_planner_service_failures,
            llm_call_count=generator._llm_call_times)
        if generator.profile_pipeline is not None:
            result['ego_profile']=generator.profile_pipeline.finish_attempt(env,failure or runtime_failure)
    if generator is not None:
        from collections import Counter
        successes=Counter();attempts=Counter()
        for line in (out/'execution_trace.jsonl').read_text().splitlines():
            item=json.loads(line)
            if item.get('kind')=='llm_output':
                audit=item.get('planner_trace') or {}
                attempts.update(audit.get('attempted_models') or [])
                if audit.get('successful_model'): successes[audit['successful_model']]+=1
        result['llm_successful_model_counts']=dict(successes)
        result['llm_attempted_model_counts']=dict(attempts)
    result=sanitize_nonfinite(result)
    dump_json(out/'attempt_result.json',result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--group',choices=GROUPS,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--policy',choices=('idm','rl'),default='idm')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--steps',type=int,default=400)
    parser.add_argument('--model',default='qwen3.8-flash')
    parser.add_argument('--models',default=None)
    parser.add_argument('--evaluation-source',choices=('method_default','native','ctrl_sim_rollout'),default='method_default')
    parser.add_argument('--stop-file',type=Path)
    parser.add_argument('--strive-root',type=Path,default=Path('/home2/zhaoyx/STRIVE-riskweaver'))
    parser.add_argument('--strive-python',default='/home2/zhaoyx/STRIVE-riskweaver/.venv/bin/python')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/strive_official_20261008/traffic_model.pth')
    parser.add_argument('--init-iters',type=int,default=75)
    parser.add_argument('--adv-iters',type=int,default=300)
    parser.add_argument('--replan-steps',type=int,default=10)
    parser.add_argument('--timeout',type=float,default=1800)
    args=parser.parse_args()
    args.models=list(dict.fromkeys(args.models.split(','))) if args.models else [args.model]
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    manifest,files=validate_manifest(args.manifest)
    args.output_dir.mkdir(parents=True)
    dump_json(args.output_dir/'manifest.json',manifest)
    dump_json(args.output_dir/'protocol.json',protocol_snapshot(args.model,args.policy,args.seed,args.models))
    overrides=group_overrides(args.group,args.policy,args.seed,args.steps,args.model,args.models)
    overrides += [f'sim.movie_path={args.output_dir / "movies"}',f'sim.scenario_data_output_path={args.output_dir / "carla"}']
    with initialize(version_base=None,config_path='../../cfgs'):
        cfg=compose(config_name='config',overrides=overrides)
    (args.output_dir/'resolved_config.yaml').write_text(OmegaConf.to_yaml(cfg,resolve=True),encoding='utf-8')
    set_seed(args.seed)
    records=[]
    try:
        env=Simulator(cfg)
        if args.group in PROFILE_GROUPS and (not env.llm_planner.model_names or not set(env.llm_planner.model_names).issubset(args.models)):
            # 仅在冻结池内回退，不偷偷加入另一批未经记录的模型。
            dump_json(args.stop_file or args.output_dir/'STOP_LLM_SERVICE.json',{'reason':'frozen_pool_unavailable'})
            raise RuntimeError('frozen_pool_unavailable')
        env.test_files=list(map(str,files))
        policy=IDMPolicy(cfg,env) if args.policy=='idm' else __import__('policies.rl_policy',fromlist=['RLPolicy']).RLPolicy(cfg.sim)
        forecaster=None
        source=('ctrl_sim_rollout' if args.group in ('safe_sim','scenario_dreamer') else 'native') if args.evaluation_source=='method_default' else args.evaluation_source
        if source=='ctrl_sim_rollout':
            from experiments.comparison.ctrl_sim_forecast import CtrlSimJointForecaster
            forecaster=CtrlSimJointForecaster(env,cfg.sim.evaluation.reachability.horizon_seconds,args.seed,tracking='persistent')
        for index in range(len(files)):
            global_index=manifest.get('original_scenario_indices',list(range(len(files))))[index]
            try:
                record=run_episode(env,policy,index,global_index,args,forecaster)
            except Exception as error:
                # 重置/渲染等框架异常也必须占据预定索引，并让任务失败而非跳过。
                record={'group':args.group,'scenario_index':global_index,'scenario_source':str(files[index]),
                    'executed_steps':0,'collision':None,'runtime_failure':True,'generation_failure':False,
                    'status':'runtime_failure','runtime_failure_reason':type(error).__name__+': '+str(error)}
                dump_json(args.output_dir/'movies'/f'scenario_{global_index:03d}'/'framework_failure.json',record)
                dump_json(args.output_dir/'movies'/f'scenario_{global_index:03d}'/'attempt_result.json',record)
            records.append(record)
            report={'group':args.group,'protocol':protocol_snapshot(args.model,args.policy,args.seed,args.models),
                'aggregate':aggregate_attempts(records,len(files)),'episodes':records}
            dump_json(args.output_dir/'results.partial.json',report)
            if record.get('runtime_failure') or record.get('planner_service_failure_count'):
                if record.get('planner_service_failure_count') and args.stop_file:
                    dump_json(args.stop_file,{'reason':'planner_service_failure','group':args.group,'scenario_index':global_index})
                raise RuntimeError('scene_runtime_or_planner_service_failure; see persisted attempt')
        dump_json(args.output_dir/'results.json',report)
        dump_json(args.output_dir/'exit_status.json',{'exit_code':0,'finished_at':time.time(),'attempted':len(records)})
    except Exception as error:
        dump_json(args.output_dir/'results.json',{'group':args.group,'aggregate':aggregate_attempts(records,len(files)),'episodes':records})
        dump_json(args.output_dir/'exit_status.json',{'exit_code':1,'error_type':type(error).__name__,'finished_at':time.time()})
        raise


if __name__=='__main__':
    main()
