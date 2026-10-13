"""先五组各一场冒烟，验收通过才运行同一 100 场；按场景分片复用 GPU。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.comparison.e1_protocol import GROUPS,PROFILE_GROUPS,dump_json,select_100,shard_manifest,protocol_snapshot,planning_model_pool,group_overrides
from experiments.comparison.e1_metrics import aggregate_attempts

PYTHON='/home2/zhaoyx/miniconda3/envs/scenario-dreamer/bin/python'


def available_gpus(allowed,min_free_gb=14):
    """只调度显存和利用率满足条件的卡，不终止或修改其他研究任务。"""
    data=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu',
                                  '--format=csv,noheader,nounits'],text=True)
    result=[]
    for row in data.splitlines():
        gpu,free,util=map(int,row.split(','))
        if gpu in allowed and free>=min_free_gb*1024 and util<=10:
            result.append(gpu)
    return result


def smoke_acceptance(directory,group):
    """只验链路、执行和产物，不要求碰撞，不据结果更换冒烟场景。"""
    errors=[]
    result_path=directory/'results.json'
    if not result_path.exists():
        return ['missing_results']
    result=json.loads(result_path.read_text())
    episodes=result.get('episodes',[])
    if len(episodes)!=1:
        return ['wrong_smoke_attempt_count']
    row=episodes[0]
    if row.get('executed_steps',0)<=0:errors.append('no_actual_execution')
    if row.get('generation_failure'):errors.append('generation_failure')
    if row.get('runtime_failure'):errors.append('runtime_failure')
    if row.get('planner_service_failure_count'):errors.append('planner_service_failure')
    if row.get('prediction_service_failure_count'):errors.append('prediction_service_failure')
    if row.get('ea_valid_frame_count',0)<=0:errors.append('no_valid_EA')
    if group in PROFILE_GROUPS or group=='strive':
        if row.get('attack_plan_count',0)<=0:errors.append('no_attack_plan')
        if row.get('attack_executed_frames',0)<=0:errors.append('no_attack_execution')
    if group=='safe_sim' and row.get('attack_executed_frames',0)<=0:
        errors.append('official_guidance_not_executed')
    f=row.get('feasibility',{})
    if f.get('background_collision_frames') or f.get('background_static_collision_frames'):
        errors.append('background_collision')
    if row.get('obstacle_plan_count'):errors.append('unexpected_obstacle_plan')
    trace_path=Path(f.get('execution_trace_path',''))
    if not trace_path.is_file():
        errors.append('missing_trace')
    else:
        try:
            lines=[json.loads(s) for s in trace_path.read_text().splitlines()]
            states=[v for v in lines if v.get('kind')=='state']
            if len(states)!=row['executed_steps']+1:errors.append('state_trace_count_mismatch')
            if any(v.get('static_obstacles')!=[] for v in states):errors.append('unexpected_static_obstacles')
            if group=='safe_sim':
                predictions=[v.get('joint') or {} for v in lines if v.get('kind')=='execution_prediction']
                if not any('safe_sim_causecollision' in p.get('metadata',{}).get('guidance_functions',[]) for p in predictions):
                    errors.append('wrong_safe_sim_guidance')
        except Exception:
            errors.append('invalid_trace_json')
    video=Path(row.get('video_path') or '')
    if not video.is_file():
        errors.append('missing_video')
    else:
        try:
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries',
                'stream=nb_frames','-of','json',str(video)],text=True))
            frames=int(probe['streams'][0]['nb_frames'])
            if frames<row['executed_steps']+1:errors.append('video_missing_terminal_frame')
        except Exception:
            errors.append('video_probe_failure')
    return errors


def job_command(group,manifest,directory,args):
    """使用无密钥命令参数；提供方脚本从用户受限配置加载认证。"""
    command=[PYTHON,'experiments/comparison/run_e1_group.py','--group',group,'--manifest',str(manifest),
        '--output-dir',str(directory),'--policy',args.policy,'--seed',str(args.seed),
        '--model',args.model,'--models',','.join(args.models),'--stop-file',str(args.output_dir/'STOP_LLM_SERVICE.json')]
    return ['bash','scripts/run_with_llm_provider.sh','dashscope',*command] if group in PROFILE_GROUPS else command


def execute_phase(jobs,args,phase):
    """任务队列保留进度和 PID；仅本任务新分片失败时停止后续调度。"""
    pending=list(jobs);running={};finished=[];failed=False;failed_groups=set()
    while pending or running:
        for gpu,(process,job,log,start) in list(running.items()):
            code=process.poll()
            if code is None:
                continue
            log.close()
            errors=smoke_acceptance(job['directory'],job['group']) if phase=='smoke' and code==0 else []
            status={**{k:str(v) if isinstance(v,Path) else v for k,v in job.items()},
                'gpu':gpu,'exit_code':code,'started_at':start,'finished_at':time.time(),
                'acceptance_errors':errors,'passed':code==0 and not errors}
            dump_json(job['directory'].parent/(job['directory'].name+'_launcher_status.json'),status)
            finished.append(status);del running[gpu]
            if not status['passed']:
                # STRIVE局部故障只阻止该组后续分片，其它方法继续保留公平尝试。
                if job['group']=='strive':
                    failed_groups.add('strive')
                else:
                    failed=True
        if (args.output_dir/'STOP_LLM_SERVICE.json').exists():failed=True
        if failed and not running:
            break
        if not failed and pending:
            for gpu in available_gpus(args.gpus,args.min_free_gb):
                if gpu in running or not pending:continue
                eligible=next((i for i,j in enumerate(pending) if j['group'] not in failed_groups),None)
                if eligible is None:break
                job=pending.pop(eligible)
                env=os.environ.copy()
                env['CUDA_VISIBLE_DEVICES']=str(gpu)
                env['PYTHONPATH']='.:safe-sim:safe-sim/trajdata/src'
                env['PYTHONUNBUFFERED']='1'
                if job['group'] in PROFILE_GROUPS:
                    env['LLM_MODEL_NAME']=args.model;env['LLM_MODEL_NAMES']=','.join(args.models)
                job['directory'].parent.mkdir(parents=True,exist_ok=True)
                log_path=job['directory'].parent/(job['directory'].name+'.log')
                log=log_path.open('x',encoding='utf-8')
                command=job_command(job['group'],job['manifest'],job['directory'],args)
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                start=time.time()
                running[gpu]=(process,job,log,start)
                dump_json(job['directory'].parent/(job['directory'].name+'_pid.json'),{'pid':process.pid,'gpu':gpu,'command':command})
        dump_json(args.output_dir/'launcher_result.partial.json',{'phase':phase,'finished':finished,
            'running':[{'group':j['group'],'directory':str(j['directory']),'pid':p.pid,'gpu':g} for g,(p,j,l,s) in running.items()],
            'pending_count':len(pending),'failed_groups':sorted(failed_groups),'failed':failed,'updated_at':time.time()})
        if not running and pending and all(j['group'] in failed_groups for j in pending):
            break
        if pending or running:
            time.sleep(5 if running else 30)
    return {'phase':phase,'jobs':finished,'unstarted_jobs':[str(j['directory']) for j in pending],
            'failed_groups':sorted(failed_groups),'global_failure':failed,
            'passed':not failed and not failed_groups and not pending and len(finished)==len(jobs)}


def formal_groups_after_smoke(smoke):
    """STRIVE冒烟故障也仅隔离该组；共享服务或其它方法故障仍阻断正式阶段。"""
    if smoke['passed']:
        return list(GROUPS)
    if (smoke.get('failed_groups') == ['strive'] and not smoke.get('global_failure', True)
            and all(job['passed'] for job in smoke['jobs'] if job['group'] != 'strive')):
        return [group for group in GROUPS if group != 'strive']
    return []


def merge_formal(jobs,manifest,args):
    """合并时严格验证全局编号、场景来源及五组完整性；不丢弃失败尝试。"""
    results={};expected=set(manifest['original_scenario_indices'])
    for group in GROUPS:
        rows=[]
        for job in jobs:
            if job['group']!=group:continue
            p=job['directory']/'results.json'
            if p.exists():rows.extend(json.loads(p.read_text()).get('episodes',[]))
        ids=[r['scenario_index'] for r in rows]
        if len(ids)!=len(set(ids)):
            raise ValueError('duplicate global scene indices')
        rows.sort(key=lambda r:r['scenario_index'])
        by_id=dict(zip(manifest['original_scenario_indices'],manifest['scenario_files']))
        if any(r['scenario_index'] not in expected or r['scenario_source']!=by_id[r['scenario_index']] for r in rows):
            raise ValueError('global scene source mismatch')
        from collections import Counter
        models=Counter()
        for r in rows: models.update(r.get('llm_successful_model_counts',{}))
        results[group]={'aggregate':aggregate_attempts(rows,100),'complete':set(ids)==expected,'episodes':rows,'successful_model_distribution':dict(models)}
    dump_json(args.output_dir/'comparison_results.json',results)
    from experiments.comparison.e1_reporting import write_tables
    write_tables(results,args.output_dir)
    return all(r['complete'] for r in results.values())


def reuse_baseline_smoke(previous,args,source):
    """复用已完成三基线；复核场景、实际配置、组件哈希和视频，而非空绕过。"""
    from hydra import initialize,compose
    from omegaconf import OmegaConf
    previous=Path(previous).resolve()
    final=json.loads((previous/'launcher_result.json').read_text())
    if not final['smoke']['passed']:
        raise ValueError('previous_smoke_failed')
    frozen=json.loads((previous/'smoke_one_scene.json').read_text())
    if frozen['scenario_files']!=[source['scenario_files'][0]] or frozen['scenario_sha256']!=[source['scenario_sha256'][0]]:
        raise ValueError('reused_smoke_scene_changed')
    hashes=json.loads((previous/'launch.json').read_text())['source_hashes']
    for f in ['policies/safe_sim_official_guidance.py','policies/diffusion_model_wrapper.py',
              'policies/strive_adapter.py','simulator.py','experiments/comparison/e1_metrics.py']:
        if hashes[f]!=hashlib.sha256((ROOT/f).read_bytes()).hexdigest():
            raise ValueError('reused_smoke_component_changed:'+f)
    validated=[]
    with initialize(version_base=None,config_path='../../cfgs'):
        for group in ('scenario_dreamer','safe_sim','strive'):
            directory=previous/'smoke'/group
            overrides=group_overrides(group,args.policy,args.seed,400,args.model,args.models)
            overrides+=[f'sim.movie_path={directory / "movies"}',f'sim.scenario_data_output_path={directory / "carla"}']
            cfg=compose(config_name='config',overrides=overrides)
            old=OmegaConf.load(directory/'resolved_config.yaml')
            if OmegaConf.to_container(cfg,resolve=True)!=OmegaConf.to_container(old,resolve=True):
                raise ValueError('reused_smoke_configuration_changed:'+group)
            exit_status=json.loads((directory/'exit_status.json').read_text())
            errors=smoke_acceptance(directory,group)
            if exit_status['exit_code']!=0 or errors:
                raise ValueError('reused_smoke_artifacts_invalid:'+group+':'+str(errors))
            validated.append({'group':group,'directory':str(directory),'passed':True})
    return {'phase':'reused_baseline_smoke','jobs':validated,'passed':True,
            'riskweaver_extra_model_smoke_waived_by_user':True}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--policy',choices=('idm','rl'),default='idm')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--model',default='qwen3.8-flash')
    parser.add_argument('--gpus',default='0,1,2,3')
    parser.add_argument('--min-free-gb',type=float,default=14)
    parser.add_argument('--shard-size',type=int,default=10)
    parser.add_argument('--smoke-only',action='store_true')
    parser.add_argument('--validated-smoke-dir',type=Path)
    parser.add_argument('--smoke-groups',default=','.join(GROUPS))
    args=parser.parse_args()
    from policies.llm_adversarial_planner import LLMAdversarialPlanner
    disabled=LLMAdversarialPlanner._exhausted_models_from_source()
    args.models,skipped=planning_model_pool([n for n in LLMAdversarialPlanner.FREE_MODEL_POOL if n not in disabled],args.model)
    args.model=args.models[0]
    args.gpus=[int(x) for x in args.gpus.split(',')]
    smoke_groups=args.smoke_groups.split(',')
    if not smoke_groups or len(set(smoke_groups))!=len(smoke_groups) or any(g not in GROUPS for g in smoke_groups):
        raise ValueError('invalid smoke group selection')
    if args.shard_size<1 or not args.gpus:
        raise ValueError('invalid scheduler configuration')
    # 在创建实验目录前拒绝场景/权重变更，不下载或替换已有权重。
    from experiments.comparison.run_passive_baseline import validate_manifest
    source,_=validate_manifest(args.manifest)
    checkpoint=ROOT/'checkpoints/strive_official_20261008/traffic_model.pth'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!='f977360762b7c2f8d66745fb781b1d1a66c4963a5d48a47bc2ca7646861088cf':
        raise ValueError('STRIVE checkpoint hash changed')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/'launcher.pid').write_text(str(os.getpid()),encoding='utf-8')
    dump_json(args.output_dir/'protocol.json',protocol_snapshot(args.model,args.policy,args.seed,args.models))
    dump_json(args.output_dir/'model_pool.json',{'models':args.models,'skipped_specialized_models':skipped,'preferred':args.model,'frozen_at':time.time()})
    selected=select_100(source,args.seed)
    manifest100=args.output_dir/'frozen_100_scenes.json'
    dump_json(manifest100,selected)
    smoke_manifest=args.output_dir/'smoke_one_scene.json'
    dump_json(smoke_manifest,shard_manifest(source,[0]))
    source_files=['experiments/comparison/e1_metrics.py','experiments/comparison/e1_protocol.py',
        'experiments/comparison/run_e1_group.py','experiments/comparison/launch_e1_five_arm.py',
        'experiments/comparison/e1_reporting.py',
        'policies/safe_sim_official_guidance.py','policies/diffusion_model_wrapper.py','policies/llm_adversarial_planner.py',
        'policies/ego_profile.py','policies/strive_adapter.py','scenario_generator.py','simulator.py']
    # 记录评估与预算组件，避免新旧预测来源和面积协议无法追溯。
    source_files += ['experiments/comparison/formal_metrics.py','experiments/comparison/ctrl_sim_forecast.py',
        'experiments/comparison/reaction_margin.py','experiments/comparison/optimization_budget.py',
        'experiments/comparison/strive_official_worker.py','experiments/comparison/commonroad_worker.py',
        'policies/risk_metrics.py','policies/commonroad_reachability.py','cfgs/sim/base.yaml',
        'requirements-commonroad-reach.lock.txt']
    dump_json(args.output_dir/'launch.json',{'argv':sys.argv,'started_at':time.time(),
        'manifest_sha256':hashlib.sha256(manifest100.read_bytes()).hexdigest(),
        'source_manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        'source_hashes':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in source_files}})
    import shutil
    for f in source_files:
        snapshot=args.output_dir/'source_snapshot'/f
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/f,snapshot)
    smoke_jobs=[{'group':g,'manifest':smoke_manifest,'directory':args.output_dir/'smoke'/g} for g in smoke_groups]
    smoke=reuse_baseline_smoke(args.validated_smoke_dir,args,source) if args.validated_smoke_dir else execute_phase(smoke_jobs,args,'smoke')
    dump_json(args.output_dir/'smoke_acceptance.json',smoke)
    formal_groups=formal_groups_after_smoke(smoke)
    if not formal_groups or args.smoke_only:
        dump_json(args.output_dir/'launcher_result.json',{'smoke':smoke,'formal_started':False,'finished_at':time.time()})
        return 0 if smoke['passed'] else 71
    jobs=[]
    for shard,start in enumerate(range(0,100,args.shard_size)):
        m=args.output_dir/'manifests'/f'shard_{shard:03d}.json'
        dump_json(m,shard_manifest(selected,list(range(start,min(start+args.shard_size,100)))))
        for group in formal_groups:
            jobs.append({'group':group,'manifest':m,'directory':args.output_dir/'formal'/group/f'shard_{shard:03d}'})
    formal=execute_phase(jobs,args,'formal')
    complete=merge_formal(jobs,selected,args)
    dump_json(args.output_dir/'launcher_result.json',{'smoke':smoke,'formal':formal,'formal_started':True,
        'all_groups_complete':complete,'finished_at':time.time()})
    return 0 if formal['passed'] and complete else 72


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        # 启动前配置/资源异常同样落盘，后台监控可辨识终态而不丢失日志。
        if '--output-dir' in sys.argv:
            directory=Path(sys.argv[sys.argv.index('--output-dir')+1])
            if directory.exists():
                dump_json(directory/'launcher_result.json',{'launcher_failure':True,
                    'formal_started':(directory/'formal').exists(),'error_type':type(error).__name__,
                    'reason':str(error),'finished_at':time.time()})
        raise
