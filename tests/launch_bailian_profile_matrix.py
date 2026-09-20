"""固定场景与模型预算的四组对比；冒烟未通过时不启动正式实验。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
BASE=ROOT/'experiments/riskweaver_bailian_profile_20260916'
SOURCE=['scenario_generator.py','run_simulation.py','simulator.py','cfgs/sim/base.yaml',
        'policies/llm_adversarial_planner.py','policies/profile_planner.py','policies/profile_obstacles.py',
        'policies/ego_profile.py','policies/joint_safety.py','policies/risk_metrics.py',
        'policies/scenario_guidance.py','policies/diffusion_model_wrapper.py',
        'safe-sim/tbsim/models/RasterizedDiffusionModel.py','safe-sim/tbsim/models/diffusion.py']
ARMS=[('trajectory_plain','trajectory_only',False),('trajectory_profile','trajectory_only',True),
      ('joint_plain','joint',False),('joint_profile','joint',True)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def freeze():
    source=ROOT/'experiments/riskweaver_target_formal_20260913/manifest.json'
    previous=json.loads(source.read_text(encoding='utf-8'))
    items=[]
    for path,expected in zip(previous['scenario_files'],previous['scenario_sha256']):
        if digest(path)!=expected:
            raise RuntimeError('历史冻结场景哈希变化')
        items.append(dict(path=path,sha256=expected))
    if len(items)!=20:
        raise RuntimeError('需要完整的 20 场景冻结清单')
    manifest=BASE/'matrix_manifest.json'
    if not manifest.exists():
        write(manifest,dict(seed=42,scene_files=items,model='qwen3.5-plus',provider='dashscope',
                            arms=[dict(name=n,attack_mode=m,profile=p) for n,m,p in ARMS],
                            smoke_scene_index=1,smoke_steps=80,formal_steps=400,
                            smoke_max_planning_calls=6,formal_max_planning_calls=10,
                            difficulty_mode='off',optimizer='fixed_by_existing_config',
                            free_quota_only_user_attested=True,
                            interpretation='whole-pipeline ablation; profile branch also changes candidate selection and safety gate'))
    else:
        old=json.loads(manifest.read_text(encoding='utf-8'))
        if [item['sha256'] for item in old['scene_files']] != [item['sha256'] for item in items]:
            raise RuntimeError('本轮冻结清单变化')
    return json.loads(manifest.read_text(encoding='utf-8'))


def inputs(manifest,phase):
    index=manifest['smoke_scene_index']
    chosen=manifest['scene_files'][index:index+1] if phase=='smoke' else manifest['scene_files']
    path=BASE/f'{phase}_scenario_manifest.json'
    if not path.exists():
        write(path,dict(scenario_files=[row['path'] for row in chosen],
                        scenario_sha256=[row['sha256'] for row in chosen]))
    elif json.loads(path.read_text())['scenario_sha256']!=[row['sha256'] for row in chosen]:
        raise RuntimeError('场景输入哈希变化')
    return path


def assess(directory,attack_mode,profile,phase):
    result_paths=sorted((directory/'movies').glob('scenario_*/attempt_result.json'))
    expected=1 if phase=='smoke' else 20
    results=[]
    for path in result_paths:
        row=json.loads(path.read_text(encoding='utf-8'))
        trace=Path(row['feasibility']['execution_trace_path'])
        video=Path(row['video_path'])
        if not trace.is_file() or not video.is_file() or video.stat().st_size<=0:
            raise RuntimeError('缺少视频或执行追踪：'+str(path))
        if attack_mode=='trajectory_only' and row['obstacle_plan_count']!=0:
            raise RuntimeError('纯轨迹组出现障碍物计划')
        feasible=row['feasibility']
        if feasible['background_collision_frames'] or feasible['background_static_collision_frames']:
            raise RuntimeError('背景车辆实际碰撞：'+str(path))
        if profile:
            if not (path.parent/'ego_profile.json').is_file() or not (path.parent/'strategy_returns.jsonl').is_file():
                raise RuntimeError('画像或策略收益表缺失')
        results.append(row)
    if len(results)!=expected:
        raise RuntimeError(f'{phase}仅有 {len(results)}/{expected} 条完整场景结果')
    summary=dict(scenarios=len(results),attack_plans=sum(x['attack_plan_count'] for x in results),
                 executed_attack_frames=sum(x['attack_executed_frames'] for x in results),
                 obstacle_plans=sum(x['obstacle_plan_count'] for x in results),
                 valid_danger=sum(bool(x['danger_valid']) for x in results),
                 generation_failures=sum(bool(x.get('generation_failure')) for x in results),
                 planner_service_failures=sum(x['planner_service_failure_count'] for x in results),
                 collisions=sum(bool(x['collision']) for x in results),
                 videos=[x['video_path'] for x in results])
    if phase=='smoke':
        if summary['planner_service_failures'] or summary['generation_failures']:
            raise RuntimeError('冒烟出现规划服务或生成失败')
        if not summary['valid_danger'] or not summary['attack_plans'] or not summary['executed_attack_frames']:
            raise RuntimeError('冒烟缺少有效 D 或实际执行攻击')
        if attack_mode=='joint' and not summary['obstacle_plans']:
            raise RuntimeError('联合模式未实际创建障碍物')
    return summary


def run_arm(name,mode,profile,phase,scenario_manifest,source_hashes):
    directory=BASE/phase/name
    directory.mkdir(parents=True,exist_ok=False)
    instructions=BASE/'common_instructions.txt'
    env=dict(os.environ,PROJECT_ROOT=str(ROOT),SCRATCH_ROOT=str(ROOT),DATASET_ROOT=str(ROOT/'metadata'),
             PYTHONPATH=f'{ROOT}:{ROOT}/safe-sim:{ROOT}/safe-sim/trajdata/src',MPLBACKEND='Agg',
             PYTHONUNBUFFERED='1',HYDRA_FULL_ERROR='1',LLM_MODEL_NAME='qwen3.5-plus')
    overrides=['sim.seed=42','+sim.continue_after_off_route=true',
               f'sim.evaluation.max_scenarios={1 if phase=="smoke" else 20}',
               f'sim.steps={80 if phase=="smoke" else 400}',
               f'+sim.scenario_manifest={scenario_manifest}',f'+sim.attack_request_file={instructions}',
               'sim.llm.provider=dashscope','sim.llm.model_name=qwen3.5-plus',
               'sim.llm.model_names=[qwen3.5-plus]',f'sim.llm.max_planning_calls={6 if phase=="smoke" else 10}',
               f'sim.llm.attack_mode={mode}',f'sim.llm.obstacles.enabled={str(mode=="joint").lower()}',
               f'sim.traffic_model.iterative_adversarial.profile_enabled={str(profile).lower()}',
               'sim.traffic_model.iterative_adversarial.escalation_enabled=false',
               'sim.traffic_model.iterative_adversarial.full_method_enabled=false',
               'sim.traffic_model.guidance.mode=llm_joint',
               'sim.evaluation.difficulty_control.mode=off',
               'sim.evaluation.difficulty_control.smoke_acceptance=false',
               'sim.visualize=true','sim.lightweight=true',
               f'sim.movie_path={directory}/movies',f'sim.scenario_data_output_path={directory}/carla',
               f'hydra.run.dir={directory}/hydra']
    command=['bash','scripts/run_with_llm_provider.sh','dashscope',sys.executable,'run_simulation.py',*overrides]
    write(directory/'launch.json',dict(command=command,started_at=time.time(),source_hashes=source_hashes,
                                       free_quota_guard='user confirmed platform free-quota stop enabled'))
    with (directory/'resolved_config.yaml').open('w',encoding='utf-8') as handle:
        subprocess.run(command+['--cfg','job','--resolve'],cwd=ROOT,env=env,stdout=handle,check=True)
    with (directory/'run.log').open('w',encoding='utf-8') as handle:
        code=subprocess.call(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
    write(directory/'exit_status.json',dict(exit_code=code,finished_at=time.time()))
    if code:
        raise RuntimeError(f'{name} 仿真退出码 {code}，检查 run.log')
    summary=assess(directory,mode,profile,phase)
    write(directory/'summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=('smoke','formal'),required=True)
    args=parser.parse_args()
    BASE.mkdir(parents=True,exist_ok=True)
    manifest=freeze()
    instructions=BASE/'common_instructions.txt'
    if not instructions.exists():
        instructions.write_text('Generate a physically executable, avoidable interaction that probes the ego driving strategy. Follow the configured attack mode and all safety validators.\n',encoding='utf-8')
    source_hashes={name:digest(ROOT/name) for name in SOURCE}
    source_file=BASE/'source_hashes.json'
    if not source_file.exists():
        write(source_file,source_hashes)
    elif json.loads(source_file.read_text())!=source_hashes:
        raise RuntimeError('实验源码已变更；须冻结新的实验批次')
    if args.phase=='formal':
        for name,_,_ in ARMS:
            path=BASE/'smoke'/name/'summary.json'
            if not path.is_file():
                raise RuntimeError('四组冒烟尚未全部通过')
    path=inputs(manifest,args.phase)
    summaries={}
    for name,mode,profile in ARMS:
        summaries[name]=run_arm(name,mode,profile,args.phase,path,source_hashes)
        write(BASE/args.phase/'matrix_summary.json',summaries)
        print(json.dumps({name:summaries[name]},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
