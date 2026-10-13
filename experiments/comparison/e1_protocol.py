"""冻结五组 E1 配置、分层清单和计算预算，避免组间静默切换。"""
import hashlib
import json
from pathlib import Path
import numpy as np

GROUPS = ('riskweaver_profile_off','riskweaver_profile_on','scenario_dreamer','safe_sim','strive')
PROFILE_GROUPS = GROUPS[:2]
STRIVE_FIDELITY = 'official_optimizer_Waymo_lane_proxy_CV_initialization_0.5_to_0.1s_interpolation'
SAFE_SIM_FIDELITY = 'official_losses_external_CV_ego_Waymo_adapter_full_diffusion_no_partial_proposals'


def dump_json(path, value):
    """原子替换本任务进度文件；从不改写历史任务目录。"""
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def select_100(manifest, seed=42):
    """固定来源配额 30/22/48，不读取任何方法的性能结果。"""
    files, digests = manifest['scenario_files'],manifest['scenario_sha256']
    if len(files)!=250 or len(digests)!=250 or len(set(digests))!=250:
        raise ValueError('requires frozen 250 unique scenes')
    pools={}
    for index,path in enumerate(files):
        pools.setdefault(str(Path(path).parent),[]).append(index)
    counts=sorted(len(v) for v in pools.values())
    if counts!=[55,75,120]:
        raise ValueError('frozen source composition changed')
    rng=np.random.default_rng(seed)
    quotas={75:30,55:22,120:48}
    selected=[]
    for source,pool in sorted(pools.items()):
        selected.extend(map(int,rng.choice(pool,quotas[len(pool)],replace=False)))
    selected.sort()
    result=dict(manifest)
    result.update(scenario_count=100,scenario_files=[files[i] for i in selected],
        scenario_sha256=[digests[i] for i in selected],original_scenario_indices=selected,
        selection_seed=seed,selection_basis='source_stratified_30_22_48_outcome_independent',
        preliminary_development_comparison=True,
        development_exposure_note='Existing scenes have prior development/test exposure; this is not a held-out generalization claim')
    return result


def shard_manifest(manifest, indices):
    """场景并行保留冻结全局编号，分片不影响随机种子或画像初始化。"""
    result=dict(manifest)
    result.update(scenario_count=len(indices),scenario_files=[manifest['scenario_files'][i] for i in indices],
        scenario_sha256=[manifest['scenario_sha256'][i] for i in indices],
        original_scenario_indices=[manifest.get('original_scenario_indices',list(range(len(manifest['scenario_files']))))[i] for i in indices])
    return result


def group_overrides(group, policy='idm',seed=42,steps=400,model='qwen3.8-flash',models=None):
    """画像两组共享求解器、预算、连续修正和候选扩展设置；只切换画像链路。"""
    if group not in GROUPS or policy not in ('idm','rl') or not 1<=steps<=400:
        raise ValueError('unsupported E1 configuration')
    shared=[f'sim.seed={seed}',f'sim.policy={policy}',f'sim.steps={steps}',
        'sim.visualize=true','sim.lightweight=true','sim.llm.attack_mode=trajectory_only',
        'sim.llm.obstacles.enabled=false','sim.llm.multiagent.mode=single',
        'sim.traffic_model.iterative_adversarial.escalation_enabled=false',
        'sim.traffic_model.iterative_adversarial.full_method_enabled=false',
        'sim.evaluation.difficulty_control.mode=off','sim.evaluation.difficulty_control.max_replays=0',
        'sim.traffic_model.multi_gpu_devices=[]','sim.traffic_model.mixed_precision=false',
        'sim.traffic_model.diffusion_agent_limit=0',
        'sim.traffic_model.guidance.params.inner_lr=0.2',
        'sim.traffic_model.guidance.params.inner_beta=0.5',
        'sim.traffic_model.guidance.params.n_guide_steps=2',
        'sim.ego_profile.history_paths=[]','sim.ego_profile.candidate_expansion_enabled=false',
        'sim.ego_profile.rear_approach_accelerations_mps2=[]']
    shared += [f'sim.traffic_model.iterative_adversarial.profile_enabled={str(group=="riskweaver_profile_on").lower()}']
    if group in PROFILE_GROUPS:
        return shared+['sim.traffic_model.backend=safe_sim_diffusion','sim.traffic_model.guidance.mode=llm_joint',
            'sim.traffic_model.dynamics_projection.enabled=true',
            'sim.traffic_model.anchor_guidance.anchor_drive_enabled=true',
            'sim.llm.provider=dashscope',f'sim.llm.model_name={model}',f"sim.llm.model_names=[{','.join(models or [model])}]",
            'sim.llm.max_planning_calls=4','+sim.llm.min_planning_step=0','+sim.llm.planning_interval_frames=10']
    if group=='safe_sim':
        return shared+['sim.traffic_model.backend=safe_sim_diffusion','sim.traffic_model.guidance.mode=manual',
            'sim.traffic_model.guidance.manual.functions=[route,safe_sim_collision,safe_sim_ttc,safe_sim_causecollision]',
            'sim.traffic_model.guidance.manual.weights=[0.5,0.5,0.5,0.5]',
            'sim.traffic_model.dynamics_projection.enabled=false','sim.traffic_model.anchor_guidance.anchor_drive_enabled=false']
    if group=='scenario_dreamer':
        return shared+['sim.traffic_model.backend=ctrl_sim','sim.traffic_model.guidance.mode=unguided',
            'sim.behaviour_model.tilt=-10','sim.behaviour_model.run_name=ctrl_sim_waymo_1M_steps']
    return shared+['sim.traffic_model.backend=external_joint','sim.traffic_model.guidance.mode=unguided']


def protocol_snapshot(model='qwen3.8-flash',policy='idm',seed=42,models=None):
    """区分方法适配和公共度量；不把单场冒烟当作原论文完整复现。"""
    return {'groups':list(GROUPS),'scene_count_per_group':100,'planned_attempts':500,
        'policy':policy,'seed':seed,'max_frames':400,'natural_termination':True,
        'llm_model':model,'llm_max_calls_per_scene':4,'llm_fixed_single_model':len(models or [model])==1,'llm_model_pool':list(models or [model]),
        'llm_quota_policy':'free_only_disable_404_or_exhausted_then_pool_fallback_stop_if_all_unavailable',
        'strive':{'init_iters':75,'adv_iters':300,'solution_iters':300,'replan_steps':10,'optimization_time_budget_seconds':300,'fidelity':STRIVE_FIDELITY},
        'safe_sim':{'losses':['route','collision','ttc','causecollision'],'weights':[.5]*4,'fidelity':SAFE_SIM_FIDELITY,
            'fallback':'unchanged guidance.mode=default, separate new experiment only'},
        'profile_initialization':'new profile and knowledge per scene; all observed frames within that scene; no cross-shard/cross-test history',
        'profile_comparison_scope':'full_profile_conditioned_candidate_ranking_pipeline_vs_free_discrete_planning',
        'primary_metrics':['collision','OBB_TTC','EA','relative_collision_speed','off_route','progress'],
        'secondary_prediction_metrics':'D and reachable area disclose native vs CtRL surrogate; no pooled cross-source D ranking',
        'evaluation_sources':{'scenario_dreamer':'ctrl_sim_rollout_native_model','safe_sim':'ctrl_sim_surrogate','riskweaver':'native_diffusion','strive':'native_adapted'},
        'reachability_backend_default':'route_grid',
        'danger_definition':'unchanged weighted EA+reachability-valid windows; legacy TCA input retained and explicitly labeled',
        'constraints':'common executed/background OBB audit; native method regularizers and RiskWeaver dynamics/positive-area gates disclosed',
        'monitor_interval_seconds':3600}


def planning_model_pool(names, preferred='qwen3.8-flash'):
    """保留用户列表顺序，剔除明显不适用文本JSON策略规划的专用接口。"""
    skipped = {}
    pool = []
    for name in dict.fromkeys(names):
        if any(token in name for token in ('ocr','realtime','omni','-mt-','math-','embedding','-tts','-asr')):
            skipped[name] = 'specialized_non_planner_interface'
        else:
            pool.append(name)
    if preferred in pool:
        pool.remove(preferred); pool.insert(0, preferred)
    if not pool:
        raise ValueError('no_text_planning_candidates')
    return pool, skipped
