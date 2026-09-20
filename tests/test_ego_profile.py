"""画像完整链路的 CPU 回归：统计、候选、封闭排序、实际执行和失败记账。"""
import ast
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from policies.ego_profile import (OnlineStatistic,EgoProfile,AttackKnowledge,CandidateBuilder,
                                  ProfileAttackPipeline,scene_conditions)
from policies.profile_planner import rank_profile_candidates
from policies.joint_safety import NoSafeJointCandidate


def environment():
    route = np.column_stack((np.arange(-50,151,2),np.zeros(101)))
    env = NS(ego_state=np.array([0.,0.,10.,0.,0.,4.,1.8]),
             data_dict={'agent':[np.array([[18.,0.,8.,0.,0.,4.,1.8]])]},
             agent_active=np.array([True]),dt=.1,current_step=0,current_scene_id='test',
             scenario_dict={'route':route,'lanes':np.array([route])},attack_intent=None)
    env.get_static_obstacles = lambda: []
    env.get_state_for_planning = lambda:dict(ego_state=env.ego_state[:5].tolist(),route=route.tolist(),
        agents=[dict(id=0,state=env.data_dict['agent'][-1][0,:5].tolist(),type='vehicle',history=[])],
        static_obstacles=[],history_order='oldest_first',current_step=env.current_step)
    env.clear_attack_intent = lambda: setattr(env,'attack_intent',None)
    env.set_attack_intent = lambda target_id,anchors,strategy: setattr(env,'attack_intent',dict(target_id=target_id,anchors=anchors,strategy=strategy,source_step=env.current_step))
    return env


def planner():
    # 运行生产语义校验和坐标变换；仅传输用替身，禁止真实 LLM 调用。
    tree = ast.parse((Path(__file__).resolve().parents[1]/'policies/llm_adversarial_planner.py').read_text())
    nodes = [n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in {'_normalize_env_state','_validate_attack_plan'}]
    for node in nodes:
        node.decorator_list = []
    scope = dict(np=np,json=json)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'production_planner','exec'),scope)
    p = NS(provider='deepseek',calls=0)
    p._normalize_env_state = lambda state:scope['_normalize_env_state'](p,state)
    p._validate_attack_plan = scope['_validate_attack_plan']
    def request(prompt,scene_image=None):
        p.calls += 1
        payload = json.loads(prompt.split('\n',1)[1])
        return '',dict(request_id=payload['request_id'],ranking=[c['candidate_id'] for c in payload['candidates']],no_attack_reason='')
    p._request_attack_plan = request
    return p


def pipeline():
    return ProfileAttackPipeline(NS(policy='idm',ego_profile=NS(policy_id='idm-test')))


def test_quantiles_ewma_and_uncertainty():
    stats = OnlineStatistic(alpha=.5,block_size=10)
    for value in range(200):
        stats.add(value)
    snapshot = stats.snapshot()
    assert snapshot['n']==200 and snapshot['quantiles']['q50']==99.5
    assert snapshot['ewma']>190 and snapshot['block_mean_ci95'][0]<99.5<snapshot['block_mean_ci95'][1]
    assert OnlineStatistic().snapshot()['block_mean_ci95'] is None
    stats.add(float('nan'))
    assert stats.n==200


def test_statistic_does_not_change_global_rng():
    np.random.seed(22)
    expected = np.random.random(3)
    np.random.seed(22)
    stat = OnlineStatistic(capacity=32,block_size=2)
    for x in range(300):
        stat.add(x)
    stat.snapshot()
    np.testing.assert_equal(expected,np.random.random(3))


def test_profile_all_frames_and_cross_episode():
    env,profile = environment(),EgoProfile(NS())
    profile.reset_episode()
    for step in range(40):
        env.current_step=step
        profile.observe(env,attack=step<5)
        profile.observe(env,attack=step<5)  # 同帧不得重复计数。
    condition = scene_conditions(env)
    snap = profile.snapshot(condition)
    assert (snap['frames'],snap['attack_frames'],snap['nonattack_frames'])==(40,5,35)
    assert snap['conditioned_metrics']['speed_mps']['n']==40
    profile.reset_episode()
    env.current_step=0
    env.ego_state[2]=20
    profile.observe(env)
    snap = profile.snapshot(scene_conditions(env))
    assert snap['frames']==41 and snap['metrics']['acceleration_mps2']['n']==39
    assert snap['conditioned_metrics']['speed_mps']['n']==1


@pytest.mark.parametrize('kwargs',[dict(alpha=0),dict(capacity=1),dict(block_size=1)])
def test_invalid_stat_settings(kwargs):
    with pytest.raises(ValueError):
        OnlineStatistic(**kwargs)


def test_candidate_has_required_explanations():
    pipe,env = pipeline(),environment()
    context = pipe.context(env,.5)
    assert context['candidates']
    candidate = context['candidates'][0]
    assert {'estimated_risk','avoidability','background_safety_margin_m','profile_match','vulnerability'}<=set(candidate)
    assert candidate['avoidability']==dict(witness=None,
        requirement='theoretical_drivable_area_must_remain_positive')
    assert 1<=candidate['estimated_ttc_s']<=5
    assert candidate['profile_evidence_frames']==0


def test_adjacent_lane_produces_cut_in_candidate():
    env=environment()
    adjacent=env.scenario_dict['lanes'][0].copy()
    adjacent[:,1]=3.6
    env.scenario_dict['lanes']=np.stack([env.scenario_dict['lanes'][0],adjacent])
    env.data_dict['agent'][-1][0,:2]=[10.,3.6]
    candidates=pipeline().context(env,.5)['candidates']
    assert any(c['attack_family']=='cut_in' for c in candidates)


def test_candidate_rejects_background_overlap_and_static():
    env = environment()
    env.data_dict['agent'][-1] = np.repeat(env.data_dict['agent'][-1],2,axis=0)
    env.agent_active = np.array([True,True])
    context = pipeline().context(env,.5)
    assert not context['candidates'] and context['candidate_rejections'].get('background_margin')
    env = environment()
    env.get_static_obstacles = lambda:[dict(center=[18.,0.],yaw=0.,length=4.,width=2.,id='wall')]
    assert not pipeline().context(env,.5)['candidates']


def test_missing_road_and_invalid_state_fail_closed():
    env = environment()
    env.scenario_dict['lanes']=[]
    assert pipeline().context(env,.5)['candidate_rejections']=={'missing_road_geometry':1}
    env = environment()
    env.data_dict['agent'][-1][0,2] = np.nan
    assert pipeline().context(env,.5)['candidate_rejections']=={'invalid_vehicle_geometry':1}


def test_candidate_does_not_require_ego_avoidance_witness():
    env = environment()
    candidates = pipeline().context(env,.5)['candidates']
    assert candidates and all(c['avoidability']['witness'] is None for c in candidates)


def test_attacker_selection_has_no_minimum_current_speed():
    env = environment()
    env.data_dict['agent'][-1][0,2:4] = 0.0
    # 低速目标仍进入候选构造循环；不得出现按当前速度拒绝的原因。
    context = pipeline().context(env,.5)
    assert 'attacker_speed' not in context['candidate_rejections']


def test_ranking_roundtrip_uses_only_catalog_and_existing_validator():
    env,pipe,p = environment(),pipeline(),planner()
    context = pipe.context(env,.5)
    result = rank_profile_candidates(p,env.get_state_for_planning(),[],context)
    assert result['attack'] and p.calls==1 and result['request_id']
    assert result['anchors']==context['candidates'][0]['anchors']
    assert p.last_trace['plan_validation']=='accepted'


@pytest.mark.parametrize('corruption',['id','duplicate','control','unknown','reason'])
def test_bad_rankings_rejected_without_free_planning(corruption):
    env,pipe,p = environment(),pipeline(),planner()
    good = p._request_attack_plan
    def request(*args,**kwargs):
        reasoning,response = good(*args,**kwargs)
        if corruption=='id': response['request_id']='wrong'
        if corruption=='duplicate': response['ranking']*=2
        if corruption=='control': response['controls']=[1.,2.]
        if corruption=='unknown': response['ranking']=['invented']
        if corruption=='reason': response['no_attack_reason']='conflicting'
        return reasoning,response
    p._request_attack_plan=request
    assert rank_profile_candidates(p,env.get_state_for_planning(),[],pipe.context(env,.5)) is None
    assert not p.last_request_failed


def test_empty_candidates_skip_service_and_failure_is_distinct():
    env,p = environment(),planner()
    result = rank_profile_candidates(p,env.get_state_for_planning(),[],dict(candidates=[]))
    assert not result['attack'] and p.calls==0
    def broken(*args,**kwargs):
        raise TimeoutError()
    p._request_attack_plan=broken
    assert rank_profile_candidates(p,env.get_state_for_planning(),[],pipeline().context(env,.5)) is None
    assert p.last_request_failed


def test_codex_top_one_cannot_change_anchors():
    env,pipe,p = environment(),pipeline(),planner()
    context=pipe.context(env,.5)
    c=context['candidates'][0]
    p.provider='codex'
    response=dict(request_id='queue-123',target_id=c['target_id'],strategy=c['strategy'],anchors=c['anchors'],duration=c['duration'],no_attack_reason=None)
    p.client=NS(request=lambda payload:response,last_trace={})
    assert rank_profile_candidates(p,env.get_state_for_planning(),[],context)['attack']
    response['anchors']=[[100.,100.]]*4
    assert rank_profile_candidates(p,env.get_state_for_planning(),[],context) is None
    assert not p.last_request_failed


def measured_window(collision=False):
    return dict(difficulty=.4,weight=10,valid=True,metrics=dict(avg_min_ttc=2.,mean_evasive_acceleration_mps2=2.,collision_rate=float(collision)))


def reachability(solvable=True):
    return [dict(original_area_m2=10.,dangerous_area_m2=6.,dangerous_solvable=solvable,difficulty=.4)]


def test_actual_execution_and_persistent_knowledge(tmp_path):
    env,pipe,p = environment(),pipeline(),planner()
    pipe.begin_attempt(env,tmp_path)
    # 攻击前已观测驾驶行为，反应时间以加速度变化为事件。
    env.current_step=1
    pipe.profile.observe(env)
    plan = rank_profile_candidates(p,env.get_state_for_planning(),[],pipe.context(env,.5))
    pipe.planned(plan)
    trace=NS(previous_background=set(),previous_static=set())
    env.current_step=2
    pipe.executed(env,None,{},trace)
    assert pipe.pending['actual_attack_execution_frames']==0
    env.current_step=3
    pipe.executed(env,dict(executed=True,target_id=0,request_id='wrong'),{},trace)
    assert pipe.pending['actual_attack_execution_frames']==0
    env.current_step=4
    env.ego_state[2]=9.7
    pipe.executed(env,dict(executed=True,target_id=0,request_id=plan['request_id']),{},trace)
    assert pipe.pending['actual_attack_execution_frames']==1
    pipe.finish_window(measured_window(),reachability())
    pipe.finish_attempt(env)
    row=json.loads((tmp_path/'strategy_returns.jsonl').read_text())
    assert row['D']==.4 and row['drivable_area_change_ratio']==-.4 and row['actual_attack_execution_frames']==1
    assert row['reward']>0 and row['profile_version']
    kb=AttackKnowledge(pipe.policy_id,[tmp_path/'strategy_returns.jsonl'])
    assert kb.retrieve(scene_conditions(env))['attempts']==1
    other=dict(scene_conditions(env),speed_bin='high')
    assert kb.retrieve(other)['attempts']==0
    assert AttackKnowledge('other',[tmp_path/'strategy_returns.jsonl']).rows==[]


@pytest.mark.parametrize('cause',['collision','unsolvable','background','generation_failure'])
def test_bad_outcomes_never_rewarded(cause):
    env,pipe = environment(),pipeline()
    pipe.context(env,.5)
    pipe.pending.update(strategy='cut_in',actual_attack_execution_frames=3,collision=cause=='collision',background_safe=cause!='background')
    pipe.finish_window(measured_window(),reachability(cause!='unsolvable'),failure='rejected' if cause=='generation_failure' else None)
    assert pipe.knowledge.rows[-1]['reward']==0


def test_null_measurement_and_zero_execution_preserved(tmp_path):
    env,pipe = environment(),pipeline()
    pipe.begin_attempt(env,tmp_path)
    pipe.context(env,.5)
    pipe.planned(None,service_failure=True)
    pipe.finish_window()
    row=pipe.knowledge.rows[-1]
    assert row['D'] is None and row['reward'] is None and row['generation_failure_reason']=='planner_service_failure'
    pipe.context(env,.5)
    pipe.pending['strategy']='cut_in'
    pipe.finish_window(measured_window(),reachability())
    assert pipe.knowledge.rows[-1]['generation_failure_reason']=='no_attack_execution'
    assert pipe.knowledge.rows[-1]['reward'] is None
    assert pipe.knowledge.retrieve(scene_conditions(env))['attempts']==2


def test_nonfinite_metrics_json_is_strict(tmp_path):
    env,pipe = environment(),pipeline()
    pipe.begin_attempt(env,tmp_path)
    pipe.context(env,.5)
    pipe.finish_window(dict(difficulty=None,valid=False,weight=1,metrics={'avg_min_ttc':float('inf')}))
    assert json.loads((tmp_path/'strategy_returns.jsonl').read_text())['metrics']['avg_min_ttc'] is None


def test_joint_gate_checks_dynamics_but_never_road_distance():
    env,pipe = environment(),pipeline()
    t=np.arange(1,11)*env.dt
    positions=np.stack((18+8*t,np.zeros_like(t)),axis=-1)[None]
    env.pending_joint_trajectory=NS(agent_ids=np.array([0]),positions_global=positions.copy(),yaws_global=np.zeros((1,10,1)),valid_mask=np.ones((1,10),bool))
    pipe.validate_joint(env)
    env.pending_joint_trajectory.positions_global[0,0,0]+=2
    with pytest.raises(NoSafeJointCandidate,match='dynamics'):
        pipe.validate_joint(env)
    env.pending_joint_trajectory.positions_global=positions.copy()
    env.scenario_dict['lanes'][:,:,1]+=10
    pipe.validate_joint(env)


def test_previously_attacked_vehicle_skips_dynamics_gate():
    env,pipe = environment(),pipeline()
    env.previously_attacked_agent_ids={0}
    positions=np.zeros((1,10,2))
    positions[0,:,0]=np.arange(10)*10
    env.pending_joint_trajectory=NS(agent_ids=np.array([0]),positions_global=positions,
        yaws_global=np.zeros((1,10,1)),valid_mask=np.ones((1,10),bool))
    pipe.validate_joint(env)


def test_generator_profile_on_uses_ranking_and_off_uses_legacy(tmp_path):
    from test_generator_wiring_unit import fixture
    env,p = environment(),planner()
    generator=fixture()
    generator.llm_planner=p
    generator.difficulty_mode='off'
    generator.profile_pipeline.begin_attempt(env,tmp_path)
    generator.plan_and_inject(env)
    assert env.attack_intent['profile_guided'] and generator.profile_pipeline.pending['accepted']
    assert generator._episode_attack_plan_count==1
    generator=fixture()
    generator.profile_enabled=False
    generator.profile_pipeline=None
    calls=[]
    generator.llm_planner=NS(generate_attack_plan=lambda *a,**kw:calls.append(kw) or dict(attack=False,reason='off'),available=True)
    generator.plan_and_inject(env)
    assert len(calls)==1 and 'adversarial_context' not in calls[0]


@pytest.mark.parametrize('outcome',['executed','unsolvable','service_failure'])
def test_real_attempt_loop_accounts_for_all_outcomes(tmp_path,outcome):
    """运行生产尝试循环、生成器及审计；只替换仿真动力学、风险计算和 LLM 传输。"""
    import time
    import torch
    import test_generator_wiring_unit as wiring
    from policies.evaluation_trace import AttemptTrace
    from policies.difficulty_audit import json_safe
    from policies.route_progress import normalized_route_progress
    env,p=environment(),planner()
    if outcome=='service_failure':
        def fail(*args,**kwargs):
            raise TimeoutError()
        p._request_attack_plan=fail
    g=wiring.fixture()
    g.llm_planner=p
    g.difficulty_mode='off'
    g.cfg.sim.evaluation.composite=NS()
    wiring.scope['compute_scenario_danger_score']=lambda *args:.4
    risk=g.risk_metrics
    risk.reset_episode=lambda:None
    risk.evaluate_ea=lambda env:risk.ea_values.append(1.)
    def measure(env):
        if env.attack_intent is None:
            return None
        event=reachability(outcome!='unsolvable')[0]
        event['source_step']=env.current_step
        risk.reachability_events.append(event)
        return event
    risk.evaluate_dangerous_reachability=measure
    cfg=NS(movie_path=str(tmp_path),visualize=False,evaluation=NS(difficulty_control=NS(smoke_acceptance=False)))
    env.steps=3
    env.test_files=['test-scene']
    env.reset=lambda index:None
    env.dump_step_data=lambda index:None
    params=dict(inner_lr=.14,inner_beta=.35,n_guide_steps=3,scale_grad_by_std=True,grad_wrt='clean_guide')
    net=NS(guide_config=NS(params=params),Loss_Calculater=NS(weights=torch.tensor([1.5,1.,1.,1.])),diffuse_args={})
    env.diffusion_controller=NS(policy=NS(nets={'policy':net}),active_guidance_functions=['scenario_collision','route','scenario_ttc','llm_anchor'])
    def prepare():
        state=env.data_dict['agent'][-1][0]
        times=np.arange(1,11)*env.dt
        pos=state[:2]+times[:,None]*state[2:4]
        intent=env.attack_intent
        env.pending_joint_trajectory=NS(agent_ids=np.array([0]),positions_global=pos[None],yaws_global=np.zeros((1,10,1)),
            valid_mask=np.ones((1,10),bool),source_step=env.current_step,
            metadata=dict(anchor_guidance_active=bool(intent),anchor_guidance_target_id=0 if intent else None,
                          planner_request_id=intent.get('request_id') if intent else None))
    env.prepare_background_traffic=prepare
    env.get_attack_target_prediction=lambda:None
    def step(*args):
        env.current_step+=1
        env.ego_state[:2]+=env.ego_state[2:4]*env.dt
        env.data_dict['agent'][-1][0,:2]=env.pending_joint_trajectory.positions_global[0,0]
        return None,env.current_step==3,dict(collision=False,off_route=False,completed=False,progress=.1)
    env.step=step
    tree=ast.parse((Path(__file__).resolve().parents[1]/'run_simulation.py').read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_run_scenario_attempt')
    scope=dict(Path=Path,AttemptTrace=AttemptTrace,time=time,np=np,json=json,json_safe=json_safe,
               normalized_route_progress=normalized_route_progress,NoSafeJointCandidate=NoSafeJointCandidate)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'real_attempt','exec'),scope)
    evaluator=NS(env=env,generator=g,cfg=cfg,policy=NS(act=lambda obs:None),_metric_offsets=lambda:{},
                 _build_episode_result=lambda *args:dict(attack_plan_count=g._episode_attack_plan_count,obstacle_plan_count=0))
    _,_,result,_=scope['_run_scenario_attempt'](evaluator,0,0,False)
    directory=tmp_path/'scenario_000'
    rows=[json.loads(line) for line in (directory/'strategy_returns.jsonl').read_text().splitlines()]
    profile=json.loads((directory/'ego_profile.json').read_text())
    assert len(rows)==1 and (directory/'attempt_result.json').exists() and (directory/'execution_trace.jsonl').exists()
    if outcome=='executed':
        assert result['attack_executed_frames']==rows[0]['actual_attack_execution_frames']==3
        assert profile['frames']==4 and rows[0]['D']==.4
    else:
        assert result['attack_executed_frames']==rows[0]['actual_attack_execution_frames']==0
        assert result['generation_failure'] and rows[0]['generation_failure_reason']
        assert rows[0]['D'] is None
        assert profile['frames']==(4 if outcome=='service_failure' else 1)
