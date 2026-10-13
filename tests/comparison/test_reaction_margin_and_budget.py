"""验证时间预算、响应空间曲线和失败隔离，不以碰撞效果作门禁。"""
from types import SimpleNamespace
import numpy as np
from experiments.comparison.optimization_budget import OptimizationBudget
from experiments.comparison.reaction_margin import summarize_attacks
from policies.risk_metrics import RouteReachability


def test_budget_stops_after_completed_iteration_and_retains_count():
    now = [0.]
    class Bar:
        def __init__(self, iterable):self.iterable=iterable
        def __iter__(self):return iter(self.iterable)
        def close(self):pass
    def optimizer():
        for k in tqdm.tqdm(range(10)):
            now[0] += 2
    optimizer.__globals__['tqdm']=SimpleNamespace(tqdm=Bar)
    budget=OptimizationBudget(5,clock=lambda:now[0])
    budget.install(optimizer,'adversarial')
    optimizer()
    assert budget.iterations['adversarial']==3
    assert budget.truncated_stages==['adversarial']


def test_attack_minimum_preserves_missing_and_separates_plans():
    rows=[]
    for step,area in [(1,8.),(2,None),(3,2.)]:
        rows.append({'attack':{'source_step':0,'target_id':1,'strategy':'cut_in'},
            'valid':area is not None,'reachability':None if area is None else
            {'source_step':step,'dangerous_area_m2':area,'original_area_m2':10.,
             'minimum_dangerous_area_m2':area/2}})
    r=summarize_attacks(rows)[0]
    assert r['minimum_terminal_area_during_attack_m2']==2.
    assert r['minimum_future_area_during_attack_m2']==1.
    assert r['valid_frame_count']==2 and r['observed_frame_count']==3
    assert r['area_over_attack_frames'][1]['area_m2'] is None


def test_legacy_curve_terminal_equals_original_area_and_empty_tail_zero():
    env=SimpleNamespace(scenario_dict={'route':np.array([[0.,0.],[30.,0.]])},
        ego_state=np.array([0.,0.,1.,0.,0.,4.,2.,1.]),
        data_dict={'agent':[np.empty((0,8))]},agent_active=np.empty(0,bool),
        pending_joint_trajectory=None,dt=.1)
    reach=RouteReachability({'horizon_seconds':1.,'timestep_seconds':.5})
    r=reach.compute(env,False)
    assert len(r['area_curve'])==2
    assert r['area_curve'][-1]['area_m2']==r['area_m2']
    obs=[{'center':[0.,0.],'yaw':0.,'length':100.,'width':100.}]
    blocked=reach.compute(env,False,obs)
    assert [x['area_m2'] for x in blocked['area_curve']]==[0.,0.]


def test_scheduler_isolates_strive_and_keeps_other_jobs(monkeypatch, tmp_path):
    from experiments.comparison import launch_e1_five_arm as launcher
    # 运行器故障仍保留失败条目，但不能吞掉其它方法的尝试。
    class Process:
        pid=123
        def __init__(self, command, **kw):self.group=command[0]
        def poll(self):return 1 if self.group=='strive' else 0
    monkeypatch.setattr(launcher,'available_gpus',lambda *x:[0])
    monkeypatch.setattr(launcher,'job_command',lambda g,*x:[g])
    monkeypatch.setattr(launcher.subprocess,'Popen',Process)
    monkeypatch.setattr(launcher.time,'sleep',lambda *x:None)
    args=SimpleNamespace(output_dir=tmp_path,gpus=[0],min_free_gb=1,model='unused',models=['unused'])
    jobs=[{'group':g,'manifest':tmp_path/'manifest.json','directory':tmp_path/g/f'shard_{i}'}
          for i,g in enumerate(['strive','strive','scenario_dreamer','safe_sim'])]
    result=launcher.execute_phase(jobs,args,'formal')
    assert [r['group'] for r in result['jobs']]==['strive','scenario_dreamer','safe_sim']
    assert result['failed_groups']==['strive']
    assert len(result['unstarted_jobs'])==1 and not result['passed']


def test_area_observation_remains_valid_when_ea_and_d_are_invalid():
    record={'attack':{'source_step':2,'target_id':1,'strategy':'cut_in'},'source_step':3,'frame_dt':.1,
            'valid':False,'measurement_status':'invalid_ea',
            'reachability':{'source_step':3,'dangerous_area_m2':4.,'original_area_m2':8.}}
    summary=summarize_attacks([record])[0]
    assert summary['minimum_terminal_area_during_attack_m2']==4.
    row=summary['area_over_attack_frames'][0]
    assert row['valid'] and not row['D_valid']
    assert row['time_since_attack_seconds']==.1


def test_scene_budget_is_cumulative_across_requests():
    from policies.strive_adapter import StriveController
    # 无需启动优化器即可验证预算不会在每次重规划恢复为300秒。
    c=object.__new__(StriveController);c.optimization_seconds=300.;c.optimization_spent=215.
    assert c.remaining_optimization_seconds==85.
    c.optimization_spent+=90.
    assert c.remaining_optimization_seconds==0.


def test_strive_smoke_failure_does_not_block_other_formal_groups():
    from experiments.comparison.launch_e1_five_arm import formal_groups_after_smoke
    smoke={'passed':False,'failed_groups':['strive'],'global_failure':False,
           'jobs':[{'group':'strive','passed':False},{'group':'safe_sim','passed':True}]}
    groups=formal_groups_after_smoke(smoke)
    assert 'strive' not in groups and 'scenario_dreamer' in groups and 'safe_sim' in groups
    smoke['global_failure']=True
    assert formal_groups_after_smoke(smoke)==[]
