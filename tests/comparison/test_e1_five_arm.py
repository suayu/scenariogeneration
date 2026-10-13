"""新 E1 的几何语义、失败分母、画像消融及来源固定验证。"""
import math
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.comparison.e1_metrics import obb_ttc,center_tca,aggregate_attempts,ExecutionMetrics,trajectory_diagnostics
from experiments.comparison.e1_protocol import select_100,shard_manifest,group_overrides,GROUPS
from policies.traffic_types import JointTrajectory


def car(x,y,vx=0,vy=0,yaw=0):
    return np.array([x,y,vx,vy,yaw,4.,2.,1.])


def test_ttc_requires_geometric_contact_not_just_closest_approach():
    a,b=car(0,0,2),car(10,5)
    assert center_tca(a,b)==5.
    assert math.isinf(obb_ttc(a,b))


def test_ttc_front_rear_accounts_for_dimensions():
    assert obb_ttc(car(0,0,2),car(10,0))==pytest.approx(3.)


def test_ttc_crossing_oriented_boxes_and_diverging_motion():
    assert obb_ttc(car(0,0,2),car(5,5,0,-2,np.pi/2))==pytest.approx(1.)
    assert math.isinf(obb_ttc(car(0,0,-2),car(10,0)))


def test_touching_and_overlapping_boxes_have_zero_ttc():
    assert obb_ttc(car(0,0),car(4,0))==0.
    assert obb_ttc(car(0,0),car(3,0))==0.


@pytest.mark.parametrize('bad',[np.nan,np.inf])
def test_ttc_bad_input_is_invalid_not_no_collision(bad):
    a=car(0,0);a[0]=bad
    with pytest.raises(ValueError):obb_ttc(a,car(10,0))


def test_partial_failed_attempt_collision_remains_in_attempt_yield():
    rows=[{'executed_steps':10,'collision':True,'runtime_failure':True},
          {'executed_steps':0,'collision':None,'generation_failure':True},
          {'executed_steps':30,'collision':False,'completed':True,'progress':.99}]
    result=aggregate_attempts(rows,4)
    assert result['collision_yield_per_planned_attempt']==.25
    assert result['collision_rate_among_actual_execution']==.5
    assert result['collision_rate']==0.
    assert result['unattempted_scenario_count']==1
    assert result['execution_coverage']==.5
    assert result['generation_failure_count']==1
    assert result['runtime_failure_count']==1


def test_no_execution_or_valid_d_is_missing_not_zero():
    result=aggregate_attempts([{'executed_steps':0,'collision':None,'generation_failure':True}],100)
    assert result['collision_rate'] is None
    assert result['mean_scenario_danger_score'] is None
    assert result['danger_valid_count']==0


def test_precontact_velocity_preserved_when_partner_deactivated():
    env=SimpleNamespace(ego_state=car(0,0,3),data_dict={'agent':[np.array([car(6,0,-2)])]},
        agent_active=np.array([True]),current_step=0,scenario_dict={'lanes':np.array([[[0,0],[20,0]]])})
    metrics=ExecutionMetrics(env)
    env.ego_state=car(1,0,0);env.data_dict['agent'].append(np.array([car(4,0,0)]))
    env.agent_active[:]=False;env.current_step=1
    metrics.observe(env,{'collision':True})
    r=metrics.result()
    assert r['relative_collision_speed_mps']==5.
    assert r['ego_collision_speed_mps']==3.
    assert r['ego_collision_events'][0]['partner_id']==0


def manifest250():
    files=[f'/source_{name}/s_{i}.pkl' for name,count in [('old',75),('ldm1',55),('ldm2',120)] for i in range(count)]
    return {'scenario_files':files,'scenario_sha256':[str(i) for i in range(250)]}


def test_selection_stratified_and_stable_no_outcome_inputs():
    a=select_100(manifest250());b=select_100(manifest250())
    assert a==b and len(a['scenario_files'])==100
    assert len(set(a['original_scenario_indices']))==100
    assert sum('/source_old/' in x for x in a['scenario_files'])==30
    assert sum('/source_ldm1/' in x for x in a['scenario_files'])==22
    assert sum('/source_ldm2/' in x for x in a['scenario_files'])==48


def test_sharding_preserves_global_ids():
    m=select_100(manifest250())
    shards=[shard_manifest(m,list(range(i,min(i+10,100)))) for i in range(0,100,10)]
    assert sum([s['original_scenario_indices'] for s in shards],[])==m['original_scenario_indices']


def test_profile_groups_share_budget_solver_and_candidate_expansion():
    off=group_overrides(GROUPS[0]);on=group_overrides(GROUPS[1])
    different=set(off)^set(on)
    assert different=={'sim.traffic_model.iterative_adversarial.profile_enabled=false',
                       'sim.traffic_model.iterative_adversarial.profile_enabled=true'}
    for row in (off,on):
        assert 'sim.llm.max_planning_calls=4' in row
        assert 'sim.ego_profile.history_paths=[]' in row
        assert 'sim.llm.obstacles.enabled=false' in row


def test_safe_original_losses_are_explicit_legacy_mode_unchanged():
    overrides=group_overrides('safe_sim')
    assert 'sim.traffic_model.guidance.manual.weights=[0.5,0.5,0.5,0.5]' in overrides
    assert any('safe_sim_causecollision' in x for x in overrides)
    from experiments.comparison.methods import METHODS
    assert 'sim.traffic_model.guidance.mode=default' in METHODS['safe_sim'].overrides


def test_strive_interpolation_spikes_diagnosed_without_repair():
    state=np.array([car(0,0,2)])
    positions=np.array([[[.2,0],[1.,0],[1.8,0]]],dtype=np.float32)
    joint=JointTrajectory(0,np.array([0]),positions,np.zeros((1,3,1)),np.zeros((1,3,2)),np.ones((1,3),bool))
    before=joint.positions_global.copy()
    r=trajectory_diagnostics(state,joint,.1)
    assert r['max_acceleration_mps2']>50 and r['max_jerk_mps3']>500
    np.testing.assert_array_equal(before,joint.positions_global)


def test_deactivated_noncontact_actor_not_counted_as_future_threat():
    env=SimpleNamespace(ego_state=car(0,0,3),data_dict={'agent':[np.array([car(10,0)])]},
        agent_active=np.array([True]),current_step=0,scenario_dict={'lanes':np.array([[[0,0],[20,0]]])})
    metrics=ExecutionMetrics(env)
    env.agent_active[:]=False;env.current_step=1
    metrics.observe(env,{})
    r=metrics.result()
    assert r['min_ttc_s'] is None and r['min_ttc_capped_7s']==7.
    assert r['ttc_no_contact_in_horizon'] is True


def test_paired_missing_values_not_zero_and_no_pair_is_null():
    from experiments.comparison.e1_reporting import paired_summary
    a=[{'scenario_index':0,'executed_steps':0,'collision':None},
       {'scenario_index':1,'executed_steps':14,'collision':True,'runtime_failure':True}]
    b=[{'scenario_index':0,'executed_steps':10,'collision':False},
       {'scenario_index':1,'executed_steps':20,'collision':False}]
    r=paired_summary(a,b,'collision')
    assert r['paired_observed_count']==1 and r['mean_difference']==1.
    assert r['matched_attempt_count']==2
    assert paired_summary(a,b,'min_ttc_s')['mean_difference'] is None
