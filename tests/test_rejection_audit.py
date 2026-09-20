"""拒绝审计应保存未执行预测和实际越界指标，不修改轨迹。"""
from types import SimpleNamespace as NS
import numpy as np
from policies.rejection_audit import rejection_record


def test_snapshot_reports_road_distance_as_diagnostic_without_mutation():
    positions=np.array([[[2.,1.],[2.,2.],[2.,3.]]])
    joint=NS(source_step=0,agent_ids=np.array([0]),positions_global=positions,
             yaws_global=np.full((1,3,1),np.pi/2),valid_mask=np.ones((1,3),bool),metadata={})
    state=np.array([2.,0.,0.,10.,np.pi/2,4.,2.])
    env=NS(current_step=0,dt=.1,attack_intent={'target_id':0},pending_joint_trajectory=joint,
           data_dict={'agent':[[state]]},scenario_dict={'lanes':[np.array([[0.,-10.],[0.,10.]])]})
    pipeline=NS(dynamic_limits=dict(max_speed=20.,max_acceleration=6.,max_jerk=20.,max_step_distance=2.),builder=NS(half_width=1.8))
    record=rejection_record(env,pipeline,'profile_joint_background_margin',joint)
    assert record['executed'] is False
    assert record['rejected_metrics'][0]['road']['initial_max_corner_distance_m']==3.
    assert record['rejected_metrics'][0]['road']['gate_enabled'] is False
    assert record['rejected_metrics'][0]['road']['violated'] is False
    assert record['rejected_metrics'][0]['road']['exceeds_diagnostic_reference']
    assert record['original_joint']['positions']==positions.tolist()
    np.testing.assert_array_equal(positions,[[[2.,1.],[2.,2.],[2.,3.]]])


def test_missing_prediction_is_explicit():
    env=NS(current_step=0,attack_intent=None,pending_joint_trajectory=None)
    record=rejection_record(env,None,'no_joint',None)
    assert record['original_joint'] is None and record['rejected_joint'] is None
