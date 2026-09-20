"""独立验证连续约束投影；不使用碰撞率替代物理可行性。"""
import numpy as np
import pytest

from policies.trajectory_projection import project_joint,project_positions
from policies.joint_safety import NoSafeJointCandidate

LIMITS = dict(max_speed=20., max_acceleration=6., max_jerk=12., max_step_distance=2.)


def test_constant_velocity_is_preserved():
    initial = np.array([0., 0., 0., 5., np.pi/2, 4., 2.])
    reference = np.column_stack((np.zeros(32), np.arange(1, 33)*.5))
    projected, velocity, yaw, audit = project_positions(reference, initial, .1, LIMITS)
    np.testing.assert_allclose(projected, reference, atol=1e-6)
    assert audit['max_deviation_m'] < 1e-6


def test_jerky_path_becomes_feasible_without_changing_initial_state():
    initial = np.array([0., 0., 0., 5., np.pi/2, 4., 2.])
    reference = np.column_stack((.01*(-1.)**np.arange(32), np.arange(1, 33)*.5))
    projected, velocity, yaw, audit = project_positions(reference, initial, .1, LIMITS)
    actual_velocity = np.diff(np.vstack((initial[:2], projected)), axis=0)/.1
    actual_acceleration = np.diff(np.vstack((initial[2:4], actual_velocity)), axis=0)/.1
    actual_jerk = np.diff(actual_acceleration, axis=0)/.1
    assert np.linalg.norm(actual_acceleration, axis=1).max() <= 6.
    assert np.linalg.norm(actual_jerk, axis=1).max() <= 12.
    assert audit['max_deviation_m'] < .05


def test_large_trajectory_rewrite_is_rejected():
    initial = np.array([0., 0., 0., 0., 0., 4., 2.])
    reference = np.column_stack((np.arange(1, 33)*5., np.zeros(32)))
    with pytest.raises(NoSafeJointCandidate):
        project_positions(reference, initial, .1, LIMITS, max_deviation=.1)


def test_previous_attacker_is_not_projected():
    from types import SimpleNamespace as NS
    from policies.traffic_types import JointTrajectory
    positions=np.array([[[10.,0.],[30.,0.]]])
    joint=JointTrajectory(source_step=0,agent_ids=np.array([7]),positions_global=positions.copy(),
                          velocities_global=np.array([[[100.,0.],[200.,0.]]]),
                          yaws_global=np.zeros((1,2,1)),valid_mask=np.ones((1,2),bool),metadata={})
    settings=NS(max_seconds_per_vehicle=1.,max_deviation_m=.1)
    result=project_joint(joint,np.zeros((8,7)),.1,LIMITS,settings,exempt_agent_ids={7})
    np.testing.assert_equal(result.positions_global,positions)
    assert result.metadata['dynamics_projection']['vehicles'][0]['exempt']
