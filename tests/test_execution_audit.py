"""验证实际占用统计、执行首段保护和控制参数审计。"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from policies.evaluation_trace import AttemptTrace, collision_pairs
from policies.joint_safety import NoSafeJointCandidate, validate_execution_prefix


def joint(points, headings=None):
    points = np.asarray(points, dtype=float)
    count = len(points)
    return SimpleNamespace(agent_ids=np.arange(count), positions_global=points[:, None],
                           yaws_global=np.zeros((count, 1, 1)) if headings is None else headings,
                           valid_mask=np.ones((count, 1), dtype=bool))


def test_inactive_overlaps_are_not_counted():
    states = np.array([[0, 0, 0, 0, 0, 4, 2], [0, 0, 0, 0, 0, 4, 2]])
    assert collision_pairs(states, [True, False], []) == (set(), set())
    assert collision_pairs(states, [True, True], [])[0] == {(0, 1)}


def test_actual_first_segment_crossing_is_rejected():
    states = np.array([[-4, 0, 0, 0, 0, 2, 2], [4, 0, 0, 0, 0, 2, 2]])
    with pytest.raises(NoSafeJointCandidate):
        validate_execution_prefix(states, joint([[4, 0], [-4, 0]]), [])


def test_static_obstacle_checks_first_segment():
    states = np.array([[-4, 0, 0, 0, 0, 2, 2]])
    with pytest.raises(NoSafeJointCandidate):
        validate_execution_prefix(states, joint([[4, 0]]), np.array([[0, 0, 0, 1, 1]]))
    validate_execution_prefix(states, joint([[-3, 0]]), np.array([[0, 0, 0, 1, 1]]))


def test_actual_parameters_cannot_be_replaced_by_expected_records(tmp_path):
    trace = AttemptTrace(tmp_path)
    net = SimpleNamespace(guide_config=SimpleNamespace(params={
        'inner_lr': .2, 'inner_beta': .35, 'n_guide_steps': 3,
        'scale_grad_by_std': True, 'grad_wrt': 'clean_guide'}),
        Loss_Calculater=SimpleNamespace(weights=torch.tensor([1.5, 1., 1.5, 3.25])),
        diffuse_args={'sample_step': 4})
    env = SimpleNamespace(diffusion_controller=SimpleNamespace(policy=SimpleNamespace(nets={'policy': net})),
                          pending_joint_trajectory=None)
    generator = SimpleNamespace(difficulty_mode='target', _iterative_stage=.5,
        difficulty_controller=SimpleNamespace(parameters=lambda _: {
            'inner_lr': .14, 'inner_beta': .35, 'n_guide_steps': 3,
            'scenario_ttc': 1.5, 'llm_anchor': 3.25}))
    with pytest.raises(RuntimeError, match='inner_lr'):
        trace.prediction(env, generator, 0.)


def test_numpy_candidate_metadata_serializes(tmp_path):
    trace = AttemptTrace(tmp_path)
    trace.append({'candidate': np.zeros((4, 2)), 'missing': np.float64('nan')})
    assert 'null' in trace.path.read_text()
