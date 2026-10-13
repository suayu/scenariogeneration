"""公共EA区分完整空集合与缺失/求解失败，所有方法使用同一口径。"""
from types import SimpleNamespace

import numpy as np

from policies.risk_metrics import AdversarialRiskMetrics


def world(agents=()):
    return SimpleNamespace(dt=.1, ego_state=np.array([0, 0, 5, 0, 0, 4.5, 1.8, 1.]),
        data_dict={'agent': [np.asarray(agents, dtype=float).reshape(-1, 8)]},
        agent_active=np.ones(len(agents), bool), get_static_obstacles=lambda: [])


def test_empty_complete_world_requires_no_evasion():
    metrics = AdversarialRiskMetrics()
    assert metrics.evaluate_ea(world()) == 0.
    assert metrics.last_ea_diagnostics['status'] == 'valid_empty_object_set'
    assert metrics.ea_undefined_frames == 0


def test_distant_agents_do_not_create_undefined_evasion():
    metrics = AdversarialRiskMetrics()
    assert metrics.evaluate_ea(world([[100, 0, 5, 0, 0, 4.5, 1.8, 1]])) == 0.


def test_invalid_active_agent_cannot_be_misclassified_as_empty():
    metrics = AdversarialRiskMetrics()
    value = metrics.evaluate_ea(world([[np.nan, 0, 5, 0, 0, 4.5, 1.8, 1]]))
    assert np.isnan(value)
    assert metrics.last_ea_diagnostics['status'] == 'invalid_input'


def test_solver_failure_is_not_silently_dropped_among_finite_pairs():
    metrics = AdversarialRiskMetrics()
    calls = iter([.5, np.nan])
    metrics.ea_solver.compute = lambda *args: next(calls)
    value = metrics.evaluate_ea(world([[10, 0, 5, 0, 0, 4.5, 1.8, 1], [20, 0, 5, 0, 0, 4.5, 1.8, 1]]))
    assert np.isnan(value)
    assert metrics.last_ea_diagnostics['invalid_pair_count'] == 1


def test_misaligned_state_and_activation_are_invalid():
    metrics = AdversarialRiskMetrics()
    env = world()
    env.agent_active = np.array([True])
    assert np.isnan(metrics.evaluate_ea(env))


def test_disabled_ea_is_not_filled_with_zero():
    metrics = AdversarialRiskMetrics({'ea': {'enabled': False}})
    assert metrics.evaluate_ea(world()) is None
