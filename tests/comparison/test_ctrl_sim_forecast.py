"""评估预测的隔离和有效性门禁，不镜像模型实现。"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.comparison.ctrl_sim_forecast import CtrlSimJointForecaster, clone_model_state, validate_forecast
from experiments.comparison.formal_metrics import FormalRiskObserver
from policies.traffic_types import JointTrajectory


def joint(step=7, horizon=30, ids=(0, 2)):
    n = len(ids)
    return JointTrajectory(step, np.asarray(ids), np.zeros((n, horizon, 2)),
                           np.zeros((n, horizon, 1)), np.zeros((n, horizon, 2)),
                           np.ones((n, horizon), bool))


def env():
    return SimpleNamespace(current_step=7, dt=.1, agent_active=np.array([1, 0, 1], bool),
                           pending_joint_trajectory=object())


def test_forecast_accepts_complete_aligned_joint():
    validate_forecast(joint(), env(), 3.)


@pytest.mark.parametrize('forecast,reason', [
    (joint(step=6), 'source_step'), (joint(horizon=29), 'horizon'),
    (joint(ids=(0,)), 'active_agent'),
])
def test_forecast_rejects_missing_or_stale_world(forecast, reason):
    with pytest.raises(ValueError, match=reason):
        validate_forecast(forecast, env(), 3.)


def test_forecast_rejects_missing_agent_future_and_nan():
    forecast = joint()
    forecast.valid_mask[0, 29] = False
    with pytest.raises(ValueError, match='coverage'):
        validate_forecast(forecast, env(), 3.)
    forecast.valid_mask[:] = True
    forecast.positions_global[0, 2, 0] = np.nan
    with pytest.raises(ValueError, match='nonfinite'):
        validate_forecast(forecast, env(), 3.)


def test_model_copy_shares_parameters_but_isolates_mutable_buffers():
    model = torch.nn.BatchNorm1d(2).eval()
    cloned = clone_model_state(model)
    assert cloned.weight is model.weight
    assert cloned.running_mean is not model.running_mean
    cloned.running_mean.add_(4.)
    assert torch.equal(model.running_mean, torch.zeros(2))


def test_forecast_initialization_also_preserves_random_stream(monkeypatch):
    import simulator

    class Model:
        def to(self, device): return self

    def load(path):
        torch.rand(10)  # 模拟checkpoint加载时的随机参数初始化。
        np.random.rand(10)
        return Model()

    monkeypatch.setattr(simulator.CtRLSim, 'load_from_checkpoint', load)
    monkeypatch.setattr(simulator, 'CtRLSimBehaviourModel', lambda **kw: SimpleNamespace(**kw))
    cfg = SimpleNamespace(model_path='fixture', use_rtg=True, predict_rtgs=True,
                          action_temperature=1., tilt=10)
    world = SimpleNamespace(behaviour_model=None, mode='scenario_dreamer', dt=.1, steps=400,
                            ctrl_sim_dset=object(), cfg=SimpleNamespace(sim=SimpleNamespace(behaviour_model=cfg)))
    cpu_state = torch.random.get_rng_state().clone()
    numpy_state = np.random.get_state()
    CtrlSimJointForecaster(world)
    assert torch.equal(cpu_state, torch.random.get_rng_state())
    np.testing.assert_array_equal(numpy_state[1], np.random.get_state()[1])
    assert numpy_state[2:] == np.random.get_state()[2:]


def test_external_forecast_never_overwrites_execution_prediction():
    class Risk:
        reachability = SimpleNamespace(horizon=3.)

        def evaluate_ea(self, world): return 2.
        def begin_attack(self, world): pass
        def evaluate_dangerous_reachability(self, world):
            self.seen = world.pending_joint_trajectory
            return {'difficulty': .2, 'dangerous_solvable': True}

    world, forecast, risk = env(), joint(), Risk()
    actual = world.pending_joint_trajectory
    observer = FormalRiskObserver(SimpleNamespace(), risk)
    observer.begin_frame(world)
    observer.finish_prediction(world, forecast, 'ctrl_sim_rollout')
    assert risk.seen is forecast
    assert world.pending_joint_trajectory is actual
    assert observer._prediction_available
    observer.begin_frame(world)
    observer.finish_prediction(world, joint(horizon=2), 'ctrl_sim_rollout')
    assert not observer._prediction_available
    assert observer.finish_transition({}, 2.)['measurement_status'] == 'forecast_horizon_too_short'
    assert observer.result()['danger'] is None
