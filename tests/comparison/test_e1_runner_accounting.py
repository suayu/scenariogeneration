"""验证闭环失败后的实际观测不会被丢弃，未执行 EA 不进入指标。"""
from types import SimpleNamespace
import numpy as np
import pytest


@pytest.mark.parametrize('executed_before_failure', [0, 1])
def test_rejection_keeps_partial_execution_and_excludes_pending_ea(tmp_path, monkeypatch, executed_before_failure):
    from experiments.comparison import run_e1_group as runner

    class Observer:
        def __init__(self, config):
            self.risk = SimpleNamespace(ea_values=[])
            self.records = []
        def reset(self): pass
        def begin_frame(self, env, **kwargs): self.risk.ea_values.append(999.)
        def finish_prediction(self, *args): pass
        def finish_transition(self, *args):
            self.records.append({'ea_mps2': 2., 'reachability': None, 'prediction_error': None})
        def result(self):
            return {'records': self.records, 'danger': None, 'valid': False,
                    'valid_frame_count': 0, 'total_frame_count': len(self.records), 'coverage': 0.}

    class Trace:
        def __init__(self, out): pass
        def state(self, *args, **kwargs): pass
        def append(self, *args): pass
        def summary(self): return {}

    class Env:
        def __init__(self):
            self.cfg = SimpleNamespace(sim=SimpleNamespace(evaluation=None))
            self.ego_state = np.array([0., 0., 2., 0., 0., 4., 2., 1.])
            self.data_dict = {'agent': [np.empty((0, 8))]}
            self.agent_active = np.array([], dtype=bool)
            self.scenario_dict = {'route': np.array([[0., 0.], [20., 0.]]),
                                 'lanes': np.array([[[0., 0.], [20., 0.]]])}
            self.test_files = ['/frozen/a.pkl']
            self.current_step = 0
            self.pending_joint_trajectory = None
        def reset(self, index): return None
        def render_state(self, **kwargs): pass
        def prepare_background_traffic(self):
            if self.current_step == executed_before_failure:
                raise runner.NoSafeJointCandidate('no_safe_candidates')
            return None
        def step(self, action, anchors, refined):
            self.current_step += 1
            self.ego_state[0] += .2
            return None, False, {'collision': True, 'completed': False, 'progress': .01}

    monkeypatch.setattr(runner, 'FormalRiskObserver', Observer)
    monkeypatch.setattr(runner, 'AttemptTrace', Trace)
    monkeypatch.setattr(runner, 'generate_video', lambda **kw: tmp_path / 'video.mp4')
    args = SimpleNamespace(output_dir=tmp_path, seed=42, group='scenario_dreamer', steps=2)
    policy = SimpleNamespace(reset=lambda obs: None, act=lambda obs: None)
    result = runner.run_episode(Env(), policy, 0, 0, args)
    assert result['generation_failure_reason'] == 'no_safe_candidates'
    assert result['executed_steps'] == executed_before_failure
    assert result['ea_valid_frame_count'] == executed_before_failure
    assert result['collision'] is (True if executed_before_failure else None)
    assert result['mean_evasive_acceleration_mps2'] == (2. if executed_before_failure else None)
    assert result['scenario_danger_score'] is None
    assert (tmp_path / 'movies/scenario_000/attempt_result.json').is_file()
