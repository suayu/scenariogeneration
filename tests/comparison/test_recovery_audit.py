"""STRIVE空场景续行与失败不能被当成有效基线结果。"""
from experiments.comparison.run_e1_smoke import smoke_acceptance
from experiments.comparison.run_passive_baseline import aggregate
from experiments.comparison.run_strive_baseline import empty_joint_prediction, slice_joint


def test_empty_prediction_can_be_shifted_without_claiming_model_or_attack():
    joint = empty_joint_prediction(3, 60)
    assert joint.positions_global.shape == (0, 60, 2)
    assert joint.metadata['model_invoked'] is False
    shifted = slice_joint(joint, 5, 2)
    assert shifted.positions_global.shape == (0, 58, 2)
    assert shifted.metadata['attack_row'] == 0


def test_runtime_failure_is_retained_without_fabricating_zero_collision():
    row = {'runtime_failure': True, 'generation_failure': False, 'executed_steps': 0,
           'collision': None, 'off_route': None, 'completed': None, 'progress': None,
           'danger_valid': False, 'ability_valid': False,
           'feasibility': {'background_collision_frames': 0, 'background_static_collision_frames': 0}}
    result = aggregate([row])
    assert result['runtime_failure_count'] == 1
    assert result['generation_failure_count'] == 0
    assert result['attempted_scenario_count'] == 1
    assert result['actual_execution_scenario_count'] == 0
    assert result['collision_rate'] is None


def test_strive_smoke_requires_real_attack_execution(tmp_path):
    video = tmp_path/'video.mp4'
    video.write_bytes(b'fixture')
    row = {'executed_steps': 10, 'video_path': str(video)}
    assert 'no_actual_strive_attack' in smoke_acceptance('strive', {'episodes': [row]})
    row.update(attack_plan_count=1, attack_executed_frames=10)
    assert smoke_acceptance('strive', {'episodes': [row]}) == []


def test_passive_cli_keeps_tracking_out_of_seed_function(monkeypatch, tmp_path):
    """验证真实入口参数接线，避免预测参数误传给全局随机种子。"""
    import contextlib
    from types import SimpleNamespace
    import pytest
    from experiments.comparison import run_passive_baseline as runner
    monkeypatch.setattr(runner.sys, 'argv', ['runner', '--method', 'safe_sim',
        '--manifest', str(tmp_path/'manifest.json'), '--output-dir', str(tmp_path/'output'),
        '--reachability-source', 'ctrl_sim_rollout', '--forecast-tracking', 'persistent'])
    monkeypatch.setattr(runner, 'validate_manifest', lambda path: ({}, [tmp_path/'scene.pkl']))
    monkeypatch.setattr(runner, 'initialize', lambda **kw: contextlib.nullcontext())
    monkeypatch.setattr(runner, 'compose', lambda **kw: SimpleNamespace(sim=SimpleNamespace(seed=42)))
    observed = []
    monkeypatch.setattr(runner, 'set_seed', lambda seed: observed.append(seed))
    class StartupReached(Exception):
        pass
    def stop_at_simulator(cfg):
        raise StartupReached
    monkeypatch.setattr(runner, 'Simulator', stop_at_simulator)
    with pytest.raises(StartupReached):
        runner.main()
    assert observed == [42]
