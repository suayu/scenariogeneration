"""单场冒烟的成功、预测失败和产物门禁。"""
from experiments.comparison.run_e1_smoke import smoke_acceptance


def test_smoke_refuses_two_scenes():
    assert smoke_acceptance('safe_sim', {'episodes': [{}, {}]}) == ['expected_exactly_one_episode']


def test_smoke_requires_execution_and_rejects_forecast_failure(tmp_path):
    video = tmp_path / 'video.mp4'
    video.write_bytes(b'fixture')
    row = {'executed_steps': 10, 'danger_valid': True, 'video_path': str(video)}
    report = {'reachability_source': 'ctrl_sim_rollout', 'episodes': [row]}
    assert smoke_acceptance('safe_sim', report) == []
    row['forecast_failure_count'] = 1
    assert 'forecast_failure' in smoke_acceptance('safe_sim', report)
    row.update(executed_steps=0, generation_failure=True)
    assert {'no_actual_execution', 'generation_failure'} <= set(smoke_acceptance('safe_sim', report))


def test_native_scenario_dreamer_may_keep_unmeasurable_danger_null(tmp_path):
    video = tmp_path / 'video.mp4'
    video.write_bytes(b'fixture')
    report = {'reachability_source': 'native', 'episodes': [
        {'executed_steps': 10, 'danger_valid': False, 'video_path': str(video)}]}
    assert smoke_acceptance('scenario_dreamer', report) == []
    report['reachability_source'] = 'ctrl_sim_rollout'
    assert 'no_valid_ea_and_reachability' in smoke_acceptance('safe_sim', report)
