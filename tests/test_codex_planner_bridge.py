"""控制模式、严格规划契约和服务失败的轻量回归。"""
import json
from pathlib import Path
import threading
import time
import uuid

import pytest

from policies.codex_planner_bridge import CodexQueueClient, PlannerServiceFailure, validate_plan, finite_context, validate_request
from policies.difficulty_control import DifficultyController
from policies.planner_smoke_gate import validate_planner_smoke


def plan(rid):
    return dict(request_id=rid, target_id=1, strategy='slow_down', anchors=[[0, 12], [0, 20], [0, 27], [0, 33]], duration=10, no_attack_reason=None)


@pytest.mark.parametrize('observed', [.1, .8, None, float('nan')])
def test_target_ignores_collision(observed):
    controller = DifficultyController({'mode': 'target'})
    yes = controller.update(.5, observed, True, .5, .1)
    no = controller.update(.5, observed, False, .5, .1)
    assert yes['after'] == no['after']
    assert yes['reason'] == no['reason']


def test_adaptive_collision_and_off():
    assert DifficultyController({'mode': 'adaptive'}).update(.5, .1, True, .5, .1)['after'] == .35
    assert DifficultyController({'mode': 'off'}).update(.5, .1, True, .5, .1)['after'] == .5


def test_weights_change_optimizer_fixed():
    controller = DifficultyController({'mode': 'target'})
    parameters = [controller.parameters(u / 100) for u in range(101)]
    assert len({p['llm_anchor'] for p in parameters}) == 101
    assert len({p['scenario_ttc'] for p in parameters}) == 101
    assert len({(p['inner_lr'], p['inner_beta'], p['n_guide_steps']) for p in parameters}) == 1


@pytest.mark.parametrize('value', [-1, 9, 2.5, float('nan')])
def test_invalid_fixed_optimizer(value):
    with pytest.raises(ValueError):
        DifficultyController({'fixed_optimizer': dict(inner_lr=.14, inner_beta=.35, n_guide_steps=value)})


@pytest.mark.parametrize('field,value', [('trajectory', []), ('request_id', 'bad'), ('duration', 0), ('anchors', [[0, 0]] * 30), ('target_id', True)])
def test_schema_rejects_wrong_id_dense_trajectory_and_invalid_fields(field, value):
    rid = uuid.uuid4().hex
    payload = plan(rid)
    payload[field] = value
    with pytest.raises(PlannerServiceFailure):
        validate_plan(payload, rid)


def test_queue_response_and_trace(tmp_path):
    client = CodexQueueClient(tmp_path / 'queue', timeout=2)
    def worker():
        for _ in range(100):
            pending = list(client.directory.glob('*/pending.json'))
            if pending:
                rid = pending[0].parent.name
                response = dict(request_id=rid, status='ok', exit_code=0, plan=plan(rid))
                temp = pending[0].parent / 'result.tmp'
                temp.write_text(json.dumps(response))
                temp.rename(temp.with_name('response.json'))
                return
            time.sleep(.01)
    thread = threading.Thread(target=worker)
    thread.start()
    assert client.request({'state': {}})['target_id'] == 1
    thread.join()
    assert client.last_trace['validation'] == 'schema_valid'


def test_timeout_is_service_failure(tmp_path):
    client = CodexQueueClient(tmp_path / 'queue', timeout=1)
    with pytest.raises(PlannerServiceFailure, match='timeout'):
        client.request({})
    assert client.last_trace['validation'] == 'planner_service_failure'


def test_smoke_fails_closed_without_attack_or_on_service_failure():
    with pytest.raises(RuntimeError, match='planner_service_failure'):
        validate_planner_smoke({'planner_service_failure_count': 1})
    with pytest.raises(RuntimeError, match='no_attack_execution'):
        validate_planner_smoke({})


def test_codex_plan_passes_existing_validator_and_converts_coordinates(tmp_path):
    from policies.llm_adversarial_planner import LLMAdversarialPlanner
    from types import SimpleNamespace
    planner = LLMAdversarialPlanner(provider='codex')
    rid = uuid.uuid4().hex
    payload = plan(rid)
    planner.client = SimpleNamespace(request=lambda context: payload,
                                    last_trace={'request_id': rid, 'validation': 'schema_valid', 'exit_code': 0})
    state = dict(ego_state=[10, 20, 0, 10, 1.5707963267948966], route=[[10, 20], [10, 120]],
                 agents=[dict(id=1, type=0, state=[10, 32, 0, 8, 1.5707963267948966])])
    actual = planner.generate_attack_plan(state, ['slow down'])
    assert actual['attack'] is True
    assert actual['anchors'][0] == [10., 32.]
    assert planner.last_trace['plan_validation'] == 'accepted'
    assert actual['duration'] == 10


def test_semantic_rejection_is_not_execution():
    from policies.llm_adversarial_planner import LLMAdversarialPlanner
    from types import SimpleNamespace
    planner = LLMAdversarialPlanner(provider='codex')
    payload = plan(uuid.uuid4().hex)
    payload['anchors'][0] = [0, 0]
    planner.client = SimpleNamespace(request=lambda context: payload, last_trace={})
    state = dict(ego_state=[0, 0, 0, 10, 1.5707963267948966], route=[[0, 0], [0, 100]],
                 agents=[dict(id=1, type=0, state=[0, 12, 0, 8, 1.5707963267948966])])
    assert planner.generate_attack_plan(state, []) is None
    assert planner.last_trace['plan_validation'] == 'plan_validation_failed'
    assert not planner.last_request_failed


def test_all_planners_reject_continuous_trajectory_fields():
    from policies.llm_adversarial_planner import LLMAdversarialPlanner
    with pytest.raises(ValueError, match='LLM'):
        LLMAdversarialPlanner._validate_attack_plan({'attack': False, 'trajectory': []}, {})


def test_nonfinite_history_serializes_as_missing():
    assert finite_context({'ttc': float('inf'), 'observed_difficulty': None}) == {'ttc': None, 'observed_difficulty': None}


@pytest.mark.parametrize('rid', ['../escape', '', 'x' * 32])
def test_request_rejects_bad_identifiers(rid):
    with pytest.raises(PlannerServiceFailure):
        validate_request({'request_id': rid, 'deadline': time.time()+100, 'context': {}})


def test_queue_claim_and_delivery_are_single_use(tmp_path):
    import os
    import subprocess
    import sys
    queue = tmp_path / 'queue'
    queue.mkdir(mode=0o700)
    rid = uuid.uuid4().hex
    folder = queue / rid
    folder.mkdir()
    request = dict(request_id=rid, deadline=time.time()+100, context={})
    (folder / 'pending.json').write_text(json.dumps(request))
    env = dict(os.environ, RISKWEAVER_CODEX_QUEUE=str(queue))
    script = Path(__file__).resolve().parents[1] / 'scripts/codex_queue.py'
    def run(action, payload=None):
        return subprocess.run([sys.executable, str(script), action], env=env,
                              input=payload, text=True, capture_output=True, timeout=10)
    assert json.loads(run('take').stdout)['request_id'] == rid
    assert json.loads(run('take').stdout) is None
    response = json.dumps(dict(request_id=rid, status='ok', exit_code=0, plan=plan(rid)))
    assert run('deliver', response).returncode == 0
    before = (folder / 'response.json').read_bytes()
    assert run('deliver', response).returncode != 0
    assert (folder / 'response.json').read_bytes() == before


def test_expired_request_fails():
    with pytest.raises(PlannerServiceFailure, match='expired'):
        validate_request(dict(request_id=uuid.uuid4().hex, deadline=time.time()-1, context={}))


def test_smoke_requires_matching_request_through_execution(tmp_path):
    rid = uuid.uuid4().hex
    trace = tmp_path / 'execution_trace.jsonl'
    video = tmp_path / 'video.mp4'
    video.write_bytes(b'fixture')
    records = [dict(kind='state', static_obstacles=[]),
               dict(kind='llm_input', state={'static_obstacles': []}),
               dict(kind='llm_output', request_failed=False,
                    planner_trace=dict(request_id=rid, exit_code=0, validation='schema_valid', plan_validation='accepted')),
               dict(kind='plan_accepted', request_id=rid),
               dict(kind='attack_execution', request_id=rid, executed=True)]
    result = dict(attack_plan_count=1, attack_executed_frames=1, obstacle_plan_count=0,
                  video_path=str(video), feasibility=dict(execution_trace_path=str(trace)))
    def save():
        trace.write_text('\n'.join(json.dumps(record) for record in records))
    save()
    assert validate_planner_smoke(result)['passed']
    records[-1]['request_id'] = uuid.uuid4().hex
    save()
    with pytest.raises(RuntimeError, match='missing_correlated'):
        validate_planner_smoke(result)
