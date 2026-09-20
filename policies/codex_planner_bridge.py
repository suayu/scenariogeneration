"""本机 Codex 与远端仿真之间的受限文件队列契约；不传输登录凭据。"""
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import re
import time
import uuid

STRATEGIES = ['cut_in', 'hard_brake', 'slow_down', 'lane_change', 'occlusion', 'sudden_acceleration', 'others', 'none']
PLAN_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['request_id', 'target_id', 'strategy', 'anchors', 'duration', 'no_attack_reason'],
    'properties': {
        'request_id': {'type': 'string'},
        'target_id': {'type': ['integer', 'null']},
        'strategy': {'type': 'string', 'enum': STRATEGIES},
        'anchors': {'type': 'array', 'maxItems': 4, 'items': {
            'type': 'array', 'minItems': 2, 'maxItems': 2, 'items': {'type': 'number'}}},
        'duration': {'type': 'integer', 'minimum': 0, 'maximum': 10},
        'no_attack_reason': {'type': ['string', 'null']},
    },
}


class PlannerServiceFailure(RuntimeError):
    """服务、传输或响应契约失败，不能当作正常不攻击。"""


def validate_request(request):
    """验证队列信封，禁止路径注入、过期请求或未定义字段。"""
    if not isinstance(request, dict) or set(request) != {'request_id', 'deadline', 'context'}:
        raise PlannerServiceFailure('planner_service_failure: request_schema')
    if not isinstance(request['request_id'], str) or not re.fullmatch(r'[a-f0-9]{32}', request['request_id']):
        raise PlannerServiceFailure('planner_service_failure: request_id')
    if type(request['deadline']) not in (float, int) or not math.isfinite(request['deadline']) or request['deadline'] <= time.time():
        raise PlannerServiceFailure('planner_service_failure: expired_request')
    if not isinstance(request['context'], dict):
        raise PlannerServiceFailure('planner_service_failure: request_context')
    return request


def validate_plan(plan, request_id):
    """两端采用同一严格契约；语义和安全仍由远端既有验证器判断。"""
    if not isinstance(plan, dict) or set(plan) != set(PLAN_SCHEMA['required']):
        raise PlannerServiceFailure('planner_service_failure: schema_fields')
    if plan['request_id'] != request_id or not re.fullmatch(r'[a-f0-9]{32}', request_id):
        raise PlannerServiceFailure('planner_service_failure: request_id')
    if plan['strategy'] not in STRATEGIES or type(plan['duration']) is not int:
        raise PlannerServiceFailure('planner_service_failure: strategy_duration')
    anchors = plan['anchors']
    if not isinstance(anchors, list) or len(anchors) not in (0, 4):
        raise PlannerServiceFailure('planner_service_failure: sparse_anchors')
    for point in anchors:
        if not isinstance(point, list) or len(point) != 2 or any(
                type(v) not in (int, float) or not math.isfinite(v) for v in point):
            raise PlannerServiceFailure('planner_service_failure: anchor_coordinates')
    if plan['target_id'] is None:
        if anchors or plan['strategy'] != 'none' or plan['duration'] != 0 or not isinstance(plan['no_attack_reason'], str) or not plan['no_attack_reason'].strip():
            raise PlannerServiceFailure('planner_service_failure: no_attack_contract')
    elif type(plan['target_id']) is not int or plan['target_id'] < 0 or len(anchors) != 4 or not 1 <= plan['duration'] <= 10 or plan['strategy'] == 'none' or plan['no_attack_reason'] is not None:
        raise PlannerServiceFailure('planner_service_failure: attack_contract')
    return plan


def write_json_exclusive(path, value):
    """独占写入，发布由调用方原子重命名，拒绝覆盖历史请求。"""
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False)


def finite_context(value):
    """历史统计中的无穷 TTC 表示缺失；在 JSON 输入中显式传 null。"""
    if isinstance(value, dict):
        return {key: finite_context(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_context(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value) if math.isfinite(value) else None
    raise PlannerServiceFailure('planner_service_failure: unsupported_context_type')


class CodexQueueClient:
    def __init__(self, directory=None, timeout=None):
        self.directory = Path(directory or os.environ.get('RISKWEAVER_CODEX_QUEUE', '/home2/zhaoyx/.local/state/riskweaver/codex_queue'))
        self.timeout = float(timeout or os.environ.get('RISKWEAVER_CODEX_TIMEOUT', '180'))
        if not math.isfinite(self.timeout) or not 1 <= self.timeout <= 600:
            raise ValueError('Codex timeout 必须在 1–600 秒内')
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name != 'nt' and (self.directory.stat().st_uid != os.getuid() or self.directory.stat().st_mode & 0o077):
            raise ValueError('Codex 队列必须由当前用户独占，权限为 0700')
        self.last_trace = None

    def request(self, context):
        context = finite_context(context)
        request_id = uuid.uuid4().hex
        start = time.monotonic()
        folder = self.directory / request_id
        folder.mkdir(mode=0o700)
        request = {'request_id': request_id, 'deadline': time.time() + self.timeout,
                   'context': context}
        validate_request(request)
        self.last_trace = {'request_id': request_id, 'input_sha256': hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest(),
                           'validation': 'pending', 'exit_code': None}
        write_json_exclusive(folder / 'request.tmp', request)
        (folder / 'request.tmp').rename(folder / 'pending.json')
        try:
            while time.monotonic() - start < self.timeout:
                response_path = folder / 'response.json'
                if response_path.exists():
                    response = json.loads(response_path.read_text(encoding='utf-8'))
                    self.last_trace['exit_code'] = response.get('exit_code')
                    if response.get('request_id') != request_id or response.get('status') != 'ok' or response.get('exit_code') != 0:
                        raise PlannerServiceFailure('planner_service_failure: response_status')
                    plan = validate_plan(response.get('plan'), request_id)
                    self.last_trace.update(validation='schema_valid', output_summary={k: plan[k] for k in ('target_id', 'strategy', 'duration')})
                    return plan
                time.sleep(.2)
            raise PlannerServiceFailure('planner_service_failure: timeout')
        except (OSError, ValueError) as error:
            raise PlannerServiceFailure('planner_service_failure: transport_json') from error
        finally:
            self.last_trace['elapsed_seconds'] = time.monotonic() - start
            if self.last_trace['validation'] == 'pending':
                self.last_trace['validation'] = 'planner_service_failure'
            write_json_exclusive(folder / 'client_audit.json', self.last_trace)
