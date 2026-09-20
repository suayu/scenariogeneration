"""独立核验真实规划、扩散和执行闭环；失败尝试仍保留全部证据。"""
import json
from pathlib import Path


def validate_planner_smoke(result):
    """通过编号关联证据，不能把计划存在或无碰撞误认为有效攻击。"""
    failures = []
    if result.get('planner_service_failure_count', 0):
        failures.append('planner_service_failure')
    if result.get('attack_plan_count', 0) <= 0:
        failures.append('no_accepted_attack')
    if result.get('attack_executed_frames', 0) <= 0:
        failures.append('no_attack_execution')
    if result.get('obstacle_plan_count', 0) != 0:
        failures.append('obstacle_plan')
    feasibility = result.get('feasibility', {})
    if feasibility.get('background_collision_frames', 0) or feasibility.get('background_static_collision_frames', 0):
        failures.append('background_collision')
    if result.get('generation_failure'):
        failures.append('generation_failure')
    trace = Path(feasibility.get('execution_trace_path', ''))
    video = Path(result.get('video_path', ''))
    if not video.is_file() or video.stat().st_size == 0:
        failures.append('missing_video')
    if not trace.is_file():
        failures.append('missing_trace')
    else:
        records = [json.loads(line) for line in trace.read_text(encoding='utf-8').splitlines()]
        inputs = [r for r in records if r.get('kind') == 'llm_input']
        states = [r for r in records if r.get('kind') == 'state']
        if not states or any(r.get('static_obstacles') != [] for r in states) or any(r.get('state', {}).get('static_obstacles') != [] for r in inputs):
            failures.append('static_obstacles_present_or_missing')
        valid = set()
        accepted = set()
        executed = set()
        for record in records:
            if record.get('kind') == 'llm_output':
                audit = record.get('planner_trace') or {}
                if record.get('request_failed') or audit.get('validation') == 'planner_service_failure':
                    failures.append('planner_service_failure')
                if audit.get('exit_code') == 0 and audit.get('validation') == 'schema_valid' and audit.get('plan_validation') == 'accepted':
                    valid.add(audit.get('request_id'))
            if record.get('kind') == 'plan_accepted':
                accepted.add(record.get('request_id'))
            if record.get('kind') == 'attack_execution' and record.get('executed') is True:
                executed.add(record.get('request_id'))
        if not ((valid & accepted & executed) - {None}):
            failures.append('missing_correlated_planner_diffusion_execution')
    if failures:
        raise RuntimeError('冒烟失败：' + ', '.join(sorted(set(failures))))
    return {'passed': True, 'attack_executed_frames': result['attack_executed_frames']}
