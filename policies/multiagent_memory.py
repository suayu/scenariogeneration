"""离线复盘生成冻结记忆；数值结论保留原始证据，禁止评估集泄漏。"""
import json
import time
import uuid
from pathlib import Path

from policies.multiagent_planner import fingerprint


def load_records(paths, excluded_scene_ids=()):
    """加载全部成功和失败尝试；发现评估场景交集时直接拒绝建库。"""
    records, seen = [], set()
    excluded = set(map(str, excluded_scene_ids))
    for path in paths:
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            required = {'record_id', 'policy_id', 'scene_id', 'scene_conditions',
                        'actual_attack_execution_frames', 'D', 'generation_failure_reason'}
            if not required <= row.keys():
                raise ValueError('incomplete strategy return record')
            if str(row['scene_id']) in excluded:
                raise ValueError('offline memory overlaps evaluation scenes')
            if row['record_id'] not in seen:
                seen.add(row['record_id'])
                records.append(row)
    if not records:
        raise ValueError('offline reflection requires historical attempts')
    return records


def reflect(records, request):
    """一次有界 LLM 复盘；摘要只能引用当前输入中的证据记录。"""
    request_id = uuid.uuid4().hex
    payload = dict(request_id=request_id, records=records,
                   task='Summarize reusable lessons about strategy selection. Distinguish service, validation, diffusion and execution failures. Missing D is unknown, never zero. Do not alter safety limits or generate controls.',
                   schema={'request_id': 'repeat exactly', 'lessons': [
                       {'evidence_ids': ['existing record_id'], 'lesson': 'brief evidence-grounded advice'}]})
    started = time.monotonic()
    response = request(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    elapsed = time.monotonic()-started
    if not isinstance(response, dict) or set(response) != {'request_id', 'lessons'} or response['request_id'] != request_id:
        raise ValueError('reflection schema or request_id mismatch')
    lessons = response['lessons']
    if not isinstance(lessons, list) or not 1 <= len(lessons) <= 20:
        raise ValueError('reflection lesson count invalid')
    by_id = {row['record_id']: row for row in records}
    indexed = []
    for item in lessons:
        if not isinstance(item, dict) or set(item) != {'evidence_ids', 'lesson'}:
            raise ValueError('reflection lesson schema invalid')
        ids, lesson = item['evidence_ids'], item['lesson']
        if not isinstance(ids, list) or not ids or any(not isinstance(i, str) or i not in by_id for i in ids):
            raise ValueError('reflection cites unknown evidence')
        if not isinstance(lesson, str) or not 1 <= len(lesson.strip()) <= 2000:
            raise ValueError('reflection lesson text invalid')
        # 一条经验可按多种条件索引，但绝不能改变或替代源记录中的实际指标。
        for record_id in dict.fromkeys(ids):
            row = by_id[record_id]
            indexed.append(dict(policy_id=row['policy_id'], scene_conditions=row['scene_conditions'],
                                scene_id=row['scene_id'], evidence_id=record_id, lesson=lesson,
                                strategy=row.get('strategy'), D=row['D'],
                                actual_attack_execution_frames=row['actual_attack_execution_frames'],
                                generation_failure_reason=row['generation_failure_reason']))
    return dict(schema_version=1, source_hash=fingerprint(records), source_records=records,
                records=indexed, reflection_request_id=request_id,
                reflection_elapsed_seconds=elapsed, source_scene_ids=sorted({str(r['scene_id']) for r in records}),
                interpretation='LLM lessons are hypotheses; measured values remain authoritative')


def save_memory(bank, destination):
    """只创建新文件，防止覆盖已冻结的实验知识库。"""
    with Path(destination).open('x', encoding='utf-8') as handle:
        json.dump(bank, handle, ensure_ascii=False, indent=2, allow_nan=False)
