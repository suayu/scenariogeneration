"""有界多大模型协作；连续轨迹与安全裁决仍由既有下游负责。"""
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import uuid


MODES = ('single', 'conditional_critic', 'parallel_arbiter', 'parallel_memory')


class AuditedClient:
    """仅记录请求计量，不保存鉴权、响应头或异常正文。"""
    def __init__(self, client, events):
        def wrap(create):
            def request(**kwargs):
                started = time.monotonic()
                event = dict(model=kwargs.get('model'), status='started')
                events.append(event)
                try:
                    response = create(**kwargs)
                    usage = getattr(response, 'usage', None)
                    event.update(status='ok', usage=usage.model_dump() if hasattr(usage, 'model_dump') else None)
                    return response
                except Exception as error:
                    event.update(status='failed', error_type=type(error).__name__,
                                 status_code=getattr(error, 'status_code', None))
                    raise
                finally:
                    event['elapsed_seconds'] = time.monotonic() - started
            return request
        if hasattr(client, 'chat'):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=wrap(client.chat.completions.create)))
        if hasattr(client, 'responses'):
            self.responses = SimpleNamespace(create=wrap(client.responses.create))


def fingerprint(value):
    """只对非敏感规划内容计算稳定摘要，供并行请求和记忆审计使用。"""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


class MultiAgentPlanner:
    """复用原规划器验证结果，以有限调用数实现四组实验。"""

    def __init__(self, settings):
        self.mode = str(getattr(settings, 'mode', 'single'))
        if self.mode not in MODES:
            raise ValueError('unknown multiagent mode: ' + self.mode)
        self.timeout = float(getattr(settings, 'timeout_seconds', 60.0))
        if not 0 < self.timeout <= 300:
            raise ValueError('multiagent timeout must be in (0, 300]')
        self.memory_path = str(getattr(settings, 'memory_path', '') or '')
        self.memory = None
        if self.mode == 'parallel_memory':
            if not self.memory_path:
                raise ValueError('parallel_memory requires a frozen offline memory bank')
            self.memory = json.loads(Path(self.memory_path).read_text(encoding='utf-8'))
            if self.memory.get('schema_version') != 1 or not isinstance(self.memory.get('records'), list):
                raise ValueError('invalid offline memory bank')
        self.last_trace = None

    def retrieve(self, context, planning_regime='profile'):
        """只检索同一规划模式和策略身份的历史，避免数据泄漏。"""
        if self.memory is None:
            return []
        # 旧库只来自画像模式；无画像组必须使用单独冻结的训练库。
        bank_regime = self.memory.get('planning_regime', 'profile')
        if bank_regime != planning_regime:
            raise ValueError('offline memory planning regime mismatch')
        scene_id = str((context or {}).get('scene_id', ''))
        if scene_id and scene_id in self.memory.get('source_scene_ids', []):
            raise ValueError('offline memory overlaps current evaluation scene')
        policy_id = (context or {}).get('policy_id')
        if not policy_id:
            return []
        condition = (context or {}).get('scene_conditions')
        records = [r for r in self.memory['records'] if r.get('policy_id') == policy_id]
        records.sort(key=lambda r: r.get('scene_conditions') != condition)
        return copy.deepcopy(records[:5])

    def run(self, planner, env_state, instruction, planner_kwargs=None, context=None, scene_image=None,
            memory_context=None):
        """关闭时直接调用原函数；开启时隔离角色状态并保留完整失败统计。"""
        from policies.profile_planner import rank_profile_candidates
        kwargs = dict(planner_kwargs or {})
        if self.mode == 'single':
            if context is not None:
                return rank_profile_candidates(planner, env_state, instruction, context, scene_image)
            return planner.generate_attack_plan(env_state, instruction, **kwargs)

        started = time.monotonic()
        if self.mode == 'parallel_memory' and context is None and memory_context is None:
            raise ValueError('parallel_memory requires scene-conditioned memory context')
        if getattr(planner, 'provider', None) == 'codex':
            raise ValueError('multiagent currently requires an API backend with bounded request timeout')
        request_id = uuid.uuid4().hex
        memory = self.retrieve(memory_context if memory_context is not None else context,
                               'profile' if context is not None else 'unprofiled')
        catalog_hash = fingerprint((context or {}).get('candidates', env_state.get('agents', [])))
        events = []

        def call(role, proposals=()):
            # 每个角色独享跟踪和模型选择状态；API 客户端派生出有限超时且不重试的连接配置。
            worker = copy.copy(planner)
            worker.last_trace = None
            worker.last_request_failed = False
            client = getattr(worker, 'client', None)
            api_calls = []
            if hasattr(client, 'with_options'):
                worker.client = AuditedClient(client.with_options(timeout=self.timeout, max_retries=0), api_calls)
            if hasattr(worker, 'model_name'):
                # 固定本轮模型，不因免费额度耗尽而轮换到其他可能收费的模型。
                worker.model_names = (worker.model_name,)
                worker._preferred_model_index = 0
            role_id = request_id + ':' + role
            message = dict(request_id=role_id, role=role, candidate_catalog_hash=catalog_hash,
                           proposals=list(proposals), offline_memory=memory,
                           instruction={
                               'baseline': 'Use the original unmodified planning request.',
                               'proposer_risk': 'Prioritize evidence-backed ego vulnerability and low reaction margin.',
                               'proposer_feasibility': 'Independently prioritize executable interaction, non-target background safety, and positive theoretical ego drivable area; no explicit avoidance witness is required.',
                               'critic': 'Review the proposal for concrete contradictions; retain it if no supported objection exists. Return the best valid plan using the original schema.'
                           }[role])
            call_start = time.monotonic()
            try:
                if context is not None:
                    role_context = copy.deepcopy(context)
                    if role == 'baseline':
                        # 这些字段由协作入口补充，仅供记忆隔离使用，不进入基线提示。
                        for metadata_key in ('policy_id', 'scene_id', 'scene_conditions'):
                            role_context.pop(metadata_key, None)
                    if role != 'baseline':
                        role_context['collaboration'] = message
                    plan = rank_profile_candidates(worker, env_state, instruction, role_context, scene_image)
                else:
                    role_instructions = list(instruction or [])
                    if role != 'baseline':
                        role_instructions.append(json.dumps(message, ensure_ascii=False, allow_nan=False))
                    plan = worker.generate_attack_plan(env_state, role_instructions, **kwargs)
                service_failure = bool(getattr(worker, 'last_request_failed', False))
                event = dict(role=role, request_id=role_id, plan=plan,
                             service_failure=service_failure, trace=getattr(worker, 'last_trace', None))
            except Exception as error:
                # 只记录异常类型，避免 SDK 异常正文带出服务端敏感信息。
                event = dict(role=role, request_id=role_id, plan=None,
                             service_failure=True, error_type=type(error).__name__)
            event['elapsed_seconds'] = time.monotonic() - call_start
            event['api_calls'] = api_calls
            return event

        # 无候选时沿用原函数的明确不攻击结果，避免无意义的额外调用。
        if context is not None and not context.get('candidates') and not context.get('obstacle_candidates'):
            result = rank_profile_candidates(planner, env_state, instruction, context, scene_image)
            self.last_trace = dict(mode=self.mode, request_id=request_id, events=[],
                                   elapsed_seconds=time.monotonic()-started, skipped='no_feasible_candidates')
            return result

        if self.mode == 'conditional_critic':
            events.append(call('baseline'))
            first = events[0]
            # 用可复现的输入证据触发质疑，不把模型自报信心当概率。
            history = (context or {}).get('history', [])
            reasons = []
            if first['plan'] is None:
                reasons.append('invalid_plan')
            if not history or (isinstance(history, dict) and history.get('attempts', 0) == 0):
                reasons.append('missing_history')
            if len((context or {}).get('candidates', [])) > 1:
                reasons.append('multiple_candidates')
            if reasons and not first['service_failure']:
                events.append(call('critic', [first['plan']]))
            selected = events[-1]['plan']
        else:
            # 两次独立提案并行执行，归并顺序固定以免线程完成顺序影响结果。
            with ThreadPoolExecutor(max_workers=2) as pool:
                jobs = [pool.submit(call, role) for role in ('proposer_risk', 'proposer_feasibility')]
                events = [job.result() for job in jobs]
            reasons = []
            valid = [e for e in events if e['plan'] is not None]
            # 采用候选排序的 Borda 分数；只从已通过原验证器的提案中选择。
            scores = {}
            for event in valid:
                ranking = (event.get('trace') or {}).get('candidate_ranking', [])
                for index, candidate_id in enumerate(ranking):
                    scores[candidate_id] = scores.get(candidate_id, 0) + len(ranking)-index
            arbitration_reason = 'profile_candidate_borda'
            if context is None:
                # 无画像模式没有结构化候选分；可行性角色明确不攻击时保守否决。
                veto = next((e for e in valid if e['role']=='proposer_feasibility'
                             and not e['plan'].get('attack')), None)
                if veto is not None:
                    selected = veto['plan']
                    arbitration_reason = 'feasibility_veto'
                else:
                    attacks = [e for e in valid if e['plan'].get('attack')]
                    selected = sorted(attacks or valid, key=lambda e:e['role'])[0]['plan'] if valid else None
                    arbitration_reason = 'validated_attack_preferred' if attacks else 'no_valid_attack'
            else:
                valid.sort(key=lambda e: (-scores.get(e['plan'].get('candidate_id'), 0), e['role']))
                selected = valid[0]['plan'] if valid else None
        # 任一服务失败均使本次协作失败，不能用另一提案掩盖免费额度耗尽。
        failed = any(e['service_failure'] for e in events)
        if failed:
            selected = None
        planner.last_request_failed = failed
        # 对外保留每个角色的明确意见，避免只记录最终计划而丢失分歧依据。
        agent_opinions = []
        for event in events:
            plan = event.get('plan') or {}
            agent_opinions.append(dict(
                role=event['role'], service_failure=bool(event['service_failure']),
                decision='attack' if plan.get('attack') else ('no_attack' if plan else 'invalid'),
                target_id=plan.get('attack_target_id'), strategy=plan.get('strategy'),
                reason=plan.get('reason') or event.get('error_type') or 'no_reason_returned'))
        selected_reason = (selected or {}).get('reason') if isinstance(selected, dict) else None
        if failed:
            arbitration_detail = 'At least one role had a planner service failure; no proposal may mask it.'
        elif self.mode == 'conditional_critic':
            arbitration_detail = ('The critic was invoked because: ' + ', '.join(reasons)
                                  + '. The final valid critic result replaced the baseline.'
                                  if len(events) > 1 else
                                  'No deterministic critic trigger fired; the validated baseline was retained.')
        elif arbitration_reason == 'feasibility_veto':
            arbitration_detail = ('The feasibility role explicitly returned no attack, so its veto was selected. '
                                  f'Reason: {selected_reason or "no reason returned"}')
        elif arbitration_reason == 'validated_attack_preferred':
            arbitration_detail = ('No feasibility veto was present; arbitration selected a validated attack '
                                  'from the fixed role order after original schema validation.')
        else:
            arbitration_detail = ('Profile candidates were aggregated using Borda scores over each role ranking; '
                                  'the highest-scoring validated catalog candidate was selected.')
        memory_conclusions = [dict(evidence_id=row.get('evidence_id'), scene_id=row.get('scene_id'),
                                   conclusion=row.get('lesson')) for row in memory]
        self.last_trace = dict(mode=self.mode, request_id=request_id,
                               candidate_catalog_hash=catalog_hash, events=events,
                               agent_opinions=agent_opinions,
                               critic_triggers=reasons, selected_plan=selected,
                               arbitration_reason=arbitration_reason if self.mode != 'conditional_critic' else None,
                               arbitration_detail=arbitration_detail,
                               memory_hash=fingerprint(self.memory) if self.memory else None,
                               memory_hits=len(memory), memory_conclusions=memory_conclusions,
                               elapsed_seconds=time.monotonic()-started,
                               plan_validation='planner_service_failure' if failed else
                               ('accepted' if selected and selected.get('attack') else 'no_attack_or_invalid'))
        planner.last_trace = self.last_trace
        return selected
