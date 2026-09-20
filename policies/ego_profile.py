"""全程自车画像、场景条件化候选及可审计的攻击知识库。"""
import hashlib
import json
import math
from pathlib import Path
import uuid

import numpy as np

from policies.joint_safety import obb_clearance
from policies.collision_ttc import minimum_obb_ttc


def finite(value):
    """未知值保留为空，禁止用零填充缺失风险。"""
    try:
        return float(value) if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def clean_json(value):
    """递归保留空测量，收益表不得写入 JSON 的 NaN/Infinity 扩展。"""
    if isinstance(value, dict):
        return {k:clean_json(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.ndarray,np.generic)):
        return clean_json(value.tolist())
    if isinstance(value,float):
        return finite(value)
    return value


def obstacle_box(obstacle):
    return np.array([*obstacle['center'],obstacle['yaw'],obstacle['length'],obstacle['width']],float)


class OnlineStatistic:
    """均匀蓄水池估计全程分位数；分块均值区间缓解帧间相关性。"""
    def __init__(self, alpha=.05, capacity=2048, block_size=20):
        if not 0 < alpha <= 1 or capacity < 32 or block_size < 2:
            raise ValueError("画像 alpha/capacity/block_size 越界")
        self.alpha, self.capacity, self.block_size = alpha, capacity, block_size
        self.n = 0
        self.mean = self.ewma = self.ewvar = 0.
        self.samples, self.block = [], []
        self.blocks = []
        # 使用独立随机数，不能改变扩散采样或场景随机种子。
        self.rng = np.random.default_rng(731)

    def add(self, value):
        value = finite(value)
        if value is None:
            return
        self.n += 1
        self.mean += (value - self.mean) / self.n
        delta = value - self.ewma
        self.ewma = value if self.n == 1 else self.ewma + self.alpha * delta
        self.ewvar = 0. if self.n == 1 else (1-self.alpha)*(self.ewvar+self.alpha*delta*delta)
        if len(self.samples) < self.capacity:
            self.samples.append(value)
        else:
            i = int(self.rng.integers(self.n))
            if i < self.capacity:
                self.samples[i] = value
        self.block.append(value)
        if len(self.block) >= self.block_size:
            self.end_block()

    def end_block(self):
        if self.block:
            self.blocks.append(float(np.mean(self.block)))
            self.blocks = self.blocks[-self.capacity:]
            self.block = []

    def snapshot(self):
        ci = None
        if len(self.blocks) >= 8:
            # 分块自助法仅给均值区间；不冒称分位数或独立场景的置信区间。
            rng = np.random.default_rng(419)
            values = np.asarray(self.blocks)
            means = np.mean(rng.choice(values, (256, len(values))), axis=1)
            ci = np.quantile(means, [.025, .975]).tolist()
        return dict(n=self.n, mean=self.mean if self.n else None,
                    quantiles=dict(zip(("q10", "q50", "q90"), np.quantile(self.samples, [.1,.5,.9]).tolist())) if self.n else None,
                    ewma=self.ewma if self.n else None, ew_std=math.sqrt(self.ewvar) if self.n else None,
                    block_mean_ci95=ci, ci_blocks=len(self.blocks),
                    ci_method="recent_nonoverlapping_block_mean_bootstrap; residual dependence possible")


def project(points, line):
    """连续线段投影距离，避免用路点序号冒充路线进度。"""
    points, line = np.asarray(points, float), np.asarray(line, float)
    if line.ndim != 2 or len(line) < 2:
        return None
    starts, delta = line[:-1,:2], np.diff(line[:,:2], axis=0)
    length2 = (delta*delta).sum(axis=1)
    valid = length2 > 1e-8
    if not valid.any():
        return None
    starts, delta, length2 = starts[valid], delta[valid], length2[valid]
    fraction = np.clip(((points[...,None,:]-starts)*delta).sum(-1)/length2, 0, 1)
    return np.min(np.linalg.norm(points[...,None,:]-starts-fraction[...,None]*delta, axis=-1), axis=-1)


def scene_conditions(env):
    ego = np.asarray(env.ego_state, float)
    agents = np.asarray(env.data_dict['agent'][-1], float)[np.asarray(env.agent_active, bool)]
    near = np.linalg.norm(agents[:,:2]-ego[:2],axis=1) < 35 if len(agents) else []
    route = np.asarray(env.scenario_dict['route'], float)
    near_route = route[np.linalg.norm(route[:,:2]-ego[:2], axis=1) < 30]
    delta = np.diff(near_route[:,:2],axis=0)
    delta = delta[np.linalg.norm(delta,axis=1)>1e-3]
    turns = np.diff(np.unwrap(np.arctan2(delta[:,1],delta[:,0]))) if len(delta)>1 else []
    return dict(speed_bin="low" if np.linalg.norm(ego[2:4])<5 else "medium" if np.linalg.norm(ego[2:4])<15 else "high",
                density_bin="sparse" if sum(near)<3 else "dense",
                road_shape="curved" if len(turns) and max(abs(np.asarray(turns)))>.15 else "straight")


class EgoProfile:
    def __init__(self, settings):
        self.settings = settings
        self.stats, self.conditioned = {}, {}
        self.frames = self.attack_frames = self.episodes = 0
        self.run_id = uuid.uuid4().hex[:12]
        self.previous = self.last_key = None

    def reset_episode(self):
        # 累积画像不清空，差分计算不得跨场景连接。
        for stat in list(self.stats.values()) + [v for d in self.conditioned.values() for v in d.values()]:
            stat.end_block()
        self.previous = self.last_key = None
        self.episodes += 1

    def observe(self, env, attack=False):
        key = (self.episodes, int(env.current_step))
        if key == self.last_key:
            return
        self.last_key = key
        ego = np.asarray(env.ego_state, float)
        speed = float(np.linalg.norm(ego[2:4]))
        dt = float(env.dt)
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError("画像需要有效仿真 dt")
        metrics = dict(speed_mps=speed, route_error_m=finite(project(ego[:2], env.scenario_dict['route'])))
        agents = np.asarray(env.data_dict['agent'][-1],float)
        active = np.asarray(env.agent_active,bool)
        metrics['min_ttc_s'] = finite(minimum_obb_ttc(ego,agents,active))
        forward = np.array([math.cos(ego[4]),math.sin(ego[4])])
        side = np.array([-forward[1],forward[0]])
        metrics['lateral_speed_mps'] = float(ego[2:4]@side)
        nearby = agents[active]
        if len(nearby):
            relative = nearby[:,:2]-ego[:2]
            front = (relative@forward>0)&(np.abs(relative@side)<2.)
            if front.any():
                gap = np.min((relative@forward)[front]-(nearby[front,5]+ego[5])/2)
                metrics['following_gap_m'] = float(gap)
                metrics['time_headway_s'] = float(gap/speed) if speed>.1 else None
        if self.previous is not None:
            steps = int(env.current_step)-self.previous['step']
            elapsed = steps*dt
            if elapsed > 0:
                accel = (speed-self.previous['speed'])/elapsed
                metrics['acceleration_mps2'] = accel
                metrics['yaw_rate_rps'] = math.atan2(math.sin(ego[4]-self.previous['yaw']),math.cos(ego[4]-self.previous['yaw']))/elapsed
                if self.previous.get('accel') is not None:
                    metrics['jerk_mps3'] = (accel-self.previous['accel'])/elapsed
        self.previous = dict(step=int(env.current_step), speed=speed, yaw=float(ego[4]), accel=metrics.get('acceleration_mps2'))
        self.frames += 1
        self.attack_frames += int(attack)
        condition = json.dumps(scene_conditions(env),sort_keys=True)
        stores = [self.stats, self.conditioned.setdefault(condition,{})]
        for store in stores:
            for name,value in metrics.items():
                if name not in store:
                    store[name] = OnlineStatistic(float(getattr(self.settings,'alpha',.05)), int(getattr(self.settings,'reservoir_size',2048)), int(getattr(self.settings,'block_frames',20)))
                store[name].add(value)

    def snapshot(self, condition):
        return dict(version=f"ego-v1-{self.run_id}-e{self.episodes}-f{self.frames}", frames=self.frames,
                    attack_frames=self.attack_frames, nonattack_frames=self.frames-self.attack_frames,
                    metrics={k:v.snapshot() for k,v in self.stats.items()},
                    conditioned_metrics={k:v.snapshot() for k,v in self.conditioned.get(json.dumps(condition,sort_keys=True),{}).items()})


class AttackKnowledge:
    """只追加记录；按自车标识和场景条件隔离，失败与空测量参与覆盖率统计。"""
    def __init__(self, policy_id, read_paths=()):
        self.policy_id, self.rows, self.seen = policy_id, [], set()
        self.path = None
        for path in read_paths:
            with Path(path).open(encoding='utf-8') as handle:
                for line in handle:
                    row = json.loads(line)
                    if row.get('policy_id') == policy_id and row.get('schema_version') == 1:
                        self._remember(row)

    def _remember(self, row):
        required = {'record_id','scene_conditions','strategy','profile_version','target','u','actual_attack_execution_frames','D','drivable_area_change_ratio','generation_failure_reason','reward'}
        if not required <= set(row) or not isinstance(row['scene_conditions'],dict):
            raise ValueError('攻击知识库记录缺少必需字段')
        for key in ('D','reward'):
            if row[key] is not None and (finite(row[key]) is None or not 0<=float(row[key])<=1):
                raise ValueError('攻击知识库风险或收益字段无效')
        if row['record_id'] not in self.seen:
            self.seen.add(row['record_id'])
            self.rows.append(row)

    def append(self, row):
        row = clean_json(dict(row, schema_version=1, policy_id=self.policy_id, record_id=uuid.uuid4().hex))
        if self.path is not None:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            with self.path.open('a',encoding='utf-8') as handle:
                handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        self._remember(row)

    def retrieve(self, condition, strategy=None):
        rows = [r for r in self.rows if r['scene_conditions'] == condition and (strategy is None or r['strategy']==strategy)]
        valid = [r for r in rows if r.get('reward') is not None]
        # 中性先验收缩；少量偶然成功不会成为确定的“脆弱点”。
        values = [r['reward'] for r in valid]
        mean = (sum(values)+1.)/(len(values)+2)
        radius = min(1., math.sqrt(math.log(40)/(2*len(values)))) if values else 1.
        center = float(np.mean(values)) if values else .5
        return dict(attempts=len(rows), measured_executions=len(valid),
                    failure_count=sum(bool(r.get('generation_failure_reason')) for r in rows),
                    vulnerability_estimate=mean, uncertainty_interval=[max(0.,center-radius),min(1.,center+radius)],
                    estimator="heuristic shrinkage mean; Hoeffding interval assumes independent bounded returns",
                    recent_examples=rows[-5:])


def _box(state, positions, yaw=None):
    positions = np.asarray(positions,float)
    return np.concatenate((positions,np.broadcast_to(np.asarray([state[4] if yaw is None else yaw,state[5],state[6]]),positions.shape[:-1]+(3,))),axis=-1)


class CandidateBuilder:
    """规则生成稀疏意图；预测筛查不代替扩散联合筛查与执行前 OBB 门禁。"""
    def __init__(self, settings):
        self.margin = float(getattr(settings,'background_margin_m',.5))
        self.half_width = float(getattr(settings,'lane_half_width_m',1.8))
        self.ttc_range = tuple(getattr(settings,'ttc_range_s',(1.,4.)))
        # 默认无低速模板；实验显式启用时仍须经过完整道路、背景和规避筛查。
        self.low_speed_decelerations = tuple(float(x) for x in getattr(settings,'low_speed_decelerations_mps2',()))
        if self.margin < 0 or self.half_width <= 0 or len(self.ttc_range)!=2 or not 0 < self.ttc_range[0] < self.ttc_range[1]:
            raise ValueError("画像候选安全边界或 TTC 区间无效")
        if any(not np.isfinite(x) or not 0 < x <= 2 for x in self.low_speed_decelerations):
            raise ValueError('低速候选减速度必须在 (0, 2] m/s²')

    def build(self, env, profile, knowledge):
        ego = np.asarray(env.ego_state,float)
        states = np.asarray(env.data_dict['agent'][-1],float)
        active = np.flatnonzero(env.agent_active)
        if len(ego)<7 or states.shape[1]<7:
            return [], {'missing_vehicle_geometry':1}
        if not np.isfinite(ego[:7]).all() or not np.isfinite(states[active,:7]).all() or np.any(states[active,5:7]<=0) or np.any(ego[5:7]<=0):
            return [], {'invalid_vehicle_geometry':1}
        lanes = [np.asarray(line)[:,:2] for line in env.scenario_dict.get('lanes',[]) if np.asarray(line).ndim==2 and len(line)>1 and np.isfinite(line).all() and np.any(np.linalg.norm(np.diff(np.asarray(line)[:,:2],axis=0),axis=1)>1e-3)]
        if not lanes:
            return [], {'missing_road_geometry':1}
        condition = scene_conditions(env)
        forward = np.array([math.cos(ego[4]),math.sin(ego[4])])
        lateral = np.array([-forward[1],forward[0]])
        proposals, rejected = [], {}
        # 一秒间隔四锚点描述意图，连续轨迹仍由 diffusion 生成。
        t = np.arange(4,dtype=float)
        for target in active:
            state = states[target]
            relative = state[:2]-ego[:2]
            longitudinal,side = relative@forward, relative@lateral
            if not 5 < longitudinal < 45 or abs(side)>7 or abs(math.atan2(math.sin(state[4]-ego[4]),math.cos(state[4]-ego[4])))>.7:
                continue
            speed = np.linalg.norm(state[2:4])
            proposals_for_target = []
            if abs(side)<1.5:
                decelerations = (1.,2.) + (self.low_speed_decelerations if speed < 3. else ())
                for decel in dict.fromkeys(decelerations):
                    distance = speed*t-.5*decel*t*t
                    if speed-decel*3 >= 0:
                        proposals_for_target.append(('slow_down','forward_pressure',state[:2]+distance[:,None]*forward))
            else:
                # 平滑横向位移是稀疏候选模板，并非交给执行器的连续控制。
                blend = (t/3)**2*(3-2*t/3)
                anchors = state[:2]+t[:,None]*state[2:4]-blend[:,None]*side*lateral
                proposals_for_target.append(('cut_in','cut_in',anchors))
                heading_delta = abs(math.atan2(math.sin(state[4]-ego[4]),math.cos(state[4]-ego[4])))
                if heading_delta>.12:
                    proposals_for_target.append(('lane_change','merge_conflict',anchors))
            for strategy,family,anchors in proposals_for_target:
                estimates,reason = self.screen(env,int(target),anchors,lanes)
                if reason:
                    rejected[reason] = rejected.get(reason,0)+1
                    continue
                prior = knowledge.retrieve(condition,family)
                stats = profile.get('conditioned_metrics',{})
                yaw = stats.get('yaw_rate_rps',{})
                brake = stats.get('acceleration_mps2',{})
                # 有证据才使用行为匹配；冷启动给中性值并明示不确定性。
                metric = brake if family=='forward_pressure' else yaw
                match = .5
                if metric.get('n',0)>=20:
                    match = float(np.clip(.5+(.15*float(metric['ewma']) if family=='forward_pressure' else -.5*abs(float(metric['ewma']))),0,1))
                proposals.append(dict(candidate_id=f"c{len(proposals)}",target_id=int(target),strategy=strategy,
                                      attack_family=family,anchors=anchors.tolist(),duration=10,
                                      scene_conditions=condition,profile_match=match,profile_evidence_frames=metric.get('n',0),
                                      vulnerability=prior,**estimates))
        return proposals,rejected

    def screen(self, env, target, anchors, lanes):
        """筛查交互、动力学与非目标背景安全；攻击车不受道路拓扑约束。"""
        states = np.asarray(env.data_dict['agent'][-1],float)
        state,ego = states[target],np.asarray(env.ego_state,float)
        anchors = np.asarray(anchors,float)
        if anchors.shape!=(4,2) or not np.isfinite(anchors).all():
            return None,'invalid_anchors'
        velocities = np.diff(anchors,axis=0)
        if np.linalg.norm(anchors[0]-state[:2])>.01 or np.max(np.linalg.norm(velocities,axis=1))>35 or np.max(np.linalg.norm(np.diff(velocities,axis=0),axis=1))>4 or np.linalg.norm(velocities[0]-state[2:4])>4:
            return None,'dynamics'
        times = np.linspace(0,3,61)
        pos = np.column_stack([np.interp(times,np.arange(4),anchors[:,i]) for i in range(2)])
        segment = np.minimum(times.astype(int),2)
        yaw = np.arctan2(velocities[segment,1],velocities[segment,0])
        boxes = np.column_stack((pos,yaw,np.full(len(times),state[5]),np.full(len(times),state[6])))
        clearance = float('inf')
        for other in np.flatnonzero(env.agent_active):
            if other == target:
                continue
            s = states[other]
            distance = float(np.min(obb_clearance(boxes,_box(s,s[:2]+times[:,None]*s[2:4]))))
            clearance = min(clearance,distance)
        for obstacle in env.get_static_obstacles():
            clearance = min(clearance,float(np.min(obb_clearance(boxes,np.broadcast_to(obstacle_box(obstacle),boxes.shape)))))
        if clearance < self.margin:
            return None,'background_margin'
        nominal = _box(ego,ego[:2]+times[:,None]*ego[2:4])
        gap = obb_clearance(boxes,nominal)
        contact = np.flatnonzero(gap<=0)
        closing = max(0.,float(np.linalg.norm(ego[2:4])-np.linalg.norm(velocities[-1])))
        ttc = float(times[contact[0]]) if len(contact) else (float(gap[-1]/closing+3) if closing>1e-3 else None)
        if ttc is None or not self.ttc_range[0]<=ttc<=self.ttc_range[1]:
            return None,'outside_ttc_band'
        return dict(estimated_risk=float(np.clip(1-ttc/(self.ttc_range[1]+1),0,1)),
                    estimated_ttc_s=ttc,avoidability=dict(witness=None,
                        requirement='theoretical_drivable_area_must_remain_positive'),
                    background_safety_margin_m=finite(clearance),
                    screening_model='constant_velocity_others; 0.05s samples; background OBB margin; attacker road topology exempt; not continuous safety guarantee'),None


def policy_id_for_sim(sim_cfg):
    """在画像开关两侧使用同一自车策略身份，避免跨策略检索。"""
    settings = getattr(sim_cfg,'ego_profile',None)
    identity = str(getattr(settings,'policy_id','') or '|'.join(str(getattr(sim_cfg,k,'')) for k in ('policy','rl_model_path','rl_model_name')))
    return hashlib.sha256(identity.encode()).hexdigest()[:20]


class ProfileAttackPipeline:
    def __init__(self, sim_cfg):
        from types import SimpleNamespace
        settings = getattr(sim_cfg,'ego_profile',SimpleNamespace())
        self.policy_id = policy_id_for_sim(sim_cfg)
        self.profile = EgoProfile(settings)
        self.knowledge = AttackKnowledge(self.policy_id,getattr(settings,'history_paths',[]))
        self.builder = CandidateBuilder(settings)
        collision_cfg = getattr(getattr(getattr(getattr(sim_cfg,'traffic_model',None),'guidance',None),'loss_configs',None),'scenario_collision',None)
        self.dynamic_limits = {name:float(getattr(collision_cfg,name,default)) for name,default in
                               (('max_speed',20.),('max_acceleration',6.),('max_jerk',20.),('max_step_distance',2.))}
        if any(not np.isfinite(v) or v<=0 for v in self.dynamic_limits.values()):
            raise ValueError('画像硬门槛需要有效且为正的动力学边界')
        self.pending = None
        self.attempt_id = None
        self.attempt_dir = None

    def begin_attempt(self, env, directory):
        self.profile.reset_episode()
        self.attempt_id = uuid.uuid4().hex
        self.attempt_dir = Path(directory)
        self.knowledge.path = self.attempt_dir/'strategy_returns.jsonl'
        self.profile.observe(env)

    def context(self, env, u):
        condition = scene_conditions(env)
        profile = self.profile.snapshot(condition)
        candidates,rejected = self.builder.build(env,profile,self.knowledge)
        self.pending = dict(attempt_id=self.attempt_id,scene_id=str(getattr(env,'current_scene_id','unknown')),request_id=uuid.uuid4().hex,profile_version=profile['version'],
                            scene_conditions=condition, strategy=None,target=None,u=float(u),
                            actual_attack_execution_frames=0,D=None,drivable_area_change_ratio=None,
                            area_basis='reachable_area_not_map_drivable_area',generation_failure_reason=None,
                            accepted=False,collision=False,background_safe=True,reward=None,reaction_time_s=None,
                            reaction_censored=True,first_execution_step=None,candidate_rejections=rejected)
        self.pending['pre_attack_acceleration_mps2'] = (self.profile.previous or {}).get('accel')
        return dict(profile=profile,candidates=candidates,candidate_rejections=rejected,
                    history=self.knowledge.retrieve(condition),objective='low response margin; retain positive theoretical drivable area; collision is not the reward objective')

    def planned(self, plan, service_failure=False):
        if not self.pending:
            return
        self.pending['obstacle_plan'] = plan.get('obstacle_plan',[]) if plan else []
        self.pending['obstacle_created_ids'] = []
        self.pending['actual_obstacle_exposure_frames'] = 0
        if plan and plan.get('attack'):
            self.pending.update(request_id=plan.get('request_id') or self.pending['request_id'],
                                strategy=plan.get('attack_family',plan['strategy']),target=plan['attack_target_id'],
                                candidate_id=plan.get('candidate_id'))
        elif plan and plan.get('obstacle_plan'):
            self.pending.update(strategy='obstacle_only',request_id=plan.get('request_id') or self.pending['request_id'])
        else:
            self.pending['generation_failure_reason'] = 'planner_service_failure' if service_failure else (plan.get('reason','no_attack') if plan else 'plan_validation_failed')

    def executed(self, env, audit, info, trace):
        executed = bool(audit and audit.get('executed'))
        self.profile.observe(env,executed)
        row = self.pending
        if not row:
            return
        row['collision'] |= bool(info.get('collision'))
        row['background_safe'] &= not bool(trace.previous_background or trace.previous_static)
        if set(row.get('obstacle_created_ids',[])) & {str(o['id']) for o in env.get_static_obstacles()}:
            row['actual_obstacle_exposure_frames'] += 1
        if executed and audit.get('target_id')==row['target'] and audit.get('request_id')==row['request_id']:
            row['actual_attack_execution_frames'] += 1
            if row['first_execution_step'] is None:
                row['first_execution_step'] = int(env.current_step)
            previous = self.profile.previous or {}
            baseline = row.get('pre_attack_acceleration_mps2')
            if row['reaction_censored'] and baseline is not None and abs((previous.get('accel') or 0)-baseline)>1.:
                row['reaction_time_s'] = (int(env.current_step)-row['first_execution_step'])*float(env.dt)
                row['reaction_censored'] = False

    def validate_joint(self, env):
        """对实际联合预测施加动力学和背景硬门槛；不使用道路距离拒绝。"""
        from policies.joint_safety import NoSafeJointCandidate
        joint = env.pending_joint_trajectory
        if joint is None or joint.positions_global.shape[1]<2 or not joint.valid_mask.all():
            raise NoSafeJointCandidate('profile_missing_joint_prediction')
        states = np.asarray(env.data_dict['agent'][-1],float)
        if sorted(map(int,joint.agent_ids)) != list(np.flatnonzero(env.agent_active)):
            raise NoSafeJointCandidate('profile_incomplete_joint_prediction')
        if not np.isfinite(states[np.flatnonzero(env.agent_active),:7]).all():
            raise NoSafeJointCandidate('profile_invalid_agent_geometry')
        positions = np.asarray(joint.positions_global,float)
        yaws = np.asarray(joint.yaws_global,float).reshape(positions.shape[:2])
        if not np.isfinite(positions).all() or not np.isfinite(yaws).all():
            raise NoSafeJointCandidate('profile_invalid_joint_geometry')
        boxes = []
        for row,agent_id in enumerate(joint.agent_ids):
            s = states[agent_id]
            velocity = np.diff(np.vstack((s[:2],positions[row])),axis=0)/env.dt
            acceleration = np.diff(np.vstack((s[2:4],velocity)),axis=0)/env.dt
            jerk = np.diff(acceleration,axis=0)/env.dt
            limits = self.dynamic_limits
            previously_attacked = set(getattr(env, 'previously_attacked_agent_ids', ()))
            if agent_id not in previously_attacked and (np.max(np.linalg.norm(velocity,axis=1))>limits['max_speed']
                    or np.max(np.linalg.norm(velocity*env.dt,axis=1))>limits['max_step_distance']
                    or np.max(np.linalg.norm(acceleration,axis=1))>limits['max_acceleration']
                    or (len(jerk) and np.max(np.linalg.norm(jerk,axis=1))>limits['max_jerk'])):
                raise NoSafeJointCandidate('profile_joint_dynamics')
            # 历史攻击车跳过加速度/jerk 等动力学门控，避免攻击结束状态阻断下一次规划。
            # 所有背景参与者均不再因道路中心线距离被拒绝。
            pos = np.vstack((s[:2],positions[row]))
            yaw = np.unwrap(np.r_[s[4],yaws[row]])
            pos = np.vstack((pos,(pos[1:]+pos[:-1])/2))
            yaw = np.r_[yaw,(yaw[1:]+yaw[:-1])/2]
            boxes.append(np.column_stack((pos,yaw,np.full(len(pos),s[5]),np.full(len(pos),s[6]))))
        for i,box in enumerate(boxes):
            if any(np.min(obb_clearance(box,other))<self.builder.margin for other in boxes[:i]):
                raise NoSafeJointCandidate('profile_joint_background_margin')
            if any(np.min(obb_clearance(box,np.broadcast_to(obstacle_box(o),box.shape)))<self.builder.margin for o in env.get_static_obstacles()):
                raise NoSafeJointCandidate('profile_joint_static_margin')

    def finish_window(self, window=None, events=(), failure=None):
        row = self.pending
        if row is None:
            return
        self.pending = None
        row['generation_failure_reason'] = failure or row['generation_failure_reason']
        row['D'] = window.get('difficulty') if window else None
        row['valid_observation_frames'] = window['weight'] if window and window.get('valid') else 0
        row['metrics'] = window.get('metrics',{}) if window else {}
        valid = [e for e in events if finite(e.get('original_area_m2')) is not None and e['original_area_m2']>0 and finite(e.get('dangerous_area_m2')) is not None]
        row['reachability_events'] = [{k:finite(e.get(k)) for k in ('original_area_m2','dangerous_area_m2','difficulty','dangerous_solvable')} for e in valid]
        if valid:
            row['drivable_area_change_ratio'] = float(np.mean([(e['dangerous_area_m2']-e['original_area_m2'])/e['original_area_m2'] for e in valid]))
        if row['strategy'] and not row['actual_attack_execution_frames'] and not row.get('actual_obstacle_exposure_frames') and not row['generation_failure_reason']:
            row['generation_failure_reason'] = 'no_attack_execution'
        solvable = bool(valid) and all(e.get('dangerous_solvable') is True or e.get('dangerous_solvable')==1 for e in valid)
        if (row['actual_attack_execution_frames'] or row.get('actual_obstacle_exposure_frames')) and row['D'] is not None and valid:
            if row['collision'] or not row['background_safe'] or not solvable or row['generation_failure_reason']:
                row['reward'] = 0.
            else:
                metrics = row['metrics']
                ttc = finite(metrics.get('avg_min_ttc'))
                in_band = ttc is not None and self.builder.ttc_range[0]<=ttc<=self.builder.ttc_range[1]
                ea = finite(metrics.get('mean_evasive_acceleration_mps2')) or 0.
                reduction = max(0.,-row['drivable_area_change_ratio'])
                delay = row['reaction_time_s']
                # 无响应为删失，不能伪造为高收益的无限反应时间。
                row['reward'] = float(.4*in_band+.25*min(1.,ea/4)+.25*min(1.,reduction)+.1*(min(1.,delay/2) if delay is not None else 0))
        self.knowledge.append(row)

    def finish_attempt(self, env, failure=None):
        self.finish_window(failure=failure)
        snapshot = self.profile.snapshot(scene_conditions(env))
        snapshot['policy_id'] = self.policy_id
        self.attempt_dir.mkdir(parents=True,exist_ok=True)
        (self.attempt_dir/'ego_profile.json').write_text(json.dumps(snapshot,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')
        return dict(policy_id=self.policy_id,profile_version=snapshot['version'],
                    profile_path=str(self.attempt_dir/'ego_profile.json'),knowledge_path=str(self.knowledge.path))
