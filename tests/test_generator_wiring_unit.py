"""生成器真实配置/查询接线测试；LLM 和风险算法使用记录调用的替身。"""
import ast
import os
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from policies.difficulty_control import DifficultyController, weighted_difficulty, reachability_query_due
from policies.collision_ttc import minimum_obb_ttc


class FakeRisk:
    def __init__(self,cfg):
        self.cfg,self.ea_values,self.reachability_events,self.started = cfg,[],[],0
    def begin_attack(self,env):
        self.started += 1
    def evaluate_dangerous_reachability(self,env):
        return self.started


tree = ast.parse((ROOT/'scenario_generator.py').read_text(encoding='utf-8'))
nodes = [n for n in tree.body if isinstance(n,(ast.ClassDef,ast.Assign))]
scope = dict(os=os,random=random,np=np,SimpleNamespace=NS,DifficultyController=DifficultyController,
             weighted_difficulty=weighted_difficulty,reachability_query_due=reachability_query_due,
             minimum_obb_ttc=minimum_obb_ttc,AdversarialRiskMetrics=FakeRisk)
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'actual_generator','exec'),scope)
Generator = scope['AdversarialScenarioGenerator']


def fixture(root=True):
    sim = NS(evaluation=NS(difficulty_control=NS(mode='target',target_difficulty=.5,tolerance=.1,controller=None,reachability_every_queries=3)),
             traffic_model=NS(iterative_adversarial=NS(profile_enabled=True,escalation_enabled=True,full_method_enabled=True)),
             llm=NS(provider='qwen',max_consecutive_failures=3),policy='idm')
    planner = NS(available=True,use_multimodal=False,generate_attack_plan=lambda *a,**kw:dict(attack=False,reason='no target'))
    return Generator(NS(sim=sim) if root else sim,['test'],llm_planner=planner)


class WiringTests(unittest.TestCase):
    def test_missing_window_is_not_zero_and_off_never_adjusts(self):
        scope['compute_scenario_danger_score'] = lambda metrics,cfg: .2
        for mode,measured,collision in [('target',False,False),('target',False,True),('target',True,False),('off',True,True)]:
            g = fixture()
            g.cfg.sim.evaluation.composite = NS()
            g.difficulty_mode = mode
            g.full_method_enabled = False
            g._iterative_window_metric_offsets = dict(ea=0,reachability=0)
            g._iterative_window = [dict(ttc=10.,collision=collision,accel=0.,yaw_rate=0.,jerk=0.,route_error=0.,near_miss=False)]
            if measured:
                g.risk_metrics.ea_values = [0.]
                g.risk_metrics.reachability_events = [dict(difficulty=.2,dangerous_solvable=True)]
            before = g._iterative_stage
            g._finish_iterative_window()
            w = g._episode_attack_difficulties[-1]
            self.assertEqual(w['valid'],measured)
            self.assertEqual(g.episode_difficulty(),.2 if measured else None)
            if mode == 'off':
                self.assertEqual(g._iterative_stage,before)
            elif not measured:
                self.assertEqual(g._iterative_stage,before)
            else:
                self.assertGreater(g._iterative_stage,before)

    def test_off_mode_still_collects_and_closes_observation_window(self):
        # off 关闭强度调节，不关闭危险度测量和攻击窗口结算。
        g = fixture()
        g.profile_enabled = g.escalation_enabled = False
        g.difficulty_mode = 'off'
        g._observation_started = True
        seen = []
        g.risk_metrics.evaluate_ea = lambda env: None
        g._record_full_profile = lambda env, info: seen.append('observed')
        g._compute_min_ttc = lambda env: 5.0
        g.collision_list, g.near_miss_list = [], []
        g.evaluate_reaction(NS(attack_intent=None), {})
        self.assertEqual(seen, ['observed'])
        g._finish_iterative_window = lambda: seen.append('closed')
        g._attacks_enabled = False
        env = NS(attack_intent={'target_id': 1}, clear_attack_intent=lambda: None)
        g.last_attack_frame, g.attack_duration = 0, 10
        g.step(env, 10)
        self.assertEqual(seen, ['observed', 'closed'])

    def test_root_and_sim_configs_enable_actual_controls(self):
        for root in (True,False):
            g = fixture(root)
            self.assertEqual((g.difficulty_mode,g.target_difficulty,g.difficulty_tolerance),('target',.5,.1))
            self.assertTrue(g.profile_enabled and g.escalation_enabled and g.full_method_enabled)
            self.assertIs(g.risk_metrics.cfg,g.cfg.sim.evaluation)

    def test_no_attack_third_query_still_begins_reachability(self):
        g = fixture()
        # 此例隔离既有危险度查询周期；完整画像链路由 test_ego_profile 覆盖。
        g.profile_enabled = False
        g.profile_pipeline = None
        env = NS(current_step=0,attack_intent=None,get_state_for_planning=lambda:dict(agents=[]),clear_attack_intent=lambda:None)
        for i in range(1,7):
            env.current_step=i*4
            g._llm_call_times=i-1
            g.plan_and_inject(env)
            g.evaluate_reachability(env)
        self.assertEqual(g.risk_metrics.started,2)


if __name__ == '__main__':
    unittest.main()
