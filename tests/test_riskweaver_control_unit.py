"""无需模型权重的控制、几何与统计最小回归测试。"""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from policies.difficulty_control import DifficultyController, weighted_difficulty, ability_score, reachability_query_due
from policies.guidance_state import guidance_local_yaw
from policies.joint_safety import obb_clearance, safe_joint_candidates
from policies.difficulty_audit import summarize_attempts, json_safe
from policies.route_progress import normalized_route_progress
from policies.collision_ttc import minimum_obb_ttc


class ControlTests(unittest.TestCase):
    def test_collision_decreases_even_when_below_target(self):
        f = DifficultyController().update(.5,.1,True,.5,.1)
        self.assertLess(f['after'],f['before'])
        self.assertEqual(f['reason'],'collision_decrease')

    def test_errors_drive_both_directions_and_deadband(self):
        c = DifficultyController()
        self.assertGreater(c.update(.5,.2,False,.5,.1)['after'],.5)
        self.assertLess(c.update(.5,.8,False,.5,.1)['after'],.5)
        self.assertEqual(c.update(.5,.55,False,.5,.1)['after'],.5)

    def test_bounds_and_more_than_four_levels(self):
        c = DifficultyController()
        values = [c.parameters(u) for u in np.linspace(0,1,101)]
        self.assertEqual(len(set(p['llm_anchor'] for p in values)),101)
        for p in values:
            for k,(lo,hi) in c.bounds.items():
                self.assertTrue(lo <= p[k] <= hi)
        self.assertEqual(c.update(0,.1,True,.5,.1)['after'],0)
        self.assertEqual(c.update(1,.1,False,.5,.1)['after'],1)

    def test_missing_is_not_zero(self):
        self.assertIsNone(weighted_difficulty([]))
        self.assertIsNone(ability_score(None,True,False,False,1))
        self.assertEqual(DifficultyController().update(.5,None,False,.5,.1)['after'],.5)

    def test_weighted_score_single_basis(self):
        d = weighted_difficulty([dict(difficulty=.2,weight=1),dict(difficulty=.6,weight=3)])
        self.assertAlmostEqual(d,.5)
        self.assertAlmostEqual(ability_score(d,True,False,False,1),50)
        self.assertAlmostEqual(ability_score(d,False,True,False,.4),10)

    def test_every_third_query(self):
        self.assertEqual([i for i in range(1,11) if reachability_query_due(i)],[3,6,9])

    def test_invalid_bounds_fail(self):
        with self.assertRaises(ValueError):
            DifficultyController({'bounds':{'llm_anchor':(2,1)}})

    def test_route_progress_is_fraction_not_metres(self):
        self.assertAlmostEqual(normalized_route_progress([25,2],[[0,0],[100,0]]),.25)
        self.assertAlmostEqual(normalized_route_progress([10,5],[[0,0],[10,0],[10,10]]),.75)
        self.assertIsNone(normalized_route_progress([0,0],[[0,0],[0,0]]))


class GeometryTests(unittest.TestCase):
    def test_obb_ttc_stationary_ego_and_parallel_miss(self):
        ego = np.array([0.,0.,0.,0.,0.,4.,2.,1.])
        other = np.array([[10.,0.,-2.,0.,0.,4.,2.,1.]])
        self.assertAlmostEqual(minimum_obb_ttc(ego,other,[True]),3.)
        other[0,1] = 3.
        self.assertTrue(np.isinf(minimum_obb_ttc(ego,other,[True])))
        other[0,:2] = [0.,0.]
        self.assertEqual(minimum_obb_ttc(ego,other,[True]),0)
    def test_yaw_reads_fourth_column_not_speed(self):
        state = np.array([[[20.,1.,15.,np.pi/2]]])
        self.assertAlmostEqual(guidance_local_yaw(state).item(),np.pi/2)
        state[...,2] = 99
        self.assertAlmostEqual(guidance_local_yaw(state).item(),np.pi/2)
        with self.assertRaises(ValueError):
            guidance_local_yaw(np.zeros((1,1,8)))

    def test_heading_changes_obb_collision(self):
        a = np.array([0.,0.,0.,8.,2.])
        b = np.array([0.,3.,0.,8.,2.])
        self.assertGreater(obb_clearance(a,b),0)
        b[2] = np.pi/2
        self.assertLess(obb_clearance(a,b),0)

    def test_select_safe_joint_candidate(self):
        pos = np.zeros((2,2,2,2))
        pos[1,1,:,1] = 5
        safe = safe_joint_candidates(pos,np.zeros((2,2,2,1)),np.tile(np.eye(3),(2,1,1)),np.array([[4.,2.],[4.,2.]]))
        self.assertEqual(safe.tolist(),[False,True])

    def test_midpoint_crossing_is_rejected(self):
        pos = np.zeros((2,1,2,2))
        pos[0,0,:,0] = [-5,5]
        pos[1,0,:,0] = [5,-5]
        safe = safe_joint_candidates(pos,np.zeros((2,1,2,1)),np.tile(np.eye(3),(2,1,1)),np.array([[2.,1.],[2.,1.]]))
        self.assertFalse(safe[0])

    def test_transforms_and_nonfinite(self):
        pos = np.zeros((2,1,1,2))
        transforms = np.tile(np.eye(3),(2,1,1))
        transforms[1,1,2] = 10
        self.assertTrue(safe_joint_candidates(pos,np.zeros((2,1,1,1)),transforms,np.ones((2,2)))[0])
        pos[0,0,0,0] = np.nan
        self.assertFalse(safe_joint_candidates(pos,np.zeros((2,1,1,1)),transforms,np.ones((2,2)))[0])


class AuditTests(unittest.TestCase):
    def test_first_final_missing_and_bootstrap(self):
        def a(d):
            return dict(attack_difficulty=d,difficulty_target=.5,difficulty_tolerance=.1)
        episodes = [dict(attempts=[a(.1),a(.5)],replay_count=1,difficulty_control_status='accepted'),
                    dict(attempts=[a(None),a(None)],replay_count=1,difficulty_control_status='uncontrollable')]
        s = summarize_attempts(episodes,samples=100)
        self.assertEqual(s['initial']['hit_rate'],0)
        self.assertEqual(s['final']['hit_rate'],.5)
        self.assertEqual(s['final']['missing_count'],1)
        self.assertEqual(s['uncontrollable_rate'],.5)
        self.assertEqual(s,summarize_attempts(episodes,samples=100))
        self.assertIsNone(json_safe({'x':float('nan')})['x'])


if __name__ == '__main__':
    unittest.main()
