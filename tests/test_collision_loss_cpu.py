"""执行真实碰撞损失函数；仅隔离 Safe-Sim 注册、坐标工具与基类依赖。"""
import ast
import os
import time
import numpy as np
import sys
import unittest
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
# CPU 验证依赖独立于项目环境；远端可直接使用已有 torch。
local_deps = ROOT.parent / 'validation_deps'
if local_deps.exists():
    sys.path.insert(0,str(local_deps))
import torch
from policies.guidance_state import guidance_local_yaw
from policies.joint_safety import safe_joint_candidates, NoSafeJointCandidate
from types import SimpleNamespace


class Base:
    def __init__(self,loss_timesteps,filter_timesteps,loss_scale):
        self.loss_timesteps,self.filter_timesteps,self.loss_scale = loss_timesteps,filter_timesteps,loss_scale


def enlarge(value,b,n):
    return value.repeat_interleave(n,dim=0) if value.shape[0] == b else value


def transform(points,matrix):
    return torch.einsum('bij,btj->bti',matrix[:,:2,:2],points)+matrix[:,None,:2,2]


# 从当前生产源码抽取原函数，避免复制损失实现导致测试与实现脱节。
tree = ast.parse((ROOT/'policies/scenario_guidance.py').read_text(encoding='utf-8'))
names = {'_world_positions','_ego_future','_obb_signed_clearance','ScenarioCollisionLossCalculator'}
nodes = [n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
for node in nodes:
    node.decorator_list = []
module = ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]))
scope = dict(torch=torch,F=torch.nn.functional,Guidance=Base,Any=Any,Dict=Dict,
             transform_points_tensor=transform,enlarge_batch_samples=enlarge,
             guidance_local_yaw=guidance_local_yaw)
# 阶段调度函数同样直接抽取生产源码。
schedule = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='resolve_attack_contact_schedule')
exec(compile(ast.fix_missing_locations(ast.Module(body=[schedule],type_ignores=[])),'schedule','exec'),scope)
exec(compile(module,'actual_collision_loss','exec'),scope)
Calculator = scope['ScenarioCollisionLossCalculator']


def params(b=2,n=1):
    return dict(batch_size=b,num_samples=n,world_from_agent=torch.eye(3).repeat(b*n,1,1),
                scenario_ego_state=torch.tensor([[1000.,0,0,0,0,4.,2.,1.]]).repeat(b,1),
                ego_extents=torch.tensor([[8.,2.]]).repeat(b,1),curr_speed=torch.zeros(b),
                yaw=torch.zeros(b*n),guidance_attack_active=torch.zeros(b),dt=.1)


class CollisionLossTests(unittest.TestCase):
    def test_collision_penalty_covers_prediction_tail(self):
        # 仅在第31个点重叠；旧20点配置会完全忽略该危险。
        s = torch.zeros(2,32,4)
        s[:,:,0] = 20
        s[1,:,1] = 5
        s[1,30:,1] = 1
        options = dict(max_speed=1e9,max_acceleration=1e9,max_jerk=1e9,max_step_distance=1e9)
        full = Calculator(**options).calculate_loss(s[...,:2],s,params())
        truncated = Calculator(loss_timesteps=20,**options).calculate_loss(s[...,:2],s,params())
        self.assertGreater(full[:,30:].sum().item(),100)
        self.assertEqual(truncated[:,30:].sum().item(),0)

    def calculator(self):
        return Calculator(loss_timesteps=2,max_speed=1e9,max_acceleration=1e9,max_jerk=1e9,max_step_distance=1e9)

    def test_real_loss_uses_yaw_not_speed(self):
        s = torch.zeros(2,2,4)
        s[:,:,0] = 20
        s[1,:,1] = 3
        c,p = self.calculator(),params()
        straight = c.calculate_loss(s[...,:2],s,p).sum()
        s[1,:,3] = torch.pi/2
        rotated = c.calculate_loss(s[...,:2],s,p).sum()
        self.assertGreater(rotated.item(),straight.item()+100)
        s[:,:,2] = 17
        torch.testing.assert_close(c.calculate_loss(s[...,:2],s,p).sum(),rotated)

    def test_old_index_reproduces_missing_collision_penalty(self):
        s = torch.zeros(2,2,4)
        s[:,:,0] = 20
        s[1,:,1] = 3
        s[1,:,3] = torch.pi/2
        c,p = self.calculator(),params()
        fixed = c.calculate_loss(s[...,:2],s,p).sum()
        scope['guidance_local_yaw'] = lambda x:x[...,2]
        try:
            broken = c.calculate_loss(s[...,:2],s,p).sum()
        finally:
            scope['guidance_local_yaw'] = guidance_local_yaw
        self.assertGreater(fixed.item(),broken.item()+100)

    def test_batched_gradients_are_finite(self):
        s = torch.zeros(6,2,4)
        s[:,:,0] = 20
        s[3:,:,1] = 2
        s[3:,:,3] = .6
        s.requires_grad_()
        loss = self.calculator().calculate_loss(s[...,:2],s,params(2,3))
        self.assertEqual(tuple(loss.shape),(6,2))
        loss.sum().backward()
        self.assertTrue(torch.isfinite(s.grad).all())
        self.assertGreater(s.grad[...,3].abs().sum().item(),0)

    def test_actual_wrapper_rejects_unsafe_joint_candidates(self):
        # 执行当前封装器的真实筛选方法，仅以固定损失替代模型评分。
        wrapper_tree = ast.parse((ROOT/'policies/diffusion_model_wrapper.py').read_text(encoding='utf-8'))
        fn = next(n for n in ast.walk(wrapper_tree) if isinstance(n,ast.FunctionDef) and n.name=='_select_joint_guided_action')
        namespace = dict(torch=torch,np=np,os=os,time=time,Path=Path,safe_joint_candidates=safe_joint_candidates,NoSafeJointCandidate=NoSafeJointCandidate)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'actual_selection','exec'),namespace)
        select = namespace['_select_joint_guided_action']
        positions = torch.zeros(2,3,2,2)
        positions[1,2,:,1] = 5
        yaws = torch.zeros(2,3,2,1)
        data = dict(world_from_agent=torch.eye(3).repeat(6,1,1),ego_extents=torch.tensor([[4.,2.],[4.,2.]]))
        class Loss:
            @staticmethod
            def calculate_loss(action,state,params):
                return state[...,1].square()
        net = SimpleNamespace(Loss_Calculater=Loss(),_prepare_guidance_data=lambda _:data)
        action = SimpleNamespace(positions=positions[:,0],yaws=yaws[:,0])
        chosen,index = select(SimpleNamespace(guidance_enabled=True),net,{},action,dict(action_samples=dict(positions=positions,yaws=yaws)))
        self.assertEqual(index,2)
        positions[1,2,:,1] = 0
        with self.assertRaises(NoSafeJointCandidate):
            select(SimpleNamespace(guidance_enabled=True),net,{},action,dict(action_samples=dict(positions=positions,yaws=yaws)))


if __name__ == '__main__':
    unittest.main()
