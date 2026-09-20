"""画像安全语义必须到达真实扩散损失，不能只停留在上层提示词。"""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Dict

import torch
import numpy as np
import pytest

from test_collision_loss_cpu import scope, params, Calculator

ROOT=Path(__file__).resolve().parents[1]


def test_profile_preserves_contact_phase():
    state=torch.zeros(1,2,4)
    state[:,:,0]=10
    p=params(b=1)
    p['scenario_ego_state'][0,:2]=torch.tensor([0.,0.])
    p['guidance_attack_active']=torch.ones(1)
    p['guidance_attack_age_frames']=torch.tensor([20])
    calc=Calculator(loss_timesteps=2,max_speed=1e9,max_acceleration=1e9,max_jerk=1e9,max_step_distance=1e9)
    contact=calc.calculate_loss(state[...,:2],state,p).sum()
    p['profile_guided']=torch.ones(1)
    profiled=calc.calculate_loss(state[...,:2],state,p).sum()
    p['guidance_attack_age_frames']=torch.tensor([0])
    protected=calc.calculate_loss(state[...,:2],state,p).sum()
    torch.testing.assert_close(profiled,contact)
    assert profiled>protected


def test_profile_ttc_reward_does_not_prefer_immediate_contact():
    tree=ast.parse((ROOT/'policies/scenario_guidance.py').read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='ScenarioTTCLossCalculator')
    node.decorator_list=[]
    local=dict(scope)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'production_ttc','exec'),local)
    calc=local['ScenarioTTCLossCalculator'](loss_timesteps=1)
    def loss(ttc,profile):
        p=params(b=1)
        p['scenario_ego_state'][0]=torch.tensor([0.,0.,10.,0.,0.,4.,2.,1.])
        p['world_from_agent'][0,0,2]=2*(ttc+.1)
        p['guidance_target_mask']=torch.ones(1)
        if profile:
            p['profile_ttc_band']=torch.tensor([[1.,4.]])
        state=torch.zeros(1,1,4)
        state[0,0,0]=.8
        return calc.calculate_loss(state[...,:2],state,p).item()
    assert loss(2.,True)<loss(.2,True)
    assert loss(.2,False)<loss(2.,False)


def test_model_preparation_preserves_profile_fields():
    tree=ast.parse((ROOT/'safe-sim/tbsim/models/RasterizedDiffusionModel.py').read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_prepare_guidance_data')
    local=dict(Dict=Dict,batch_utils=lambda:NS(get_drivable_region_map=lambda image:image),GeoUtils=NS(calc_distance_map=lambda image:image))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'prepare','exec'),local)
    batch={name:torch.zeros(1,1) for name in ('image','dt','centroid','curr_speed','yaw','raster_from_agent','all_other_agents_types')}
    batch.update(agent_fut_extent=torch.ones(1,1,3),world_from_agent=torch.eye(3)[None],
                 extras=dict(centerline_xy=torch.zeros(1,2,2),has_lane=torch.ones(1)),
                 profile_guided=torch.ones(1),profile_ttc_band=torch.tensor([[1.,4.]]))
    model=NS(diffuse_args={'num_samples':2},_adjust_batch_shapes=lambda data,size:None)
    data=local['_prepare_guidance_data'](model,batch)
    assert data['profile_guided'] is batch['profile_guided']
    assert data['profile_ttc_band'] is batch['profile_ttc_band']


def test_single_profile_switch_enables_anchor_guidance():
    tree=ast.parse((ROOT/'run_simulation.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='enable_llm_joint_guidance_for_natural_language')
    local={}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'enable','exec'),local)
    traffic=NS(iterative_adversarial=NS(profile_enabled=True,full_method_enabled=False,escalation_enabled=False),guidance=NS(mode='unguided'))
    assert local[node.name](NS(sim=NS(traffic_model=traffic)),[])
    assert traffic.guidance.mode=='llm_joint'


def test_wrapper_keeps_road_exemption_for_profile_attack():
    tree=ast.parse((ROOT/'policies/diffusion_model_wrapper.py').read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='predict')
    captured={}
    class SamplingBoundary(Exception):
        pass
    def intercept(batch,device):
        captured.update(batch)
        raise SamplingBoundary()
    local=dict(np=np,torch=torch,ScenarioFrame=NS,_move_to_device=intercept)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'wrapper','exec'),local)
    frame=NS(scene_id='test',dt=.1,agent_ids=np.array([0]),active_mask=np.array([True]),step=9)
    batch=NS(row_to_agent_id=np.array([0]),data={})
    wrapper=NS(scene_id='test',step_time=.1,performance_diagnostics=False,device='cpu',
               _select_diffusion_agent_ids=lambda *args:np.array([0]),adapter=NS(build=lambda *a,**kw:batch),
               far_agent_mode='guided',uses_anchor_guidance=False,_select_adversarial_target_id=lambda *a,**kw:0,
               guidance_mode='llm_joint')
    with pytest.raises(SamplingBoundary):
        local['predict'](wrapper,frame,dict(source_step=0,profile_guided=True,ttc_range_s=[1.,4.]))
    torch.testing.assert_close(captured['guidance_route_exempt_mask'],torch.ones(1))
    assert captured['guidance_attack_age_frames'].item()==9
    torch.testing.assert_close(captured['profile_ttc_band'],torch.tensor([[1.,4.]]))
