"""真实官方损失的梯度方向、虚拟自车隔离和道路/背景正则验证。"""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'safe-sim'))
from policies.safe_sim_official_guidance import register_safe_sim_official_guidance,augment_with_fixed_ego


def inputs():
    b,n,t=2,2,20
    state=torch.zeros((b*n,t,4))
    state[:n,:,0]=8.;state[:n,:,1]=-4.;state[n:,:,0]=30.
    state[...,2]=2.
    action=torch.zeros((b*n,t,2))
    ego=torch.tensor([[0.,0.,2.,0.,0.,4.,2.,1.]]).repeat(b,1)
    center=torch.zeros((b*n,100,3));center[...,0]=torch.linspace(-50,150,100)
    # 合法前方交叉冲突；同向共线车道在官方 conflict 判定中可能无攻击项。
    center[:n,:,0]=8.;center[:n,:,1]=torch.linspace(-50,150,100)
    params={'batch_size':b,'num_samples':n,'BN':b*n,'dt':.1,
        'world_from_agent':torch.eye(3).repeat(b*n,1,1),'yaw':torch.zeros(b*n),
        'ego_extents':torch.tensor([[4.,2.]]).repeat(b,1),'curr_speed':torch.ones(b)*2,
        'scene_ids':torch.zeros(b,dtype=torch.long),'centerline':center,'lane_avail':torch.ones(b*n),
        'scenario_ego_state':ego,'guidance_target_mask':torch.tensor([1.,0.])}
    return action,state,params


def test_official_causecollision_gradient_moves_target_toward_ego():
    from tbsim.utils.guidance_utils import GUIDANCE_REGISTRY
    from tbsim.configs.guidance_config import DEFAULT_GUIDANCE_CONFIGS
    register_safe_sim_official_guidance()
    a,s,p=inputs();s.requires_grad_(True)
    loss=GUIDANCE_REGISTRY['safe_sim_causecollision'](**DEFAULT_GUIDANCE_CONFIGS['safe_sim_causecollision'])
    values=loss.calculate_loss(a,s,p)
    assert values.shape==(4,20) and torch.isfinite(values).all()
    gradient=torch.autograd.grad(values.sum(),s)[0]
    assert gradient[:2,:,0].sum()>0  # 梯度下降使前方攻击车向自车靠近。
    assert torch.count_nonzero(gradient[2:])==0


def test_virtual_ego_is_not_returned_or_differentiable():
    a,s,p=inputs();s.requires_grad_(True);p['scenario_ego_state'].requires_grad_(True)
    aa,ss,pp,target=augment_with_fixed_ego(a,s,p)
    assert ss.shape==(6,20,4) and target==1
    assert pp['batch_size']==3 and pp['centerline'].shape[0]==3
    grad=torch.autograd.grad(ss.sum(),p['scenario_ego_state'],allow_unused=True)[0]
    assert grad is None


@pytest.mark.parametrize('kind',['safe_sim_collision','safe_sim_ttc'])
def test_official_regularizers_finite_and_background_gradient(kind):
    from tbsim.utils.guidance_utils import GUIDANCE_REGISTRY
    from tbsim.configs.guidance_config import DEFAULT_GUIDANCE_CONFIGS
    register_safe_sim_official_guidance()
    a,s,p=inputs();s[2:,:,0]=9.;s.requires_grad_(True)
    loss=GUIDANCE_REGISTRY[kind](**DEFAULT_GUIDANCE_CONFIGS[kind])
    v=loss.calculate_loss(a,s,p)
    assert v.shape==(4,20) and torch.isfinite(v).all()
    grad=torch.autograd.grad(v.sum(),s)[0]
    assert torch.isfinite(grad).all()


def test_bridge_matches_direct_original_causecollision_values():
    from tbsim.utils.guidance_utils import GUIDANCE_REGISTRY,CauseCollisionLossCalculator
    from tbsim.configs.guidance_config import DEFAULT_GUIDANCE_CONFIGS
    register_safe_sim_official_guidance()
    a,s,p=inputs();aa,ss,pp,target=augment_with_fixed_ego(a,s,p)
    native=CauseCollisionLossCalculator(**DEFAULT_GUIDANCE_CONFIGS['causecollision']);native.device=s.device
    native.update_config({'batch_ego_indices':[True,False,False],'batch_ctrl_indices':[False,True,False]})
    reference=native.calculate_loss(aa,ss,pp).reshape(3,2,20)[1:].reshape(4,20)
    bridge=GUIDANCE_REGISTRY['safe_sim_causecollision'](**DEFAULT_GUIDANCE_CONFIGS['safe_sim_causecollision'])
    torch.testing.assert_close(bridge.calculate_loss(a,s,p),reference)
