"""验证STRIVE时间/坐标转换、缓存和缺失权重门禁，不将随机网络当正式结果。"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from policies.strive_adapter import make_request, resample_trajectory, StriveController, isolated_random_state
from policies.traffic_types import JointTrajectory, ScenarioFrame
from experiments.comparison.run_strive_baseline import slice_joint


def frame():
    states=np.array([[10,0,2,0,0,4,2,1]],dtype=np.float32)
    history=np.tile(states[:,None],(1,16,1))
    history[0,:,0]=10+np.arange(-15,1)*.2
    types=np.zeros((1,5),np.float32);types[:,1]=1
    return ScenarioFrame('test',15,.1,np.array([0]),states,types,np.array([True]),history,
        np.ones((1,16),bool),np.array([[[0,-1],[20,-1]]],np.float32),np.array([[0,0],[20,0]],np.float32),
        np.array([0,0,2,0,0,4,2,1],np.float32))


def test_real_history_sampled_at_half_second_and_translation_preserves_geometry():
    f=frame();ego=np.tile(f.ego_state_global,(16,1))
    request=make_request(f,ego,np.ones(16,bool))
    np.testing.assert_allclose(request['past'][1,:,0]+request['origin'][0],[7,8,9,10])
    np.testing.assert_allclose(request['future'][1,:,0]+request['origin'][0],10+np.arange(1,13))
    assert request['past_vis'].all()
    assert request['future_initialization']=='constant_velocity_no_ground_truth_future'
    assert request['raster'][1:].sum()==0
    assert 'unavailable' in request['map_semantics']


def test_missing_history_stays_missing():
    f=frame();f.history_mask[:,:-1]=False
    ego=np.tile(f.ego_state_global,(16,1));mask=np.zeros(16,bool);mask[-1]=True
    request=make_request(f,ego,mask)
    assert np.isnan(request['past'][:,:-1]).all()
    assert not request['past_vis'][:,:-1].any()
    assert request['past_vis'][:,-1].all()


def test_nonvehicles_are_not_silently_changed_to_cars():
    f=frame();f.agent_types[:]=0
    with pytest.raises(ValueError,match='vehicles only'):
        make_request(f,np.tile(f.ego_state_global,(16,1)),np.ones(16,bool))


def test_resampling_has_real_origin_correct_speed_and_six_second_horizon():
    current=np.array([[0,0,2,0,0]],float)
    future=np.zeros((1,12,4));future[0,:,0]=np.arange(1,13);future[...,2]=1
    p,y,v=resample_trajectory(current,future,.5,.1)
    assert p.shape==(1,60,2)
    np.testing.assert_allclose(p[0,0],[.2,0],atol=1e-6)
    np.testing.assert_allclose(p[0,-1],[12,0])
    np.testing.assert_allclose(v[...,0],2,atol=1e-5)


def test_heading_wrap_does_not_rotate_through_zero():
    current=np.array([[0,0,0,0,np.pi-.1]])
    future=np.array([[[0,0,np.cos(-np.pi+.1),np.sin(-np.pi+.1)]]])
    _,y,_=resample_trajectory(current,future,.5,.1)
    assert np.max(np.abs(np.diff(y[0,:,0])))<.05


@pytest.mark.parametrize('bad', [np.nan,np.inf])
def test_nonfinite_prediction_is_rejected(bad):
    future=np.array([[[1,0,1,0]]],float);future[0,0,0]=bad
    with pytest.raises(ValueError,match='nonfinite'):
        resample_trajectory(np.array([[0,0,0,0,0]]),future,.5,.1)


def test_cache_is_shifted_without_extrapolation():
    p=np.arange(8,dtype=np.float32).reshape(1,4,2)
    joint=JointTrajectory(10,np.array([0]),p,np.zeros((1,4,1)),np.zeros((1,4,2)),np.ones((1,4),bool))
    shifted=slice_joint(joint,12,2)
    assert shifted.source_step==12
    np.testing.assert_array_equal(shifted.positions_global,p[:,2:])
    with pytest.raises(ValueError,match='exhausted'):
        slice_joint(joint,14,4)


def test_official_weights_are_required(tmp_path):
    with pytest.raises(FileNotFoundError,match='checkpoint missing'):
        StriveController(tmp_path,'python',tmp_path/'missing.pth',tmp_path/'planning')


def test_counterfactual_random_draws_do_not_change_real_experiment_stream():
    import random
    import torch
    random.seed(13);np.random.seed(13);torch.manual_seed(13)
    expected=(random.random(),np.random.rand(),torch.rand(1).item())
    random.seed(13);np.random.seed(13);torch.manual_seed(13)
    @isolated_random_state
    def draws():
        random.random();np.random.rand();torch.rand(1)
        raise RuntimeError('failed counterfactual')
    with pytest.raises(RuntimeError):
        draws()
    assert (random.random(),np.random.rand(),torch.rand(1).item())==expected
