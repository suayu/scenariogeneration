"""低速攻击模板只在显式启用时生成，且仍进入现有安全筛查。"""
from types import SimpleNamespace

import numpy as np
import pytest

from policies.ego_profile import CandidateBuilder


def scene():
    route=np.column_stack((np.arange(-5,60,dtype=float),np.zeros(65)))
    return SimpleNamespace(ego_state=np.array([0.,0.,4.,0.,0.,4.,1.8]),
        data_dict={'agent':[np.array([[18.,0.,2.4,0.,0.,4.,1.8]])]},
        agent_active=np.array([True]),scenario_dict={'route':route,'lanes':np.array([route])},
        get_static_obstacles=lambda: [])


def test_low_speed_template_opt_in_uses_original_screen():
    env=scene()
    history=SimpleNamespace(retrieve=lambda condition,family:{'attempts':0})
    default=CandidateBuilder(SimpleNamespace())
    assert default.build(env,{},history)[0]==[]
    enabled=CandidateBuilder(SimpleNamespace(low_speed_decelerations_mps2=[.25,.5]))
    seen=[]
    def screen(environment,target,anchors,lanes):
        seen.append(np.asarray(anchors))
        return dict(estimated_ttc_s=2.5),None
    enabled.screen=screen
    candidates,_=enabled.build(env,{},history)
    assert len(candidates)==len(seen)==2
    assert all(c['target_id']==0 and c['strategy']=='slow_down' for c in candidates)
    assert all(a.shape==(4,2) and np.isfinite(a).all() for a in seen)


def test_low_speed_template_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        CandidateBuilder(SimpleNamespace(low_speed_decelerations_mps2=[0]))
