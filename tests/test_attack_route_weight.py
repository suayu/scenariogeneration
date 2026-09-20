"""道路引导只在显式开启的攻击窗口切换，保持求解器参数和默认路径。"""
from types import SimpleNamespace as NS

import pytest
import torch

from scenario_generator import AdversarialScenarioGenerator


def fixture(configured, margin=None):
    generator=object.__new__(AdversarialScenarioGenerator)
    generator.cfg=NS(sim=NS(traffic_model=NS(attack_route_weight=configured,
                                           attack_route_lane_margin=margin)))
    route_loss=NS(lane_margin=1.,non_linear_margin=1.5)
    calculator=NS(weights=torch.tensor([1.5,1.,1.5,3.25]),
                  loss_calculator_dict={'route':route_loss})
    net=NS(Loss_Calculater=calculator)
    calls=[]
    def update(params,weights):
        calls.append((dict(params),list(weights)))
        calculator.weights=torch.tensor(weights)
        return True
    controller=NS(active_guidance_functions=['scenario_collision','route','scenario_ttc','llm_anchor'],
                  policy_replicas=[(None,NS(nets={'policy':net}))],update_iterative_guidance=update,
                  _find_loss_calculator=lambda root,name:root.loss_calculator_dict.get(name))
    return generator,NS(diffusion_controller=controller),calls,route_loss


def test_opt_in_switches_only_route_and_restores_baseline():
    generator,env,calls,_=fixture(5.)
    assert generator.apply_attack_route_weight(env,False)['route_weight']==1.
    assert generator.apply_attack_route_weight(env,True)['route_weight']==5.
    assert generator.apply_attack_route_weight(env,False)['route_weight']==1.
    assert [entry[1] for entry in calls]==[[1.5,1.,1.5,3.25],
                                          [1.5,5.,1.5,3.25],[1.5,1.,1.5,3.25]]
    assert all(params=={} for params,_ in calls)


def test_disabled_and_invalid_route_setting():
    generator,env,calls,_=fixture(None)
    assert generator.apply_attack_route_weight(env,True) is None
    assert calls==[]
    generator,env,_,_=fixture(float('nan'))
    with pytest.raises(ValueError):
        generator.apply_attack_route_weight(env,True)


def test_attack_margin_is_reversible_and_does_not_change_solver():
    generator,env,calls,route=fixture(5.,.55)
    generator.apply_attack_route_weight(env,False)
    active=generator.apply_attack_route_weight(env,True)
    assert active['route_lane_margin']==.55
    assert route.lane_margin==.55 and route.non_linear_margin==1.05
    restored=generator.apply_attack_route_weight(env,False)
    assert restored['route_lane_margin']==1.
    assert route.lane_margin==1. and route.non_linear_margin==1.5
    assert all(params=={} for params,_ in calls)
