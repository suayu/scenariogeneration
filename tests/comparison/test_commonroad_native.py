"""在独立CommonRoad环境执行真实后端，不用mock代替传播和碰撞裁剪。"""
import pytest
pytest.importorskip('commonroad_reach')
from experiments.comparison.commonroad_worker import compute


def request():
    return {'request_id':1,'route':[[0,0],[100,0]],'ego':[20,0,2,0,0,4,2,1],
            'dt':.5,'steps':6,'lane_half_width':4.,'max_speed':25.,'max_lateral_speed':3.,
            'a_lon':[-4.,2.],'a_lat':[-2.5,2.5],'grid_size':1.,
            'dynamic':[[] for _ in range(7)],'static':[]}


def test_empty_corridor_union_and_curve():
    r=compute(request())
    assert r['area_m2']>0 and len(r['area_curve'])==6
    assert r['area_curve'][-1]['area_m2']==r['area_m2']
    assert all(-4<=g['union_bounds'][1]<=g['union_bounds'][3]<=4 for g in r['node_counts'])


@pytest.mark.parametrize('kind',['static','dynamic'])
def test_blocked_region_is_zero_even_when_obstacle_exceeds_projection_domain(kind):
    q=request()
    if kind=='static':q['static']=[{'center':[20,0],'yaw':0.,'length':100.,'width':100.}]
    else:q['dynamic']=[[([20,0],0.,100.,100.)] for _ in range(7)]
    r=compute(q)
    assert r['area_m2']==0 and not r['solvable']
    assert all(x['area_m2']==0 for x in r['area_curve'])
