"""百炼联合模式的候选边界与实际规划协议回归；传输层使用替身。"""
import json

from policies.llm_adversarial_planner import LLMAdversarialPlanner, ModelOutputError
from policies.obstacles import ObstacleCatalog
from policies.profile_obstacles import build_obstacle_candidates, chosen_obstacle
from policies.profile_planner import rank_profile_candidates
from test_ego_profile import environment,pipeline,planner as fixture_planner


def obstacle_planner():
    planner=fixture_planner()
    planner.provider='dashscope'
    planner.use_obstacles=True
    planner.obstacle_catalog=ObstacleCatalog()
    planner._validate_obstacle_plan=LLMAdversarialPlanner._validate_obstacle_plan.__get__(planner)
    return planner


def context_for_obstacles(env,planner):
    context=pipeline().context(env,.5)
    context['obstacle_candidates']=build_obstacle_candidates(env,planner,context['candidates'],pipeline().builder)
    return context


def test_joint_obstacle_only_selection_with_available_dynamic_candidate():
    env=environment()
    env.data_dict['agent'][-1][0,1]=6.
    planner=obstacle_planner()
    context=context_for_obstacles(env,planner)
    # TTC 区间门控暂时关闭后，区间外动态候选会保留；联合协议仍允许只选障碍物。
    assert context['candidates']
    assert context['obstacle_candidates']
    candidate=context['obstacle_candidates'][0]
    assert candidate['placement']['type']=='cone_line'
    assert candidate['background_margin_m']>.5

    def response(prompt,scene_image=None):
        payload=json.loads(prompt.split('\n',1)[1])
        return '',dict(request_id=payload['request_id'],ranking=[],no_attack_reason='select obstacle-only probe',
                       obstacle_candidate_id=candidate['obstacle_candidate_id'])
    planner._request_attack_plan=response
    plan=rank_profile_candidates(planner,env.get_state_for_planning(),[],context)
    assert not plan['attack'] and plan['obstacle_plan']==[candidate['placement']]


def test_joint_obstacle_id_and_combination_are_closed_catalog():
    env=environment()
    planner=obstacle_planner()
    context=context_for_obstacles(env,planner)
    assert context['candidates']
    if context['obstacle_candidates']:
        candidate=context['obstacle_candidates'][0]
        assert chosen_obstacle({'obstacle_candidate_id':candidate['obstacle_candidate_id']},context,[])==[candidate['placement']]
    try:
        chosen_obstacle({'obstacle_candidate_id':'invented'},context,[])
    except ValueError:
        pass
    else:
        raise AssertionError('unknown obstacle ID accepted')


def test_profile_catalog_includes_construction_zone_candidate():
    """施工路段必须经过画像候选目录，而不是允许 LLM 自由生成摆放参数。"""
    env=environment()
    # 构造无背景占用的道路，单独验证施工模板能进入封闭候选目录。
    env.agent_active[:]=False
    planner=obstacle_planner()
    context=context_for_obstacles(env,planner)
    construction=[item for item in context['obstacle_candidates']
                  if item['placement']['type']=='construction_zone']
    assert construction
    assert all(item['placement']['count']==1 for item in construction)
    assert all(item['placement']['spacing']==4. for item in construction)


def test_trajectory_mode_retains_existing_three_field_ranking():
    env=environment()
    planner=fixture_planner()
    planner.provider='dashscope'
    planner.use_obstacles=False
    context=pipeline().context(env,.5)
    result=rank_profile_candidates(planner,env.get_state_for_planning(),[],context)
    assert result['attack']
    assert 'obstacle_plan' not in result


def test_invalid_json_ranking_is_validation_failure_not_service_failure():
    env=environment()
    planner=fixture_planner()
    planner.provider='dashscope'
    planner.use_obstacles=False
    planner._is_service_failure=LLMAdversarialPlanner._is_service_failure
    planner.last_attempted_model_names=['bad-json']
    planner.last_model_failures=[{
        'model':'bad-json','category':'invalid_json_output','status_code':None,
    }]
    planner.last_successful_model_name=None

    def invalid_response(prompt,scene_image=None):
        raise ModelOutputError('纯文本大模型输出不是有效 JSON')

    planner._request_attack_plan=invalid_response
    context=pipeline().context(env,.5)
    assert rank_profile_candidates(
        planner,env.get_state_for_planning(),[],context
    ) is None
    assert not planner.last_request_failed
    assert planner.last_trace['plan_validation']=='plan_validation_failed'
    assert planner.last_trace['attempted_models']==['bad-json']
