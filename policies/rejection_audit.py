"""拒绝时冻结预测与指标；诊断不参与是否执行的判定。"""
import numpy as np


def snapshot_joint(joint):
    """复制保存已有轨迹；没有预测时明确返回空值。"""
    if joint is None:
        return None
    return dict(source_step=int(joint.source_step), agent_ids=joint.agent_ids.tolist(),
                positions=joint.positions_global.tolist(), yaws=joint.yaws_global.tolist(),
                valid=joint.valid_mask.tolist(), metadata=dict(joint.metadata))


def joint_metrics(env, pipeline, joint):
    """按现有公式逐车计算峰值与道路距离；道路距离只诊断，不参与门控。"""
    if joint is None or pipeline is None:
        return None
    from policies.ego_profile import project
    lanes = [np.asarray(l)[:,:2] for l in env.scenario_dict['lanes']
             if len(l)>1 and project(np.asarray(l)[0,:2], l) is not None]
    rows = []
    for index, agent_id in enumerate(joint.agent_ids):
        initial = np.asarray(env.data_dict['agent'][-1][agent_id], float)
        positions = np.asarray(joint.positions_global[index], float)
        velocity = np.diff(np.vstack((initial[:2], positions)),axis=0)/env.dt
        acceleration = np.diff(np.vstack((initial[2:4], velocity)),axis=0)/env.dt
        jerk = np.diff(acceleration,axis=0)/env.dt
        row = dict(agent_id=int(agent_id), dynamics={})
        for key, vector in [('max_speed',velocity), ('max_acceleration',acceleration),
                            ('max_jerk',jerk), ('max_step_distance',velocity*env.dt)]:
            norms = np.linalg.norm(vector,axis=1)
            worst = int(np.argmax(norms))
            measured = float(norms[worst])
            limit = pipeline.dynamic_limits[key]
            row['dynamics'][key] = dict(value=measured, limit=limit, future_index=worst,
                                         violated=bool(measured>limit))
        if lanes:
            points = np.vstack((initial[:2],positions))
            yaws = np.unwrap(np.r_[initial[4], np.asarray(joint.yaws_global[index]).reshape(-1)])
            points = np.vstack((points,(points[1:]+points[:-1])/2))
            yaws = np.r_[yaws,(yaws[1:]+yaws[:-1])/2]
            distances = []
            for along, across in ((1,1),(1,-1),(-1,1),(-1,-1)):
                corner = points+along*initial[5]/2*np.column_stack((np.cos(yaws),np.sin(yaws)))+across*initial[6]/2*np.column_stack((-np.sin(yaws),np.cos(yaws)))
                distances.append(np.min(np.stack([project(corner,line) for line in lanes]),axis=0))
            values = np.asarray(distances)
            worst = np.unravel_index(np.argmax(values),values.shape)
            row['road'] = dict(max_corner_distance_m=float(values[worst]),
                                initial_max_corner_distance_m=float(values[:,0].max()),
                                limit_m=pipeline.builder.half_width,
                                corner_index=int(worst[0]), point_index=int(worst[1]),
                                gate_enabled=False, violated=False,
                                exceeds_diagnostic_reference=bool(values[worst]>pipeline.builder.half_width),
                                index_semantics='true initial, predicted endpoints, then adjacent midpoints')
        rows.append(row)
    return rows


def rejection_record(env, pipeline, reason, original_joint):
    """审计错误不能掩盖原拒绝理由；只输出异常类别，不保存异常正文。"""
    record = dict(kind='attack_rejected',step=int(env.current_step),reason=reason,
                  attack_intent=env.attack_intent,executed=False)
    try:
        record.update(original_joint=snapshot_joint(original_joint),
                      rejected_joint=snapshot_joint(env.pending_joint_trajectory),
                      original_metrics=joint_metrics(env,pipeline,original_joint),
                      rejected_metrics=joint_metrics(env,pipeline,env.pending_joint_trajectory))
    except Exception as error:
        record['diagnostic_error_type'] = type(error).__name__
    return record
