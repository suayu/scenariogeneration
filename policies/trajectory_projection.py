"""速度空间中的凸约束投影；仅修正连续轨迹，不改变攻击策略或安全阈值。"""
import time
from dataclasses import replace

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from policies.joint_safety import NoSafeJointCandidate


def project_joint(joint, states, dt, limits, settings, exempt_agent_ids=()):
    """整组修正后返回新对象；背景碰撞、道路及可规避性由调用方重新硬校验。"""
    if joint is None or not joint.valid_mask.all():
        raise NoSafeJointCandidate('projection_missing_joint')
    positions, velocities, yaws, audits = [], [], [], []
    exempt = set(map(int, exempt_agent_ids))
    for row, agent_id in enumerate(joint.agent_ids):
        if int(agent_id) in exempt:
            # 历史攻击车保留扩散输出，避免刚结束的攻击状态触发后续动力学投影失败。
            positions.append(np.asarray(joint.positions_global[row]))
            velocities.append(np.asarray(joint.velocities_global[row]))
            yaws.append(np.asarray(joint.yaws_global[row]))
            audits.append(dict(agent_id=int(agent_id), exempt=True,
                               reason='previously_selected_attacker', elapsed_seconds=0.0))
            continue
        p, v, y, audit = project_positions(joint.positions_global[row], states[agent_id], dt, limits,
                                          float(settings.max_seconds_per_vehicle), float(settings.max_deviation_m))
        positions.append(p)
        velocities.append(v)
        yaws.append(y)
        audits.append(dict(agent_id=int(agent_id), **audit))
    metadata = dict(joint.metadata, dynamics_projection=dict(vehicles=audits,
                        elapsed_seconds=sum(row['elapsed_seconds'] for row in audits)))
    return replace(joint, positions_global=np.asarray(positions), velocities_global=np.asarray(velocities),
                   yaws_global=np.asarray(yaws), metadata=metadata)


def project_positions(positions, initial_state, dt, limits, max_seconds=2.0, max_deviation=2.0):
    """最小化轨迹偏移，使用内接多边形保守近似二维动力学范数约束。"""
    started = time.monotonic()
    reference = np.asarray(positions, dtype=float)
    initial = np.asarray(initial_state, dtype=float)
    if reference.ndim != 2 or reference.shape[1] != 2 or len(reference) < 2:
        raise NoSafeJointCandidate('projection_invalid_shape')
    if not np.isfinite(reference).all() or not np.isfinite(initial).all() or not dt > 0:
        raise NoSafeJointCandidate('projection_invalid_input')
    if any(not np.isfinite(limits[k]) or limits[k] <= 0 for k in
           ('max_speed', 'max_acceleration', 'max_jerk', 'max_step_distance')):
        raise ValueError('invalid projection limits')
    horizon = len(reference)
    integration = np.kron(np.tril(np.ones((horizon, horizon)))*dt, np.eye(2))
    desired = (reference-initial[:2]).ravel()
    ref_velocity = np.diff(np.vstack((initial[:2], reference)), axis=0)/dt
    difference = np.eye(horizon)-np.eye(horizon, k=-1)
    # 八个法向量和双侧不等式组成十六边形；缩小边距保证不会超出真实圆形上限。
    angles = np.arange(8)*np.pi/8
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    conservative = np.cos(np.pi/16)*(1-1e-6)
    accel_origin = np.zeros((horizon, 2))
    accel_origin[0] = initial[2:4]
    matrices, lower, upper = [], [], []
    for operator, origin, bound in (
        (np.eye(horizon), np.zeros((horizon, 2)), min(limits['max_speed'], limits['max_step_distance']/dt)),
        (difference, accel_origin, limits['max_acceleration']*dt),
        (np.diff(difference, axis=0), np.diff(accel_origin, axis=0), limits['max_jerk']*dt*dt),
    ):
        matrices.append(np.kron(operator, normals))
        offset = (origin@normals.T).ravel()
        lower.append(offset-bound*conservative)
        upper.append(offset+bound*conservative)
    constraints = LinearConstraint(np.vstack(matrices), np.concatenate(lower), np.concatenate(upper))

    def objective(velocity):
        delta = integration@velocity-desired
        regularizer = velocity-ref_velocity.ravel()
        return float(delta@delta+.01*(regularizer@regularizer)), 2*integration.T@delta+.02*regularizer

    def deadline(_):
        if time.monotonic()-started > max_seconds:
            raise NoSafeJointCandidate('projection_timeout')

    solution = minimize(objective, np.tile(initial[2:4], horizon), jac=True,
                        constraints=[constraints], method='SLSQP', callback=deadline,
                        options={'maxiter': 100, 'ftol': 1e-9})
    if not solution.success:
        raise NoSafeJointCandidate('projection_solver_failure')
    velocity = solution.x.reshape(horizon, 2)
    projected = initial[:2]+np.cumsum(velocity, axis=0)*dt
    # 独立按执行门禁的差分公式复核，不能只信求解器返回成功。
    acceleration = np.diff(np.vstack((initial[2:4], velocity)), axis=0)/dt
    jerk = np.diff(acceleration, axis=0)/dt
    peaks = dict(speed=float(np.linalg.norm(velocity, axis=1).max()),
                 acceleration=float(np.linalg.norm(acceleration, axis=1).max()),
                 jerk=float(np.linalg.norm(jerk, axis=1).max()))
    if (peaks['speed'] > limits['max_speed'] or peaks['speed']*dt > limits['max_step_distance']
            or peaks['acceleration'] > limits['max_acceleration'] or peaks['jerk'] > limits['max_jerk']):
        raise NoSafeJointCandidate('projection_postcheck_failed')
    deviation = float(np.linalg.norm(projected-reference, axis=1).max())
    if deviation > max_deviation:
        raise NoSafeJointCandidate('projection_excessive_deviation')
    yaw = np.arctan2(velocity[:, 1], velocity[:, 0])
    previous = float(initial[4])
    for index in range(horizon):
        if np.linalg.norm(velocity[index]) < .05:
            yaw[index] = previous
        previous = yaw[index]
    return projected, velocity, yaw[:, None], dict(elapsed_seconds=time.monotonic()-started,
             max_deviation_m=deviation, iterations=int(solution.nit), peaks=peaks,
             method='velocity_qp_inscribed_16gon')
