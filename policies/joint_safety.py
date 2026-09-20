"""联合候选的矩形占用复核；无可行解时显式拒绝。"""
import numpy as np


class NoSafeJointCandidate(RuntimeError):
    pass


def obb_clearance(a, b):
    """输入 [x,y,yaw,length,width]，返回 SAT 投影间距。"""
    def box_axes(box):
        c,s = np.cos(box[...,2]),np.sin(box[...,2])
        return np.stack((np.stack((c,s),axis=-1),np.stack((-s,c),axis=-1)),axis=-2)
    aa,bb = box_axes(a),box_axes(b)
    axes = np.concatenate((aa,bb),axis=-2)
    ra = np.sum(np.abs(np.einsum('...ij,...kj->...ik',axes,aa))*(a[...,None,3:5]*.5),axis=-1)
    rb = np.sum(np.abs(np.einsum('...ij,...kj->...ik',axes,bb))*(b[...,None,3:5]*.5),axis=-1)
    return np.max(np.abs(np.sum(axes*(b[...,:2]-a[...,:2])[...,None,:],axis=-1))-ra-rb,axis=-1)


def safe_joint_candidates(positions, yaws, transforms, extents, margin=0.0):
    """检查所有背景车对及相邻预测帧中点，保留联合样本一致性。"""
    positions, yaws = np.asarray(positions), np.asarray(yaws)
    b,n,t,_ = positions.shape
    transforms,extents = np.asarray(transforms),np.asarray(extents)
    if transforms.shape == (b*n,3,3):
        transforms = transforms.reshape(b,n,3,3)[:,0]
    if extents.shape[0] == b*n:
        extents = extents.reshape(b,n,-1)[:,0]
    if transforms.shape != (b,3,3) or extents.shape[0] != b or extents.shape[1] < 2:
        raise ValueError("联合碰撞复核的变换或尺寸维度不匹配")
    world = np.einsum('bij,bntj->bnti',transforms[:,:2,:2],positions)+transforms[:,None,None,:2,2]
    angle = yaws[...,0]+np.arctan2(transforms[:,1,0],transforms[:,0,0])[:,None,None]
    if t > 1:
        midpoint = (world[:,:,1:]+world[:,:,:-1])*.5
        delta = np.arctan2(np.sin(angle[:,:,1:]-angle[:,:,:-1]),np.cos(angle[:,:,1:]-angle[:,:,:-1]))
        middle_angle = angle[:,:,:-1]+.5*delta
        world = np.concatenate((world,midpoint),axis=2)
        angle = np.concatenate((angle,middle_angle),axis=2)
    boxes = np.concatenate((world,angle[...,None],np.broadcast_to(extents[:,None,None,:2],world.shape)),axis=-1)
    safe = np.isfinite(boxes).all(axis=(0,2,3)) & (boxes[...,3:] > 0).all(axis=(0,2,3))
    # 对候选与时间维同时矢量化，仅循环车辆对，避免逐帧 Python 检查开销。
    for i in range(b):
        for j in range(i):
            safe &= (obb_clearance(boxes[i],boxes[j]) >= margin).all(axis=-1)
    return safe


def validate_execution_prefix(states, joint, static_obstacles):
    """执行前复核真实起点、首点及中点；不宣称连续扫掠体保证。"""
    if not len(joint.agent_ids):
        return
    if joint.positions_global.shape[1] == 0 or not joint.valid_mask[:, 0].all():
        raise NoSafeJointCandidate('执行前复核：首帧缺少有效联合轨迹')
    current = np.asarray(states)[joint.agent_ids]
    positions = np.stack((current[:, :2], joint.positions_global[:, 0]), axis=1)[:, None]
    yaws = np.stack((current[:, 4], joint.yaws_global[:, 0, 0]), axis=1)[:, None, :, None]
    transforms = np.broadcast_to(np.eye(3), (len(current), 3, 3))
    if not safe_joint_candidates(positions, yaws, transforms, current[:, 5:7])[0]:
        raise NoSafeJointCandidate('执行前复核：背景车辆在真实起点至首帧存在重叠或无效几何')
    if not len(static_obstacles):
        return
    delta = np.arctan2(np.sin(yaws[:, 0, 1, 0] - yaws[:, 0, 0, 0]),
                       np.cos(yaws[:, 0, 1, 0] - yaws[:, 0, 0, 0]))
    for fraction in (0.0, 0.5, 1.0):
        points = positions[:, 0, 0] + fraction * (positions[:, 0, 1] - positions[:, 0, 0])
        headings = current[:, 4] + fraction * delta
        boxes = np.column_stack((points, headings, current[:, 5:7]))
        for obstacle in static_obstacles:
            if not np.isfinite(obstacle).all() or (obb_clearance(boxes, np.broadcast_to(obstacle, boxes.shape)) < 0).any():
                raise NoSafeJointCandidate('执行前复核：背景车辆与静态障碍物占用冲突')
