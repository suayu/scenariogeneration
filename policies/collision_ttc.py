"""固定航向、常速度假设下的 OBB 首次接触时间，非最近接近时间。"""
import numpy as np


def minimum_obb_ttc(ego, agents, active_mask):
    """全局状态为 [x,y,vx,vy,yaw,length,width,...]；无相交返回 inf。"""
    ego = np.asarray(ego,dtype=float)
    agents = np.asarray(agents,dtype=float)[np.asarray(active_mask,dtype=bool)]
    best = float('inf')
    if ego.shape[-1] < 4 or agents.ndim != 2 or agents.shape[-1] < 4:
        raise ValueError("TTC 输入至少需要 [x,y,vx,vy]")

    def point_ttc(other):
        """兼容旧式四维状态；完整仿真状态仍使用下方 OBB 首次接触时间。"""
        relative_position = other[:2]-ego[:2]
        distance = float(np.linalg.norm(relative_position))
        if distance <= 1e-10:
            return 0.0
        relative_velocity = other[2:4]-ego[2:4]
        closing_speed = -float(relative_position @ relative_velocity)/distance
        return distance/closing_speed if closing_speed > 1e-10 else float('inf')

    def axes(box):
        c,s = np.cos(box[4]),np.sin(box[4])
        return np.array([[c,s],[-s,c]])
    full_ego = ego.shape[-1] >= 7
    ea = axes(ego) if full_ego else None
    for other in agents:
        if not np.isfinite(other[:4]).all() or not np.isfinite(ego[:4]).all():
            raise ValueError("TTC 输入包含非有限状态")
        if not full_ego or other.shape[-1] < 7:
            best = min(best,point_ttc(other))
            continue
        if not np.isfinite(other[4:7]).all() or not np.isfinite(ego[4:7]).all():
            raise ValueError("TTC 输入包含非有限状态")
        oa = axes(other)
        directions = np.concatenate((ea,oa))
        radii = np.abs(directions @ ea.T) @ (ego[5:7]/2)+np.abs(directions @ oa.T) @ (other[5:7]/2)
        r = directions @ (other[:2]-ego[:2])
        v = directions @ (other[2:4]-ego[2:4])
        enter,leave = 0.0,float('inf')
        for pos,vel,radius in zip(r,v,radii):
            if abs(vel)<1e-10:
                if abs(pos)>radius:
                    leave = -1.0
                    break
                continue
            t1,t2 = (-radius-pos)/vel,(radius-pos)/vel
            enter = max(enter,min(t1,t2))
            leave = min(leave,max(t1,t2))
        if enter <= leave:
            best = min(best,enter)
    return float(best)
