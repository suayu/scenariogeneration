"""用自车到路线折线的最近投影计算归一化进度，避免把米数当成比例。"""
import numpy as np


def normalized_route_progress(position, route):
    route = np.asarray(route,dtype=float)
    if route.ndim != 2 or len(route) < 2 or route.shape[1] < 2:
        return None
    route = route[:,:2]
    delta = np.diff(route,axis=0)
    lengths = np.linalg.norm(delta,axis=1)
    total = lengths.sum()
    if not np.isfinite(total) or total <= 1e-9:
        return None
    fraction = np.clip(np.sum((np.asarray(position)[:2]-route[:-1])*delta,axis=1)/np.maximum(lengths**2,1e-12),0,1)
    projections = route[:-1]+fraction[:,None]*delta
    index = int(np.argmin(np.linalg.norm(projections-np.asarray(position)[:2],axis=1)))
    return float((np.sum(lengths[:index])+fraction[index]*lengths[index])/total)
