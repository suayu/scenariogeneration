"""Safe-Sim 指导状态契约：[x, y, speed, yaw]；区别于仿真全局八维状态。"""


def guidance_local_yaw(state):
    """保持张量类型及梯度，固定从索引 3 读取候选航向。"""
    if state.shape[-1] != 4:
        raise ValueError("Safe-Sim guidance state 必须是 [x,y,speed,yaw] 四维")
    return state[..., 3]
