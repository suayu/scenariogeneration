"""官方 SAFE-SIM 损失的外部自车适配；不修改官方数学公式或旧默认引导。"""
import copy
import torch


def augment_with_fixed_ego(action, state, params):
    """添加仅供损失计算的固定自车行，输出时剔除，绝不让 diffusion 控制自车。"""
    b, n, t = int(params['batch_size']), int(params['num_samples']), state.shape[1]
    ego = params['scenario_ego_state'][0].to(state).detach()
    times = torch.arange(1, t+1, device=state.device, dtype=state.dtype)*float(params['dt'])
    virtual = state.new_zeros((n, t, state.shape[-1]))
    virtual[..., :2] = ego[:2] + times[:, None]*ego[2:4]
    virtual[..., -2] = torch.linalg.vector_norm(ego[2:4])
    virtual[..., -1] = ego[4]
    p = dict(params)
    identity = torch.eye(3, device=state.device, dtype=state.dtype).repeat(n,1,1)
    p['world_from_agent'] = torch.cat((identity,params['world_from_agent']),dim=0)
    p['yaw'] = torch.cat((state.new_zeros(n),params['yaw']),dim=0)
    p['ego_extents'] = torch.cat((ego[None,5:7],params['ego_extents']),dim=0)
    p['curr_speed'] = torch.cat((torch.linalg.vector_norm(ego[2:4])[None],params['curr_speed']),dim=0)
    p['scene_ids'] = torch.cat((params['scene_ids'][:1],params['scene_ids']),dim=0)
    p['ego_plan'] = virtual[:1,:,:2]
    # 原 causecollision 用 B 行车道；模型预处理将车道扩至 BN，在边界压回 B。
    center = params['centerline'].reshape(b,n,*params['centerline'].shape[1:])[:,0,:,:2]
    line_count = center.shape[1]
    direction = torch.stack((torch.cos(ego[4]),torch.sin(ego[4])))
    ego_line = state.new_zeros((1,line_count,center.shape[-1]))
    ego_line[0,:,:2] = ego[:2] + torch.linspace(-50,150,line_count,device=state.device)[:,None]*direction
    if center.shape[-1] > 2:
        ego_line[0,:,2] = ego[4]
    p['centerline'] = torch.cat((ego_line,center),dim=0)
    lane = params['lane_avail'].reshape(b,n)[:,0]
    p['lane_avail'] = torch.cat((torch.ones(1,device=lane.device,dtype=lane.dtype),lane))
    p['batch_size'],p['BN'] = b+1,(b+1)*n
    target = torch.nonzero(params['guidance_target_mask'],as_tuple=False).flatten()
    if len(target) > 1:
        raise ValueError('SAFE-SIM baseline requires one discrete adversarial target')
    target_row = int(target[0])+1 if len(target) else None
    return torch.cat((action.new_zeros(n,t,action.shape[-1]),action)), torch.cat((virtual,state)),p,target_row


def register_safe_sim_official_guidance():
    """组合官方 route/collision/ttc/causecollision，虚拟自车只在损失内部存在。"""
    from tbsim.utils.guidance_utils import (Guidance,register_guidance,GUIDANCE_REGISTRY,
        CollisionLossCalculator,TimeToCollisionLossCalculator,CauseCollisionLossCalculator)
    from tbsim.configs.guidance_config import DEFAULT_GUIDANCE_CONFIGS
    kinds={'safe_sim_collision':('collision',CollisionLossCalculator),
           'safe_sim_ttc':('ttc',TimeToCollisionLossCalculator),
           'safe_sim_causecollision':('causecollision',CauseCollisionLossCalculator)}
    for name,(kind,cls) in kinds.items():
        if name in GUIDANCE_REGISTRY:
            continue
        config=copy.deepcopy(DEFAULT_GUIDANCE_CONFIGS[kind])
        DEFAULT_GUIDANCE_CONFIGS.setdefault(name,config)
        def factory(kind=kind,cls=cls):
            class OfficialLossBridge(Guidance):
                def __init__(self,**kwargs):
                    super().__init__(kwargs.get('loss_timesteps'),kwargs.get('filter_timesteps'),kwargs.get('loss_scale',1.0))
                    self.original=cls(**kwargs)
                    self.kind=kind
                def update_config(self,config):
                    pass
                def update_params(self,params):
                    pass
                def calculate_loss(self,action,state,params):
                    a,s,p,target=augment_with_fixed_ego(action,state,params)
                    b,n=int(p['batch_size']),int(p['num_samples'])
                    # 官方旧 Torch 实现做 1-mask，显式浮点掩码兼容当前 Torch。
                    ego=[0.]*b;ctrl=[0.]*b
                    ego[0]=1.
                    if target is not None:
                        ctrl[target]=1.
                    # 官方碰撞豁免语义为无向车辆对：仅移除自车—攻击车双向项。
                    if self.kind=='collision' and target is not None:
                        ego[target]=1.;ctrl[0]=1.
                    self.original.device=state.device
                    self.original.update_config({'batch_ego_indices':ego,'batch_ctrl_indices':ctrl})
                    result=self.original.calculate_loss(a,s,p)
                    if isinstance(result,tuple):
                        result=result[0]
                    if target is None and self.kind=='causecollision':
                        return state[...,0]*0
                    if not torch.isfinite(result).all():
                        raise ValueError('nonfinite official SAFE-SIM guidance')
                    # 官方无交互分支产生常量零，保留零梯度连接以兼容 inner-loop。
                    return result.reshape(b,n,state.shape[1])[1:].reshape_as(state[...,0]) + state[...,0]*0
            return OfficialLossBridge
        register_guidance(name)(factory())
