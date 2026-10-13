"""官方神经网络结构兼容性诊断；随机参数仅用于接口测试，绝非基线实验。"""
import json
from pathlib import Path
import sys

root=Path(sys.argv[1])
sys.path.insert(0,str(root/'src'))
import numpy as np
np.bool,np.int,np.float=np.bool_,int,float
import torch
from torch_geometric.data import Data,Batch
from models.traffic_model import TrafficModel
from datasets.utils import MeanStdNormalizer,NUSC_BIKE_PARAMS
from datasets.map_env import NuScenesMapEnv

torch.manual_seed(42)
model=TrafficModel(4,12,256,2)
# 可选真实权重诊断，默认随机参数接口诊断保持原样。
loaded = len(sys.argv) > 2
if loaded:
    checkpoint=torch.load(sys.argv[2],map_location='cpu')
    model.load_state_dict(checkpoint['model'],strict=True)
norm=MeanStdNormalizer(torch.zeros(6),torch.tensor([15.,15.,1.,1.,3.5,.05]))
att=MeanStdNormalizer(torch.tensor([4.8,2.]),torch.tensor([1.08,.3]))
model.set_normalizer(norm);model.set_att_normalizer(att);model.set_bicycle_params(NUSC_BIKE_PARAMS)
map_env=NuScenesMapEnv.__new__(NuScenesMapEnv)
map_env.nusc_raster=torch.ones((1,4,1000,1000),dtype=torch.uint8)
map_env.nusc_dx=torch.tensor([[.25,.25]])
map_env.bounds,map_env.L,map_env.W=[-17.,-38.5,60.,38.5],256,256
past=torch.zeros((2,4,6));past[...,0]=100;past[...,1]=100;past[...,2]=1;past[...,4]=2
past[1,:,1]+=10
future=past[:,-1:].expand(2,12,6).clone()
future[:,:,0]+=torch.arange(1,13)[None]*1.
graph=Batch.from_data_list([Data(x=torch.empty(2),pos=torch.empty(2),edge_index=torch.tensor([[0,1],[1,0]]),
  past=norm.normalize(past),past_gt=norm.normalize(past),future=norm.normalize(future),future_gt=norm.normalize(future),
  lw=att.normalize(torch.tensor([[4.8,2.],[4.8,2.]])),sem=torch.tensor([[1.,0.],[1.,0.]]),
  past_vis=torch.ones((2,4)),future_vis=torch.ones((2,12)))])
model.eval()
with torch.no_grad():
    prediction=model.sample_batched(graph,torch.tensor([0]),map_env,2,include_mean=True)
assert prediction['future_pred'].shape==(2,2,12,4)
assert torch.isfinite(prediction['future_pred']).all()
from utils.scenario_gen import detach_embed_info
from utils.init_optim import run_init_optim
from utils.adv_gen_optim import run_adv_gen_optim
with torch.no_grad():
    embedding=detach_embed_info(model.embed(graph,torch.tensor([0]),map_env))
for param in model.parameters():
    param.requires_grad_(False)
weights={'init_match_ext':10.,'init_motion_prior_ext':.1,'coll_veh':20.,'coll_veh_plan':20.,'coll_env':20.,
 'motion_prior':1.,'motion_prior_atk':.005,'init_z':.5,'init_z_atk':.05,
 'motion_prior_ext':.0001,'match_ext':10.,'adv_crash':2.}
z,_,_=run_init_optim(embedding['posterior_out'][0],graph.future_gt[...,:4],graph.future_vis,.1,weights,
                    model,graph,map_env,torch.tensor([0]),1,embedding,embedding['prior_out'])
class DiagnosticPlanner:
    def reset(self,*args):
        pass
    def rollout(self,*args,**kw):
        return future[0:1,:,:4]
out=run_adv_gen_optim(z,.05,weights,model,graph,map_env,torch.tensor([0]),1,embedding,'hardcode',
 tuple(v[:1] for v in embedding['prior_out']),tuple(v[1:] for v in embedding['prior_out']),4,0.,
 planner=DiagnosticPlanner())
assert torch.isfinite(out[1]).all()
print(json.dumps({'status':'passed','diagnostic_only':True,'pretrained_weights_loaded':loaded,
  'torch':torch.__version__,'output_shape':list(prediction['future_pred'].shape),
  'official_init_and_closed_loop_adv_backward':'passed'}))
