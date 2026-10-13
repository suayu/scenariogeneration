"""在独立子进程中调用STRIVE官方网络及初始化/对抗/解优化代码。"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def configure_rnn_autograd(model, backend):
    """只为RNN前向选择可导原生后端，CNN仍可使用cuDNN加速，保持eval语义。"""
    import torch
    if backend not in ('native', 'cudnn'):
        raise ValueError('invalid RNN backend')
    if backend == 'native':
        for layer in model.modules():
            if isinstance(layer, (torch.nn.RNN, torch.nn.LSTM, torch.nn.GRU)):
                original = layer.forward
                def forward(*args, _original=original, **kwargs):
                    with torch.backends.cudnn.flags(enabled=False):
                        return _original(*args, **kwargs)
                layer.forward = forward
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--native-data', type=Path)
    parser.add_argument('--native-index', type=int, default=0)
    parser.add_argument('--native-output', type=Path)
    parser.add_argument('--init-iters', type=int, default=75)
    parser.add_argument('--adv-iters', type=int, default=300)
    parser.add_argument('--solution-iters', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rnn-backend', choices=['native', 'cudnn'], default='native')
    args = parser.parse_args()
    expected = 'b708951f8665c97a1de9ed93b4ed3f58dd8cbf5d'
    actual = subprocess.check_output(['git','-C',str(args.root),'rev-parse','HEAD'],text=True).strip()
    if actual != expected or not args.checkpoint.is_file():
        raise RuntimeError('Official commit or checkpoint prerequisite not satisfied')
    if min(args.init_iters, args.adv_iters, args.solution_iters) < 1:
        raise ValueError('optimization budgets must be positive')
    if bool(args.input) == bool(args.native_data):
        raise ValueError('select exactly one of --input / --native-data')
    # 官方顶层包名与RiskWeaver冲突，因此仅在此子进程调整sys.path。
    sys.path.insert(0, str(args.root / 'src'))
    protocol_stdout = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        # 仅为官方旧版NumPy API提供兼容；不修改其源码或主进程全局状态。
        np.bool, np.int, np.float = np.bool_, int, float
        import torch
        import yaml
        from torch_geometric.data import Data, Batch
        from models.traffic_model import TrafficModel
        from datasets.utils import MeanStdNormalizer, NUSC_NORM_STATS, NUSC_BIKE_PARAMS
        from datasets.map_env import NuScenesMapEnv
        from utils.scenario_gen import detach_embed_info, determine_feasibility_nusc
        from utils.init_optim import run_init_optim
        from utils.adv_gen_optim import run_adv_gen_optim, compute_adv_gen_success
        from utils.sol_optim import run_find_solution_optim, compute_sol_success
        from utils.logger import Logger
        # cuDNN 的评估模式GRU不保存反向传播所需reserve；原生RNN保留eval语义和潜变量梯度。
        # 仅影响隔离worker，不能把整个模型切换train而改变Dropout/BN行为。
        # 不在worker中全局禁用cuDNN，避免连CNN都失去加速。
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 官方闭环函数有未指定device的arange等工厂；在隔离worker统一默认设备。
        # 不更改官方源码或优化目标，避免CUDA图与CPU索引混用。
        torch.set_default_device(device)
        cfg = yaml.safe_load((args.root / 'configs/adv_gen_replay.cfg').read_text())
        weights = {k[5:]:float(v) for k,v in cfg.items() if k.startswith('loss_')}
        # 官方函数使用init_/sol_前缀后直接接loss名。
        weights.update({k.replace('_loss_', '_'):float(v) for k,v in cfg.items() if k.startswith(('init_loss_', 'sol_loss_'))})
        weights['sol_init_z'] = 0.0
        native = args.native_data is not None
        if native:
            from datasets.nuscenes_dataset import NuScenesDataset
            if args.native_output is None:
                raise ValueError('--native-output required')
            args.native_output.mkdir(parents=True, exist_ok=False)
            Logger.init(str(args.native_output / 'official.log'))
            map_env = NuScenesMapEnv(str(args.native_data), device=device)
            dataset = NuScenesDataset(str(args.native_data),map_env,version='mini',split='val',seq_interval=10)
            raw_graph, map_id = dataset[args.native_index]
            graph = Batch.from_data_list([raw_graph]).to(device)
            map_idx = torch.tensor([map_id], dtype=torch.long,device=device)
            normalizer, att = dataset.get_state_normalizer(), dataset.get_att_normalizer()
            request_id = f'nuscenes-mini:{args.native_index}'
        else:
            Logger.init(str(args.input / 'official.log'))
            meta = json.loads((args.input/'input.json').read_text())
            arrays = np.load(args.input/'input.npz')
            request_id = meta['request_id']
            ninfo = NUSC_NORM_STATS[('car','truck')]
            normalizer = MeanStdNormalizer(torch.tensor([0.,0.,0.,0.,ninfo['s'][0],ninfo['hdot'][0]]),
                                         torch.tensor([15.,15.,1.,1.,ninfo['s'][1],ninfo['hdot'][1]]))
            att = MeanStdNormalizer(torch.tensor([ninfo['l'][0],ninfo['w'][0]]),torch.tensor([ninfo['l'][1],ninfo['w'][1]]))
            # 复用官方地图裁剪和碰撞查询；缺失语义层在输入审计中明确记录。
            map_env = NuScenesMapEnv.__new__(NuScenesMapEnv)
            map_env.nusc_raster = torch.tensor(arrays['raster'][None],device=device)
            map_env.nusc_dx = torch.tensor([[.25,.25]],device=device)
            map_env.bounds, map_env.L, map_env.W = [-17.,-38.5,60.,38.5],256,256
            map_env.num_layers, map_env.map_list, map_env.device = 4,['scenario_dreamer_proxy'],device
            past = normalizer.normalize(torch.tensor(arrays['past'],device=device))
            future = normalizer.normalize(torch.tensor(arrays['future'],device=device))
            count = past.shape[0]
            edges = torch.tensor([(i,j) for i in range(count) for j in range(count) if i!=j],device=device).T.long().contiguous()
            sem = torch.zeros((count,2),device=device)
            sem[:,0] = 1.0
            graph = Batch.from_data_list([Data(x=torch.empty(count,device=device),pos=torch.empty(count,device=device),
                edge_index=edges,past=past,past_gt=past.clone(),future=future,future_gt=future.clone(),sem=sem,
                lw=att.normalize(torch.tensor(arrays['lw'],device=device)),
                past_vis=torch.tensor(arrays['past_vis'],device=device),future_vis=torch.ones((count,12),device=device))])
            map_idx = torch.zeros(1,dtype=torch.long,device=device)
        model = TrafficModel(4,12,256,2,conv_channel_in=4).to(device)
        # 拒绝官方load_state(strict=False)容忍不完整权重的行为。
        checkpoint = torch.load(args.checkpoint,map_location='cpu')
        model.load_state_dict(checkpoint['model'],strict=True)
        model.set_normalizer(normalizer)
        model.set_att_normalizer(att)
        model.set_bicycle_params(NUSC_BIKE_PARAMS)
        model.eval()
        configure_rnn_autograd(model, args.rnn_backend)
        for param in model.parameters():
            param.requires_grad_(False)
        with torch.no_grad():
            embedding = model.embed(graph,map_idx,map_env)
            prior_samples = model.sample_batched(graph,map_idx,map_env,20,include_mean=True)
        feasibility = determine_feasibility_nusc(prior_samples['future_pred'],normalizer,10.0,4,0.0,
                  feasibility_infront_min=0.0,check_non_drivable_separation=True,map_env=map_env,map_idx=map_idx)
        if feasibility[0] is None or not feasibility[0].any():
            result = {'kind':'result','request_id':request_id,'status':'generation_failure',
                      'reason':'official_no_feasible_seed','model_loaded_strict':True}
            target = args.native_output if native else args.input
            (target/'generation_result.json').write_text(json.dumps(result,indent=2))
            protocol_stdout.write(json.dumps(result)+'\n'); protocol_stdout.flush()
            return
        embedding = detach_embed_info(embedding)
        z = embedding['posterior_out'][0].clone()
        z,initial,_ = run_init_optim(z,graph.future_gt[...,:4],graph.future_vis,.1,weights,
                                 model,graph,map_env,map_idx,args.init_iters,embedding,embedding['prior_out'])
        ego_mask = torch.arange(graph.past.shape[0],device=device)==0
        tgt_prior = tuple(v[ego_mask] for v in embedding['prior_out'])
        other_prior = tuple(v[~ego_mask] for v in embedding['prior_out'])
        class ExternalPlanner:
            """官方闭环优化回调，同一次请求内每轮重新计算真实策略响应。"""
            def __init__(self):
                self.calls = 0
            def reset(self,*unused):
                pass
            def rollout(self,others,plan_t,ptr,out_t,**unused):
                protocol_stdout.write(json.dumps({'kind':'rollout','request_id':request_id,
                      'callback_id':self.calls,'future':others.tolist()})+'\n')
                protocol_stdout.flush()
                line = sys.stdin.readline()
                if not line:
                    raise RuntimeError('missing planner callback response')
                response = json.loads(line)
                if response.get('request_id')!=request_id or response.get('callback_id')!=self.calls:
                    raise RuntimeError('planner callback identity mismatch')
                self.calls += 1
                ego = np.asarray(response['ego'])
                if ego.shape != (12,4) or not np.isfinite(ego).all():
                    raise ValueError('invalid ego planner rollout')
                return torch.tensor(ego[None],device=device,dtype=torch.float32)
        out = run_adv_gen_optim(z,.05,weights,model,graph,map_env,map_idx,args.adv_iters,embedding,
                     'ego' if native else 'hardcode',tgt_prior,other_prior,4,0.0,
                     planner=None if native else ExternalPlanner())
        z,final,decoder,attack_ids,attack_times = out
        attack_id = int(attack_ids[0])
        success = bool(compute_adv_gen_success(final,model,graph,attack_id))
        sol_ok = None
        if success:
            _,solution,_ = run_find_solution_optim(z,final,16,.05,weights,model,graph,map_env,map_idx,
                                      args.solution_iters,embedding,tgt_prior,other_prior)
            sol_ok = bool(compute_sol_success(solution[:,0:1],model,graph,map_env,map_idx))
            np.save(str((args.native_output if native else args.input)/'solution.npy'),normalizer.unnormalize(solution).cpu().numpy())
        result = {'kind':'result','request_id':request_id,'status':'completed','model_loaded_strict':True,
                  'rnn_backend':args.rnn_backend,'official_commit':actual,'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  'model_dt':.5,'attack_row':attack_id,'attack_time_index':int(attack_times[0]),
                  'official_adv_success':success,'official_solution_success':sol_ok,
                  'future':normalizer.unnormalize(final[:,0]).cpu().numpy().tolist()}
        target = args.native_output if native else args.input
        (target/'generation_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    protocol_stdout.write(json.dumps(result)+'\n'); protocol_stdout.flush()


if __name__ == '__main__':
    main()
