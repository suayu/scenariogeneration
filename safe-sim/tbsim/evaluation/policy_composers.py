"""A script for evaluating closed-loop simulation"""
from tbsim.algos.algos import (
    BehaviorCloning,
    DiffusionTrafficModel,
    SpatialPlanner,
)
from tbsim.utils.batch_utils import batch_utils
from tbsim.algos.multiagent_algos import MATrafficModel, HierarchicalAgentAwareModel
from tbsim.configs.registry import get_registered_experiment_config
from tbsim.utils.config_utils import get_experiment_config_from_file,update_config
from tbsim.policies.hardcoded import (
    ReplayPolicy, 
    GTPolicy, 
    # ContingencyPlanner,
    HierSplineSamplingPolicy,
    IDMPlanner,
    OptimController,
    # StrivePlanner,
    StrivePlanner_trajdata,
    # PDMClosed,
    # PDMClosed_trajdata
    )
from tbsim.configs.base import ExperimentConfig

from tbsim.policies.wrappers import (
    PolicyWrapper,
    HierarchicalWrapper,
    HierarchicalSamplerWrapper,
    SamplingPolicyWrapper,
)
from tbsim.configs.config import Dict
from tbsim.utils.experiment_utils import get_checkpoint

try:
    from Pplan.Sampling.spline_planner import SplinePlanner
    from Pplan.Sampling.trajectory_tree import TrajTree
except ImportError:
    print("Cannot import Pplan")


class PolicyComposer(object):
    """
    策略组合器基类。

    采用工厂模式,根据eval_config和device来构建策略实例。
    子类需实现 get_policy() 方法，返回 (policy, exp_config) 元组。
    该类在 tbsim.evaluation.policy_composers 模块中定义，被 build_single_policy 和 build_dual_policies 调用。
    """
    def __init__(self, eval_config, device, ckpt_root_dir="checkpoints/"):
        self.device = device
        self.ckpt_root_dir = ckpt_root_dir
        self.eval_config = eval_config
        self._exp_config = None

    def get_modality_shapes(self, exp_cfg: ExperimentConfig):
        return batch_utils().get_modality_shapes(exp_cfg)

    def get_policy(self):
        raise NotImplementedError


class ReplayAction(PolicyComposer):
    """
    回放策略：从 HDF5 文件中读取预先存储的动作序列，并逐帧回放。
    用于评估时复现真实数据中的控制指令，作为基准对比。
    """
    def get_policy(self):
        """
        加载 HDF5 文件中的动作日志，构造 ReplayPolicy 实例。
        Returns:
            (ReplayPolicy, exp_cfg): 策略实例及其对应的实验配置。
        """
        print("Loading action log from {}".format(self.eval_config.experience_hdf5_path))
        import h5py
        h5 = h5py.File(self.eval_config.experience_hdf5_path, "r")
        if self.eval_config.env == "nusc":
            exp_cfg = get_registered_experiment_config("nusc_bc")
        elif self.eval_config.env == "l5kit":
            exp_cfg = get_registered_experiment_config("l5_bc")
        else:
            raise NotImplementedError("invalid env {}".format(self.eval_config.env))
        return ReplayPolicy(h5, self.device), exp_cfg


class GroundTruth(PolicyComposer):
    """
    真值策略：直接回放数据集中记录的真实轨迹（而不是动作）。
    用于“上帝视角”评估，或者作为理想控制器对比。
    """
    """A fake policy that replays dataset trajectories."""
    def get_policy(self):
        if self.eval_config.env == "nusc":
            exp_cfg = get_registered_experiment_config("nusc_bc")
        elif self.eval_config.env == "l5kit":
            exp_cfg = get_registered_experiment_config("l5_bc")
        else:
            raise NotImplementedError("invalid env {}".format(self.eval_config.env))
        return GTPolicy(device=self.device), exp_cfg


class BC(PolicyComposer):
    """
    行为克隆,即模仿学习模型SimNet。
    从检查点加载预训练的 BC 模型，用于仿真中生成控制指令。
    """
    def get_policy(self, policy=None):
        if policy is not None:
            assert isinstance(policy, BehaviorCloning)
            policy_cfg = None
        else:
            policy_ckpt_path, policy_config_path = get_checkpoint(
                ckpt_key=self.eval_config.ckpt.policy.ckpt_key,
                ckpt_dir=self.eval_config.ckpt.policy.ckpt_dir,
                ckpt_root_dir=self.ckpt_root_dir,
            )
            policy_cfg = get_experiment_config_from_file(policy_config_path)
            # 加载模型
            policy = BehaviorCloning.load_from_checkpoint(
                policy_ckpt_path,
                algo_config=policy_cfg.algo,
                modality_shapes=self.get_modality_shapes(policy_cfg),
            ).to(self.device).eval() # 移至设备并设置为评估模式
            policy_cfg = policy_cfg.clone()

        return policy, policy_cfg


class Diffusion(PolicyComposer):
    """
    扩散模型策略,用于可控交通仿真 CTG。
    可生成满足特定约束（如避免碰撞、保持车道）的多智能体未来轨迹。
    """
    def get_policy(self, policy=None, modify_config=None):
        if policy is not None:
            assert isinstance(policy, DiffusionTrafficModel)
            policy_cfg = None
        else:
            print("Diffusion Class policy:",policy)
            policy_ckpt_path, policy_config_path = get_checkpoint(
                ckpt_key=self.eval_config.ckpt.planner.ckpt_key,
                ckpt_dir=self.eval_config.ckpt.planner.ckpt_dir,
                ckpt_root_dir=self.ckpt_root_dir,
            )
            policy_cfg = get_experiment_config_from_file(policy_config_path)
            # 若传入了 modify_config，则更新算法中的引导配置
            if modify_config is not None:
                update_config(policy_cfg.algo.guide_config, modify_config['guide_config'])
            # 若需要可视化扩散过程（如保存中间步骤的轨迹），则设置 ret_traj
            if modify_config.visualize_diffusion is not None:
                policy_cfg.algo.diffuse_config.ret_traj = modify_config.visualize_diffusion

            # 是否启用引导
            policy_cfg["algo"]["do_guidance"] = self.eval_config.guidance
            # 引导函数的名称列表
            policy_cfg["algo"]["guide_config"]["guidance_fn"]   = self.eval_config.guidance_fn

            # 加载扩散模型
            policy = DiffusionTrafficModel.load_from_checkpoint(
                policy_ckpt_path,
                algo_config=policy_cfg.algo,
                modality_shapes=self.get_modality_shapes(policy_cfg),
            ).to(self.device).eval()
            policy_cfg = policy_cfg.clone()

        return policy, policy_cfg
    
# -----------------------Heuristic-based planner------
class HierSplineSamplePolicy(PolicyComposer):
    def _get_predictor(self):
        predictor_ckpt_path, predictor_config_path = get_checkpoint(
            ckpt_key=self.eval_config.ckpt.predictor.ckpt_key,
            ckpt_dir=self.eval_config.ckpt.predictor.ckpt_dir,
            ckpt_root_dir=self.ckpt_root_dir
        )
        predictor_cfg = get_experiment_config_from_file(predictor_config_path)

        predictor = MATrafficModel.load_from_checkpoint(
            predictor_ckpt_path,
            algo_config=predictor_cfg.algo,
            modality_shapes=self.get_modality_shapes(predictor_cfg),
        ).to(self.device).eval()
        return predictor, predictor_cfg.clone()

    def get_policy(self, policy=None):
        predictor,predictor_cfg = self._get_predictor()
        policy = HierSplineSamplingPolicy( self.device, step_time=0.1, predictor=predictor)
        return policy, None
    
class IDMPolicy(PolicyComposer):
    def get_policy(self, policy=None):
        policy  = IDMPlanner(self.device)
        return policy, None  
class StrivePolicy(PolicyComposer):
    def get_policy(self, policy=None):
        policy  = StrivePlanner(self.device)
        return policy, None  
class StrivePolicy_trajdata(PolicyComposer):
    def get_policy(self, policy=None):
        policy  = StrivePlanner_trajdata(self.device)
        return policy, None  
class PDMPolicy(PolicyComposer):
    def get_policy(self, policy=None):
        policy  = PDMClosed(self.device)
        return policy, None  
class PDMPolicy_trajdata(PolicyComposer):
    def get_policy(self, policy=None):
        policy  = PDMClosed_trajdata(self.device)
        return policy, None  

class Hierarchical(PolicyComposer):
    """BITS (max)"""
    def _get_planner(self):
        #Hardcode
        planner_ckpt_path, planner_config_path = get_checkpoint(
            ckpt_key="iter41000",
            ckpt_dir="checkpoints/nusc_archresnet50_bs50_4130359/run0",
            ckpt_root_dir=self.ckpt_root_dir,
        )
        planner_cfg = get_experiment_config_from_file(planner_config_path)
        planner = SpatialPlanner.load_from_checkpoint(
            planner_ckpt_path,
            algo_config=planner_cfg.algo,
            modality_shapes=self.get_modality_shapes(planner_cfg),
        ).to(self.device).eval()

        return planner, planner_cfg.clone()

    def _get_gt_planner(self):
        return GTPolicy(device=self.device), None

    def _get_gt_controller(self):
        return GTPolicy(device=self.device), None

    def _get_controller(self):
        policy_ckpt_path, policy_config_path = get_checkpoint(
            ckpt_key=self.eval_config.ckpt.policy.ckpt_key,
            ckpt_dir=self.eval_config.ckpt.policy.ckpt_dir,
            ckpt_root_dir=self.ckpt_root_dir,
        )
        policy_cfg = get_experiment_config_from_file(policy_config_path)
        policy_cfg.lock()

        controller = MATrafficModel.load_from_checkpoint(
            policy_ckpt_path,
            algo_config=policy_cfg.algo,
            modality_shapes=self.get_modality_shapes(policy_cfg),
        ).to(self.device).eval()
        return controller, policy_cfg.clone()

    def get_policy(self, planner=None, controller=None):
        if planner is not None:
            assert isinstance(planner, SpatialPlanner)
        else:
            planner, exp_cfg = self._get_planner()

        if controller is not None:
            assert isinstance(controller, MATrafficModel)
            exp_cfg = None
        else:
            controller, exp_cfg = self._get_controller()
            exp_cfg = exp_cfg.clone()

        planner = PolicyWrapper.wrap_planner(
            planner,
            mask_drivable=self.eval_config.policy.mask_drivable,
            sample=False
        )
        policy = HierarchicalWrapper(planner, controller)
        return policy, exp_cfg


class HierarchicalSample(Hierarchical):
    """BITS (sample)"""
    def get_policy(self, planner=None, controller=None):
        if planner is not None:
            assert isinstance(planner, SpatialPlanner)
        else:
            planner, exp_cfg = self._get_planner()

        if controller is not None:
            assert isinstance(controller, MATrafficModel)
            exp_cfg = None
        else:
            controller, exp_cfg = self._get_controller()
            exp_cfg = exp_cfg.clone()

        planner = PolicyWrapper.wrap_planner(
            planner,
            mask_drivable=self.eval_config.policy.mask_drivable,
            sample=True
        )
        policy = HierarchicalWrapper(planner, controller)
        return policy, exp_cfg


class HierAgentAware(Hierarchical):
    """BITS"""
    def _get_predictor(self):
        predictor_ckpt_path, predictor_config_path = get_checkpoint(
            ckpt_key=self.eval_config.ckpt.predictor.ckpt_key,
            ckpt_dir=self.eval_config.ckpt.predictor.ckpt_dir,
            ckpt_root_dir=self.ckpt_root_dir
        )
        predictor_cfg = get_experiment_config_from_file(predictor_config_path)

        predictor = MATrafficModel.load_from_checkpoint(
            predictor_ckpt_path,
            algo_config=predictor_cfg.algo,
            modality_shapes=self.get_modality_shapes(predictor_cfg),
        ).to(self.device).eval()
        return predictor, predictor_cfg.clone()

    def get_policy(self, planner=None, predictor=None, controller=None):
        if planner is not None:
            assert isinstance(planner, SpatialPlanner)
        else:
            planner, _ = self._get_planner()

        if predictor is not None:
            assert isinstance(predictor, MATrafficModel)
            exp_cfg = None
        else:
            predictor, exp_cfg = self._get_predictor()
            exp_cfg = exp_cfg.clone()

        controller = predictor if controller is None else controller

        plan_sampler = PolicyWrapper.wrap_planner(
            planner,
            mask_drivable=self.eval_config.policy.mask_drivable,
            sample=True,
            num_plan_samples=self.eval_config.policy.num_plan_samples,
            clearance=self.eval_config.policy.diversification_clearance,
        )
        sampler = HierarchicalSamplerWrapper(plan_sampler, controller)

        policy = SamplingPolicyWrapper(ego_action_sampler=sampler, agent_traj_predictor=predictor)
        policy = PolicyWrapper.wrap_controller(policy, cost_weights=self.eval_config.policy.cost_weights)
        return policy, exp_cfg


class HierAgentAwareCVAE(Hierarchical):
    """Unused"""
    def _get_controller(self):
        controller_ckpt_path, controller_config_path = get_checkpoint(
            ckpt_key=self.eval_config.ckpt.policy.ckpt_key,
            ckpt_dir=self.eval_config.ckpt.policy.ckpt_dir,
            ckpt_root_dir=self.ckpt_root_dir
        )
        controller_cfg = get_experiment_config_from_file(controller_config_path)

        controller = DiscreteVAETrafficModel.load_from_checkpoint(
            controller_ckpt_path,
            algo_config=controller_cfg.algo,
            modality_shapes=self.get_modality_shapes(controller_cfg),
        ).to(self.device).eval()
        return controller, controller_cfg.clone()

    def _get_predictor(self):
        predictor_ckpt_path, predictor_config_path = get_checkpoint(
            ckpt_key=self.eval_config.ckpt.predictor.ckpt_key,
            ckpt_dir=self.eval_config.ckpt.predictor.ckpt_dir,
            ckpt_root_dir=self.ckpt_root_dir
        )
        predictor_cfg = get_experiment_config_from_file(predictor_config_path)

        predictor = MATrafficModel.load_from_checkpoint(
            predictor_ckpt_path,
            algo_config=predictor_cfg.algo,
            modality_shapes=self.get_modality_shapes(predictor_cfg),
        ).to(self.device).eval()
        return predictor, predictor_cfg.clone()

    def get_policy(self, planner=None, predictor=None, controller=None):
        if planner is not None:
            assert isinstance(predictor, MATrafficModel)
            assert isinstance(planner, SpatialPlanner)
            assert isinstance(controller, DiscreteVAETrafficModel)
            exp_cfg = None
        else:
            planner, _ = self._get_planner()
            predictor, _ = self._get_predictor()
            controller, exp_cfg = self._get_controller()
            controller = PolicyWrapper.wrap_controller(
                controller,
                sample=True,
                num_action_samples=self.eval_config.policy.num_action_samples
            )
            exp_cfg = exp_cfg.clone()

        planner = PolicyWrapper.wrap_planner(
            planner,
            mask_drivable=self.eval_config.policy.mask_drivable,
            sample=False
        )

        sampler = HierarchicalWrapper(planner, controller)
        policy = SamplingPolicyWrapper(ego_action_sampler=sampler, agent_traj_predictor=predictor)
        return policy, exp_cfg


class HierAgentAwareMPC(Hierarchical):
    def _get_predictor(self):
        predictor_ckpt_path, predictor_config_path = get_checkpoint(
            ckpt_key=self.eval_config.ckpt.predictor.ckpt_key,
            ckpt_dir=self.eval_config.ckpt.predictor.ckpt_dir,
            ckpt_root_dir=self.ckpt_root_dir
        )
        predictor_cfg = get_experiment_config_from_file(predictor_config_path)

        predictor = MATrafficModel.load_from_checkpoint(
            predictor_ckpt_path,
            algo_config=predictor_cfg.algo,
            modality_shapes=self.get_modality_shapes(predictor_cfg),
        ).to(self.device).eval()
        return predictor, predictor_cfg.clone()

    def get_policy(self, planner=None, predictor=None, initial_planner=None):
        if planner is not None:
            assert isinstance(planner, SpatialPlanner)
        else:
            planner, _ = self._get_planner()

        if predictor is not None:
            assert isinstance(predictor, MATrafficModel)
            exp_cfg = None
        else:
            predictor, exp_cfg = self._get_predictor()
            exp_cfg = exp_cfg.clone()
        exp_cfg.env.data_generation_params.vectorize_lane = True
        policy = ModelPredictiveController(self.device, exp_cfg.algo.step_time, predictor)
        return policy, exp_cfg

class HAASplineSampling(Hierarchical):
    def _get_predictor(self):
        predictor_ckpt_path, predictor_config_path = get_checkpoint(
            ckpt_key=self.eval_config.ckpt.predictor.ckpt_key,
            ckpt_dir=self.eval_config.ckpt.predictor.ckpt_dir,
            ckpt_root_dir=self.ckpt_root_dir
        )
        predictor_cfg = get_experiment_config_from_file(predictor_config_path)

        predictor = MATrafficModel.load_from_checkpoint(
            predictor_ckpt_path,
            algo_config=predictor_cfg.algo,
            modality_shapes=self.get_modality_shapes(predictor_cfg),
        ).to(self.device).eval()
        return predictor, predictor_cfg.clone()

    def get_policy(self, planner=None, predictor=None, controller=None):
        if planner is not None:
            assert isinstance(planner, SpatialPlanner)
        else:
            planner, _ = self._get_planner()

        if predictor is not None:
            assert isinstance(predictor, MATrafficModel)
            exp_cfg = None
        else:
            predictor, exp_cfg = self._get_predictor()
            exp_cfg = exp_cfg.clone()
        exp_cfg.env.data_generation_params.vectorize_lane = True
        policy = HierSplineSamplingPolicy(self.device, exp_cfg.algo.step_time, predictor)
        return policy, exp_cfg


class TreeContingency(Hierarchical):
    def _get_tree_predictor(self):
        model_choice="CNN"
        if model_choice in ["CNN","Unet"]:
            if model_choice=="CNN":

                # CNN-based predictor
                tree_ckpt_path, tree_config_path = get_checkpoint(
                    ckpt_key="",
                    ckpt_root_dir=""
                )
            elif model_choice == "Unet":
                # Unet based predictor
                tree_ckpt_path, tree_config_path = get_checkpoint(
                    ckpt_key="",
                    ckpt_root_dir=""
                )
            predictor_cfg = get_experiment_config_from_file(tree_config_path)

            predictor = SceneTreeTrafficModel.load_from_checkpoint(
                tree_ckpt_path,
                algo_config=predictor_cfg.algo,
                modality_shapes=self.get_modality_shapes(predictor_cfg),
            ).to(self.device).eval()
        elif model_choice == "agentformer":
            tree_ckpt_path, tree_config_path = get_checkpoint(
                    ckpt_key="",
                    ckpt_root_dir=""
                )
            predictor_cfg = get_experiment_config_from_file(tree_config_path)

            predictor = AgentFormerTrafficModel.load_from_checkpoint(
                tree_ckpt_path,
                algo_config=predictor_cfg.algo,
                modality_shapes=self.get_modality_shapes(predictor_cfg),
            ).to(self.device).eval()
        elif model_choice == "agentformer_G":
            tree_ckpt_path, tree_config_path = get_checkpoint(
                    ckpt_key="",
                    ckpt_root_dir=""
                )
            predictor_cfg = get_experiment_config_from_file(tree_config_path)

            predictor = AgentFormerTrafficModel.load_from_checkpoint(
                tree_ckpt_path,
                algo_config=predictor_cfg.algo,
                modality_shapes=self.get_modality_shapes(predictor_cfg),
            ).to(self.device).eval()
        elif model_choice == "agentformer_nonEC":
            tree_ckpt_path, tree_config_path = get_checkpoint(
                    ckpt_key="",
                    ckpt_root_dir=""
                )
            predictor_cfg = get_experiment_config_from_file(tree_config_path)

            predictor = AgentFormerTrafficModel.load_from_checkpoint(
                tree_ckpt_path,
                algo_config=predictor_cfg.algo,
                modality_shapes=self.get_modality_shapes(predictor_cfg),
            ).to(self.device).eval()
        elif model_choice=="kinematic":
            predictor_cfg = get_experiment_config_from_file("experiments/templates/nusc_tree.json")
            predictor = SimpleTreeModel(predictor_cfg.algo)

        return predictor, predictor_cfg.clone()

    def get_policy(self, planner=None, predictor=None, controller=None,mode="contingency"):
        if planner is not None:
            assert isinstance(planner, SpatialPlanner)
            exp_cfg = None
        else:
            planner, _ = self._get_planner()
            predictor, exp_cfg = self._get_tree_predictor()

        ego_sampler = SplinePlanner(self.device, N_seg=planner.algo_config.future_num_frames+1)
        agent_planner = PolicyWrapper.wrap_planner(
            planner,
            mask_drivable=self.eval_config.policy.mask_drivable,
            sample=False
        )
        config = Dict()
        config.stage = exp_cfg.algo.stage
        config.num_frames_per_stage = exp_cfg.algo.num_frames_per_stage
        config.step_time = exp_cfg.algo.step_time
        config.lane_type="vectorized"
        policy = ContingencyPlanner(
            ego_sampler=ego_sampler,
            predictor=predictor,
            config = config,
            agent_planner=agent_planner,
            device=self.device,
            mode = mode
        )
        exp_cfg.env.data_generation_params.vectorize_lane="ego"
        return policy, exp_cfg
class OneshotRobust(TreeContingency):
    def get_policy(self, planner=None, predictor=None, controller=None):
        return super(OneshotRobust, self).get_policy(planner=planner, predictor=predictor, controller=controller,mode="one_shot_all")

class OneshotSingle(TreeContingency):
    def get_policy(self, planner=None, predictor=None, controller=None):
        return super(OneshotSingle, self).get_policy(planner=planner, predictor=predictor, controller=controller,mode="one_shot_single")