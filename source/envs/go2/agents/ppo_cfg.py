from isaaclab.utils import configclass


@configclass
class PolicyCfg:
    init_noise_std: float = 1.0
    feature_width: int = 64
    fusion_width: int = 256


@configclass
class PPOCfg:
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.003
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: str = 'adaptive'
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0


@configclass
class RunnerCfg:
    policy_class_name: str = 'BasicNavActorCritic'
    algorithm_class_name: str = 'PPO'
    num_steps_per_env: int = 48
    max_iterations: int = 10000
    save_interval: int = 1000


@configclass
class AgentCfg:
    seed: int = 1
    policy = PolicyCfg()
    algorithm = PPOCfg()
    runner = RunnerCfg()
