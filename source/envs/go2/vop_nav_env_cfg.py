from isaaclab.managers import ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from assets import mdp
from .nav_env_cfg import Go2EnvCfg, ObservationsCfg, RewardsCfg


@configclass
class VOPObservationsCfg:
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        linear_velocity = ObsTerm(func=mdp.heading_linear_velocity)
        safe_velocities = ObsTerm(func=mdp.vop_intervals)

    @configclass
    class CriticCfg(PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = False

    policy = PolicyCfg()
    critic = CriticCfg()


@configclass
class VOPRewardsCfg(RewardsCfg):
    vo_velocity = RewTerm(func=mdp.vop_velocity_reward, weight=1.)


@configclass
class VOPNavEnvCfg(Go2EnvCfg):
    vop_checkpoint: str = ''
    observations = VOPObservationsCfg()
    rewards = VOPRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.rewards.forward_progress.weight = 15.


@configclass
class VOPNavPlayEnvCfg(VOPNavEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.curriculum.navigation.level = 5
        self.observations.policy.enable_corruption = False
