from isaaclab.utils import configclass
from .ppo_cfg import AgentCfg, RunnerCfg


@configclass
class VOPRunnerCfg(RunnerCfg):
    policy_class_name: str = 'VOPNavActorCritic'


@configclass
class VOPAgentCfg(AgentCfg):
    runner = VOPRunnerCfg()
