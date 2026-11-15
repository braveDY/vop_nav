import torch
from torch import nn
from .basic_nav_actor_critic import BasicNavActorCritic, BasicNavBranch, ResidualFusion

OBS_DIM = 977


class VOPNavBranch(nn.Module):
    def __init__(self, output, feature_width=64, fusion_width=256):
        super().__init__()
        basic = BasicNavBranch(output, feature_width, fusion_width)
        self.depth, self.lidar, self.decoder = basic.depth, basic.lidar, basic.decoder
        self.vop = nn.Sequential(nn.Linear(720, 128), nn.ELU(),
                                 nn.Linear(128, feature_width), nn.ELU())
        self.fusion = nn.Sequential(nn.Linear(49 + 3*feature_width, fusion_width),
                                    nn.ELU(), ResidualFusion(fusion_width))

    def forward(self, observation):
        state = torch.cat((observation[..., :46], observation[..., 254:257]), -1)
        lidar = observation[..., 46:86].clamp(.12, 6.).log2()
        depth = observation[..., 86:254].reshape(-1, 1, 12, 14).clamp(.12, 6.)
        features = torch.cat((state, self.lidar(lidar),
            self.depth(1./depth - 1./6.), self.vop(observation[..., 257:])), -1)
        return self.decoder(self.fusion(features))


class VOPNavActorCritic(BasicNavActorCritic):
    def __init__(self, num_actor_obs, num_critic_obs, num_actions, init_noise_std=1.,
                 feature_width=64, fusion_width=256):
        nn.Module.__init__(self)
        if (num_actor_obs, num_critic_obs, num_actions) != (OBS_DIM, OBS_DIM, 12):
            raise ValueError('VOP-Nav requires 977 actor/critic observations and 12 actions')
        if init_noise_std <= 0:
            raise ValueError('Initial action standard deviation must be positive')
        self.actor = VOPNavBranch(12, feature_width, fusion_width)
        self.critic = VOPNavBranch(1, feature_width, fusion_width)
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
