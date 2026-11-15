import math

import torch
from torch import nn
from torch.distributions import Normal

# Angular velocity(3), gravity(3), goal(3), timer(1), joints(12+12), actions(12).
STATE_DIM = 46
LIDAR_DIM = 40
DEPTH_SHAPE = (12, 14)
OBS_DIM = STATE_DIM + LIDAR_DIM + math.prod(DEPTH_SHAPE)


class ResidualFusion(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width, width), nn.LayerNorm(width), nn.ELU(),
                                    nn.Linear(width, width), nn.LayerNorm(width))

    def forward(self, value):
        return torch.nn.functional.elu(value + self.layers(value))


class BasicNavBranch(nn.Module):
    def __init__(self, output, feature_width=64, fusion_width=256):
        super().__init__()
        self.depth = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, 1), nn.ELU(),
            nn.Flatten(), nn.Linear(32 * 3 * 4, feature_width), nn.ELU(),
        )
        self.lidar = nn.Sequential(nn.Linear(LIDAR_DIM, feature_width), nn.ELU(),
                                   nn.Linear(feature_width, feature_width), nn.ELU())
        self.fusion = nn.Sequential(nn.Linear(STATE_DIM + 2 * feature_width, fusion_width), nn.ELU(),
                                    ResidualFusion(fusion_width))
        self.decoder = nn.Sequential(nn.Linear(fusion_width, 128), nn.ELU(), nn.Linear(128, output))
        nn.init.orthogonal_(self.decoder[-1].weight, gain=0.01 if output == 12 else 1.0)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, observation):
        state = observation[..., :STATE_DIM]
        lidar = observation[..., STATE_DIM:STATE_DIM + LIDAR_DIM].clamp(0.12, 6.0).log2()
        depth = observation[..., STATE_DIM + LIDAR_DIM:].reshape(-1, 1, *DEPTH_SHAPE).clamp(0.12, 6.0)
        features = torch.cat((state, self.lidar(lidar), self.depth(1.0 / depth - 1.0 / 6.0)), dim=-1)
        return self.decoder(self.fusion(features))


class BasicNavActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, num_actor_obs, num_critic_obs, num_actions, init_noise_std=1.0,
                 feature_width=64, fusion_width=256):
        super().__init__()
        if (num_actor_obs, num_critic_obs, num_actions) != (OBS_DIM, OBS_DIM, 12):
            raise ValueError(f"BasicNav requires {OBS_DIM} actor/critic observations and 12 actions; retrain old checkpoints")
        if init_noise_std <= 0:
            raise ValueError("Initial action standard deviation must be positive")
        self.actor = BasicNavBranch(12, feature_width, fusion_width)
        self.critic = BasicNavBranch(1, feature_width, fusion_width)
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        return self.actor(observations)

    def evaluate(self, observations, **kwargs):
        return self.critic(observations).float()

    def reset(self, dones=None):
        pass
