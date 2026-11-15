from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class NavigationGoalCommand(CommandTerm):

    def __init__(self, cfg: NavigationGoalCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_heading_w = torch.zeros(self.num_envs, device=self.device)
        self.episode_duration = torch.full((self.num_envs,), env.max_episode_length_s, device=self.device)
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self.metrics["distance_to_goal"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        self._update_command()
        return self._command

    @property
    def remaining_time(self) -> torch.Tensor:
        return self.episode_duration - self._env.episode_length_buf * self._env.step_dt

    def _update_command(self):
        orientation = yaw_quat(self._asset.data.root_quat_w)
        displacement = self.target_pos_w - self._asset.data.root_pos_w
        self._command[:, :2] = quat_apply_inverse(orientation, displacement)[:, :2]
        heading = 2.0 * torch.atan2(orientation[:, 3], orientation[:, 0])
        self._command[:, 2] = wrap_to_pi(self.target_heading_w - heading)

    def _update_metrics(self):
        self.metrics["distance_to_goal"][:] = torch.linalg.vector_norm(self.command[:, :2], dim=-1)


@configclass
class NavigationGoalCommandCfg(CommandTermCfg):
    class_type: type = NavigationGoalCommand
    asset_name: str = "robot"
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    episode_duration_range: tuple[float, float] = (7.0, 9.0)


class ArenaGoalCommand(NavigationGoalCommand):
    def _resample_command(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        # The obstacle zone ends before x=4, leaving a clear goal buffer.
        self.target_pos_w[ids] = self._env.scene.env_origins[ids]
        self.target_pos_w[ids, 0] += torch.empty(len(ids), device=self.device).uniform_(4.5, 5.0)
        self.target_pos_w[ids, 1] += torch.empty(len(ids), device=self.device).uniform_(-2.0, 2.0)
        self.target_heading_w[ids] = 0.0
        self.episode_duration[ids] = torch.empty(len(ids), device=self.device).uniform_(*self.cfg.episode_duration_range)
        self._update_command()


@configclass
class ArenaGoalCommandCfg(NavigationGoalCommandCfg):
    class_type: type = ArenaGoalCommand
