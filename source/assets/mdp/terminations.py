from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from assets.mdp.geometry import footprint_clearances
from .scene import obstacle_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def navigation_time_out(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return env.command_manager.get_term(command_name).remaining_time <= 0.0


def obstacle_clearance(env) -> torch.Tensor:
    """Minimum conservative body-footprint clearance, including the two side walls."""
    scene = obstacle_state(env)
    robot_xy = env.scene['robot'].data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    clearance = footprint_clearances(robot_xy, scene.positions, scene.radii,
                                    scene.half_sizes, scene.active).amin(-1)
    wall = 2.5 - robot_xy[:, 1].abs() - 0.35
    return torch.minimum(clearance, wall)


def obstacle_collision(env) -> torch.Tensor:
    return obstacle_clearance(env) < 0.0


def outside_navigation_area(env) -> torch.Tensor:
    local = env.scene['robot'].data.root_pos_w - env.scene.env_origins
    return local[:, 0].abs() > 5.5
