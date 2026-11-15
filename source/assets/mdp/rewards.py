from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

from assets.mdp.reward_math import agility_reward
from assets.mdp.terminations import obstacle_collision

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def heading_target(
    env: ManagerBasedRLEnv, command_name: str, sigma: float, distance_threshold: float, duration: float
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    near_goal = torch.linalg.vector_norm(command[:, :2], dim=-1) < distance_threshold
    remaining = env.command_manager.get_term(command_name).remaining_time
    return (1.0 / (1.0 + (command[:, 2] / sigma).square())) * near_goal * (remaining <= duration) / duration


def stand_at_goal(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg,
    distance_threshold: float, duration: float,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    distance = torch.linalg.vector_norm(env.command_manager.get_command(command_name)[:, :2], dim=-1)
    remaining = env.command_manager.get_term(command_name).remaining_time
    return error.abs().sum(dim=-1) * (distance < distance_threshold) * (remaining <= duration) / duration


def position_target(
    env: ManagerBasedRLEnv, command_name: str, sigma: float, duration: float
) -> torch.Tensor:
    distance = torch.linalg.vector_norm(env.command_manager.get_command(command_name)[:, :2], dim=-1)
    remaining = env.command_manager.get_term(command_name).remaining_time
    return (1.0 / (1.0 + (distance / sigma).square())) * (remaining <= duration) / duration


def stalled(env: ManagerBasedRLEnv, command_name: str, distance_threshold: float) -> torch.Tensor:
    asset = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    distance = torch.linalg.vector_norm(command[:, :2], dim=-1)
    static = (torch.linalg.vector_norm(asset.data.root_lin_vel_b[:, :2], dim=-1) < 0.1)
    static &= asset.data.root_ang_vel_b[:, 2].abs() < 0.1
    return static * (command[:, 0] / distance.clamp_min(1.0e-6) < -0.25) * (distance > distance_threshold)


def feet_collision(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids]
    return (torch.linalg.vector_norm(forces[..., :2], dim=-1) > 2.0 * forces[..., 2].abs() + 10.0).any(dim=-1)


def airborne(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2]
    supported = (forces.amax(dim=1) > threshold).any(dim=-1)
    return ~supported & (env.episode_length_buf * env.step_dt > 0.5)


def torque_soft_limits(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, soft_limit: float = 0.85
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    effort = asset.data.applied_torque[:, asset_cfg.joint_ids].abs()
    limits = asset.data.joint_effort_limits[:, asset_cfg.joint_ids]
    return (effort - soft_limit * limits).clamp_min(0.0).sum(dim=-1)


def basic_agility(env) -> torch.Tensor:
    return agility_reward(env.scene['robot'].data.root_lin_vel_b[:, 0],
                          env.command_manager.get_command('goal')[:, :2])


def basic_failure(env) -> torch.Tensor:
    terms = env.termination_manager
    failure = torch.stack([terms.get_term(name) for name in
                           ('base_contact', 'tilted', 'outside')], dim=-1).any(-1)
    return failure | obstacle_collision(env)


def vop_velocity_reward(env) -> torch.Tensor:
    """Penalize actual velocity outside predicted bounds after history warmup."""
    from .observations import heading_linear_velocity
    from .reward_math import velocity_interval_reward
    vop = env.get_vop()
    value = velocity_interval_reward(heading_linear_velocity(env), vop.intervals)
    return torch.where(vop.valid, value, 0.)
