from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def episode_time_left(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return (env.command_manager.get_term(command_name).remaining_time / env.max_episode_length_s).unsqueeze(-1)


def lidar_ranges(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Unaugmented LiDAR sensor-origin ranges in metres."""
    sensor = env.scene[sensor_cfg.name]
    offset = torch.tensor(sensor.cfg.offset.pos, device=env.device).expand(env.num_envs, -1)
    origin = sensor.data.pos_w + quat_apply(sensor.data.quat_w, offset)
    distance = (sensor.data.ray_hits_w - origin[:, None]).norm(dim=-1)
    return torch.nan_to_num(distance, nan=6.0, posinf=6.0).clamp(0.12, 6.0)


def optical_depth_image(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Flatten native camera optical-axis depth in metres, preserving pixel order."""
    image = env.scene[sensor_cfg.name].data.output['distance_to_image_plane']
    return torch.nan_to_num(image.flatten(start_dim=1), nan=6.0, posinf=6.0).clamp(0.12, 6.0)


def heading_linear_velocity(env) -> torch.Tensor:
    """Robot linear velocity expressed in the yaw-only heading frame."""
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat
    robot = env.scene['robot'].data
    return quat_apply_inverse(yaw_quat(robot.root_quat_w), robot.root_lin_vel_w)


def vop_intervals(env) -> torch.Tensor:
    vop = env.get_vop()
    return vop.intervals.flatten(1) / vop.maximum
