import math

import torch
from isaaclab.utils.math import quat_from_euler_xyz

from assets.mdp.geometry import reflect_motion


class ObstacleSceneState:
    def __init__(self, env):
        self.env = env
        self.cfg = env.cfg.scene.terrain
        self.specs = self.cfg.obstacle_specs
        self.assets = [env.scene[spec['name']] for spec in self.specs]
        count = len(self.assets)
        self.positions = torch.zeros(env.num_envs, count, 2, device=env.device)
        self.velocities = torch.zeros_like(self.positions)
        self.active = torch.zeros(env.num_envs, count, dtype=torch.bool, device=env.device)
        self.radii = torch.tensor([s['radius'] for s in self.specs], device=env.device)
        self.half_sizes = torch.tensor([s['half_size'] for s in self.specs], device=env.device)
        self.heights = torch.tensor([s['height'] for s in self.specs], device=env.device)
        self.outer_radii = torch.where(self.radii > 0, self.radii, self.half_sizes.norm(dim=-1))
        # Place the large footprints before small objects fragment free space.
        self.placement_order = self.outer_radii.argsort(descending=True).tolist()
        self.substeps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        # Initial placements use the central zone; moving people can later enter
        # either end buffer and reflect at the full 10x5m arena boundaries.
        self.lower = torch.tensor([-4.5, -2.0], device=env.device)
        self.upper = -self.lower

    def _write(self, index, ids):
        asset = self.assets[index]
        state = asset.data.default_root_state[ids].clone()
        state[:, :2] = self.positions[ids, index] + self.env.scene.env_origins[ids, :2]
        state[:, 2] = self.env.scene.env_origins[ids, 2] + self.heights[index] / 2 + 0.08
        state[~self.active[ids, index], 2] -= 30.0
        state[:, 7:] = 0.0
        state[:, 7:9] = self.velocities[ids, index]
        asset.write_root_pose_to_sim(state[:, :7], env_ids=ids)
        asset.write_root_velocity_to_sim(state[:, 7:], env_ids=ids)

    def draw_velocity(self, ids):
        level = self.env.scene.terrain.terrain_levels[ids].float()
        maximum = torch.maximum(torch.full_like(level, self.cfg.dynamic_min_speed),
                                0.1 * (level + 1) * self.cfg.dynamic_max_speed)
        minimum = torch.where(level >= 4, 0.5, 0.5 * maximum)
        speed = minimum[:, None] + (maximum - minimum)[:, None] * torch.rand(len(ids), 8, device=self.env.device)
        angle = torch.rand_like(speed) * (2 * math.pi)
        return speed[..., None] * torch.stack((angle.cos(), angle.sin()), dim=-1)

    def reset(self, ids):
        env = self.env
        level = env.scene.terrain.terrain_levels[ids]
        self.active[ids] = False
        self.velocities[ids] = 0.0
        self.substeps[ids] = 0
        if getattr(env, 'evaluation_empty', False):
            dynamic_count = static_count = box_count = torch.zeros_like(level)
        else:
            dynamic_count = torch.div(8 * (level + 1), 10, rounding_mode='floor')
            static_count = torch.div(2 * (level + 1), 10, rounding_mode='floor')
            box_count = torch.div(4 * (level + 1), 10, rounding_mode='floor')
        self.active[ids, :8] = torch.arange(8, device=env.device)[None] < dynamic_count[:, None]
        self.active[ids, 8:10] = torch.arange(2, device=env.device)[None] < static_count[:, None]
        choices = torch.randint(3, (len(ids), 4), device=env.device)
        for slot in range(4):
            self.active[ids, 10 + slot * 3 + choices[:, slot]] = slot < box_count
        # A greedy prefix can leave no room for the last object. Restart that
        # environment's whole layout rather than silently accepting an overlap.
        pending_layouts = ids
        for _ in range(32):
            failed = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
            for slot, index in enumerate(self.placement_order):
                pending = pending_layouts
                for _ in range(32):
                    if not len(pending):
                        break
                    self.positions[pending, index, 0] = torch.empty(len(pending), device=env.device).uniform_(-3.4, 3.4)
                    extent = min(2.5 - float(self.outer_radii[index]) - 0.08, 2.0)
                    self.positions[pending, index, 1] = torch.empty(len(pending), device=env.device).uniform_(-extent, extent)
                    invalid = torch.zeros(len(pending), device=env.device, dtype=torch.bool)
                    if slot:
                        placed = self.placement_order[:slot]
                        distances = (self.positions[pending[:, None], placed] - self.positions[pending, index, None]).norm(dim=-1)
                        required = self.outer_radii[placed] + self.outer_radii[index] + 0.08
                        invalid = ((distances < required) & self.active[pending[:, None], placed]).any(-1)
                    pending = pending[invalid & self.active[pending, index]]
                failed[pending] = True
            pending_layouts = pending_layouts[failed[pending_layouts]]
            if not len(pending_layouts):
                break
        if len(pending_layouts):
            raise RuntimeError('Obstacle initial packing failed after 32 complete layout attempts')
        self.velocities[ids, :8] = self.draw_velocity(ids) * self.active[ids, :8, None]
        if getattr(env, 'evaluation_static', False):
            self.velocities[ids] = 0.0
        for index in range(len(self.assets)):
            self._write(index, ids)
        for index, name in enumerate(('wall_left', 'wall_right')):
            wall = env.scene[name]
            state = wall.data.default_root_state[ids].clone()
            state[:, :3] = env.scene.env_origins[ids]
            state[:, 1] += 2.6 if index == 0 else -2.6
            state[:, 2] += 1.0
            state[:, 7:] = 0
            wall.write_root_pose_to_sim(state[:, :7], env_ids=ids)
            wall.write_root_velocity_to_sim(state[:, 7:], env_ids=ids)

    def advance(self, dt: float):
        env = self.env
        ids = torch.arange(env.num_envs, device=env.device)
        self.substeps += 1
        # 100 physics steps = 0.5 seconds at dt=0.005 (paper's simulation-step interval).
        update_ids = ((self.substeps % 100) == 0).nonzero().flatten()
        if len(update_ids) and not getattr(env, 'evaluation_static', False):
            velocity = self.velocities[update_ids, :8]
            stopped = velocity.norm(dim=-1) < 1e-6
            level = env.scene.terrain.terrain_levels[update_ids].float()[:, None]
            change = torch.rand_like(level.expand(-1, 8)) < 0.05 * level
            accelerate = torch.rand_like(change.float()) < 0.9
            delta = (torch.rand_like(velocity) - 0.5) * self.cfg.dynamic_max_speed * 0.1 * (level[..., None] + 1)
            velocity = torch.where((change & accelerate)[..., None], velocity + delta, velocity)
            velocity = torch.where((change & ~accelerate)[..., None], torch.zeros_like(velocity), velocity)
            velocity = torch.where(stopped[..., None], self.draw_velocity(update_ids), velocity)
            self.velocities[update_ids, :8] = velocity * self.active[update_ids, :8, None]
        position, velocity = reflect_motion(self.positions[:, :8], self.velocities[:, :8],
                                             self.lower, self.upper, dt)
        self.positions[:, :8] = position
        self.velocities[:, :8] = velocity
        for index in range(8):
            self._write(index, ids)


def obstacle_state(env) -> ObstacleSceneState:
    if not hasattr(env, 'obstacle_scene_state'):
        env.obstacle_scene_state = ObstacleSceneState(env)
    return env.obstacle_scene_state


def reset_navigation_scene(env, env_ids) -> None:
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    robot = env.scene['robot']
    state = robot.data.default_root_state[ids].clone()
    state[:, :3] += env.scene.env_origins[ids]
    state[:, 0] += torch.empty(len(ids), device=env.device).uniform_(-5.0, -4.5)
    state[:, 1] += torch.empty(len(ids), device=env.device).uniform_(-2.0, 2.0)
    zero = torch.zeros(len(ids), device=env.device)
    level = env.scene.terrain.terrain_levels[ids].float()
    yaw_range = math.pi * (0.25 + 0.75 * level / 9.)
    yaw = torch.empty_like(zero).uniform_(-1., 1.) * yaw_range
    state[:, 3:7] = quat_from_euler_xyz(zero, zero, yaw)
    state[:, 7:] = 0.0
    robot.write_root_pose_to_sim(state[:, :7], env_ids=ids)
    robot.write_root_velocity_to_sim(state[:, 7:], env_ids=ids)
    obstacle_state(env).reset(ids)
