"""Evaluation-only environment. First arrival is measured without ending the episode."""
import hashlib
import json

import torch
from isaaclab.managers import TerminationTermCfg as DoneTerm
from envs.go2.vop_nav_env import VOPNavigationEnv
from envs.go2.vop_nav_env_cfg import VOPNavEnvCfg
from envs.go2.nav_env_cfg import SceneCfg, Go2EnvCfg
from .layouts import SCENARIOS, obstacle_specs
from .scene import clearance, outside, reset_scene, episode_seed


def probe(env):
    collision = clearance(env) < 0
    tilted = env.termination_manager.get_term('tilted')
    contact = env.termination_manager.get_term('base_contact')
    distance = env.command_manager.get_command('goal')[:, :2].norm(dim=-1)
    arrived = distance < .5
    values = dict(collision=collision, arrived=arrived, tilted=tilted,
                  base_contact=contact, outside=outside(env))
    time = env.episode_length_buf * env.step_dt
    for key, value in values.items():
        env.tracking[key] |= value
        env.event_times[key] = torch.where(value & (env.event_times[key] < 0), time, env.event_times[key])
    outcome = torch.where(collision, 2, torch.where(tilted | contact, 3,
                          torch.where(values['outside'], 4, torch.where(arrived, 1, 0))))
    env.first_outcome = torch.where(env.first_outcome == 0, outcome, env.first_outcome)
    return torch.zeros_like(collision)


def make_config(scenario, num_envs, device, seed, vop_checkpoint, level=9):
    cfg = VOPNavEnvCfg() if vop_checkpoint else Go2EnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.sim.device = device
    cfg.seed = seed
    if vop_checkpoint:
        cfg.vop_checkpoint = str(vop_checkpoint)
    cfg.evaluation_scenario = scenario
    cfg.evaluation_seed = seed
    cfg.evaluation_level = level if scenario == 'Training' else 0
    cfg.observations.policy.enable_corruption = False
    cfg.curriculum.navigation = None
    cfg.events.push_robot = None
    cfg.terminations.outside = DoneTerm(func=outside)
    cfg.terminations.evaluation_probe = DoneTerm(func=probe)  # after tilted/base_contact
    cfg.scene.terrain.terrain_generator.seed = seed
    if scenario != 'Training':
        cfg.episode_length_s = 15.
        cfg.commands.goal.episode_duration_range = (15., 15.)
        cfg.events.reset_robot.func = reset_scene
        cfg.scene.terrain.terrain_generator.num_rows = 1
        cfg.scene.terrain.terrain_generator.size = (12., 12.)
        cfg.scene.terrain.terrain_generator.sub_terrains = {
            'flat': next(iter(cfg.scene.terrain.terrain_generator.sub_terrains.values()))}
        for spec in cfg.scene.terrain.obstacle_specs:
            delattr(cfg.scene, spec['name'])
        cfg.scene.terrain.obstacle_specs = obstacle_specs()
        SceneCfg.__post_init__(cfg.scene)
    return cfg


class EvaluationEnv(VOPNavigationEnv):
    def __init__(self, cfg, **kwargs):
        # Managers may probe terms during construction; buffers must exist beforehand.
        self.rows = []
        self.initial_states = [None] * cfg.scene.num_envs
        self.episode_indices = [0] * cfg.scene.num_envs
        self.evaluation_seed = cfg.evaluation_seed
        self.record_episodes = False
        self.tracking = {k: torch.zeros(cfg.scene.num_envs, device=cfg.sim.device, dtype=torch.bool)
                         for k in ('collision', 'arrived', 'tilted', 'base_contact', 'outside')}
        self.event_times = {k: torch.full((cfg.scene.num_envs,), -1., device=cfg.sim.device)
                            for k in self.tracking}
        self.first_outcome = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device, dtype=torch.long)
        super().__init__(cfg, **kwargs)

    def prepare(self, seed, level=None):
        """Call before a full reset at each seed/action group boundary."""
        self.record_episodes = False
        self.evaluation_seed = seed
        self.episode_indices = [0] * self.num_envs
        if level is not None and self.cfg.evaluation_scenario == 'Training':
            self.cfg.evaluation_level = level
        if self.cfg.evaluation_scenario != 'Training':
            state = self.obstacle_scene_state
            state.seed = seed
            state.episodes = [0] * self.num_envs
            state.ticks = 0
        self.rows.clear()

    def _reset_idx(self, env_ids):
        # Give every (seed, env, episode) an independent RNG stream. Action sampling
        # cannot change the layouts assigned to other episodes after early resets.
        for i in env_ids.tolist():
            ids = torch.tensor([i], device=self.device, dtype=torch.long)
            if self.record_episodes and int(self.episode_length_buf[i]) > 0:
                deadline = bool(self.termination_manager.get_term('navigation_deadline')[i])
                distance = float(self.command_manager.get_command('goal')[i, :2].norm())
                collided = bool((clearance(self) < 0)[i])
                fallen = bool(self.termination_manager.get_term('tilted')[i] |
                              self.termination_manager.get_term('base_contact')[i])
                escaped = bool(outside(self)[i])
                deadline_success = deadline and distance < .5 and not (collided or fallen or escaped)
                self.rows.append(dict(
                    env_id=i, episode=self.episode_indices[i]-1,
                    initial_state=self.initial_states[i],
                    episode_seed=episode_seed(self.evaluation_seed, i, self.episode_indices[i]-1),
                    success=int(self.first_outcome[i]) == 1,
                    first_outcome=int(self.first_outcome[i]), deadline_success=deadline_success,
                    strict_deadline_success=deadline_success and not bool(self.tracking['collision'][i]),
                    collision=collided, fall=fallen, outside=escaped,
                    timeout=deadline and not deadline_success, distance=distance,
                    duration=float(self.episode_length_buf[i] * self.step_dt),
                    **{'ever_'+k: bool(v[i]) for k, v in self.tracking.items()},
                    **{k+'_time': float(v[i]) for k, v in self.event_times.items()}))
            for v in self.tracking.values(): v[ids] = False
            for v in self.event_times.values(): v[ids] = -1
            self.first_outcome[ids] = 0
            terrain = self.scene.terrain
            level = self.cfg.evaluation_level if self.cfg.evaluation_scenario == 'Training' else 0
            terrain.terrain_levels[ids] = level
            terrain.env_origins[ids] = terrain.terrain_origins[terrain.terrain_levels[ids], terrain.terrain_types[ids]]
            devices = [torch.device(self.device).index or 0] if str(self.device).startswith('cuda') else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(episode_seed(self.evaluation_seed, i, self.episode_indices[i]))
                super()._reset_idx(ids)
                if self.cfg.evaluation_scenario != 'Training':
                    goal = self.command_manager.get_term('goal')
                    width = SCENARIOS[self.cfg.evaluation_scenario]['width']
                    goal.target_pos_w[ids] = self.scene.env_origins[ids]
                    goal.target_pos_w[ids, 0] += torch.empty(1, device=self.device).uniform_(4.5, 4.6)
                    goal.target_pos_w[ids, 1] += torch.empty(1, device=self.device).uniform_(-width/2+.8, width/2-.8)
                    goal._update_command()
            state = self.obstacle_scene_state
            initial = dict(robot=(self.scene['robot'].data.root_pos_w[i] - self.scene.env_origins[i]).tolist(),
                           goal=(self.command_manager.get_term('goal').target_pos_w[i] - self.scene.env_origins[i]).tolist(),
                           terrain_level=level, terrain_type=int(terrain.terrain_types[i]))
            layout_data = dict(positions=state.positions[i].tolist(), active=state.active[i].tolist(),
                               velocities=state.velocities[i].tolist())
            if self.cfg.evaluation_scenario != 'Training':
                layout_data.update(speeds=state.speeds[i], targets=state.target[i], reactive=sorted(state.reactive[i]))
            initial['layout_sha256'] = hashlib.sha256(json.dumps(layout_data,sort_keys=True).encode()).hexdigest()
            self.initial_states[i] = initial
            self.episode_indices[i] += 1

