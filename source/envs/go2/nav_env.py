import torch
from isaaclab.envs import ManagerBasedRLEnv
from assets.mdp.scene import obstacle_state
from assets.mdp.terminations import obstacle_collision


class Go2NavigationEnv(ManagerBasedRLEnv):

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self.sim.add_physics_callback('navigation_obstacles', self._advance_obstacles)

    @torch.inference_mode()
    def _advance_obstacles(self, dt):
        obstacle_state(self).advance(dt)

    def _reset_idx(self, env_ids):
        if not hasattr(self, 'episode_results'):
            self.episode_results = {name: torch.zeros(self.num_envs, device=self.device)
                for name in ('success', 'collision', 'fall', 'outside', 'timeout', 'distance', 'duration')}
        ids = env_ids[self.episode_length_buf[env_ids] > 0]
        values = None
        if len(ids):
            terms = self.termination_manager
            collision = obstacle_collision(self)[ids]
            fall = terms.get_term('base_contact')[ids] | terms.get_term('tilted')[ids]
            outside = terms.get_term('outside')[ids]
            distance = self.command_manager.get_command('goal')[ids, :2].norm(dim=-1)
            deadline = terms.get_term('navigation_deadline')[ids]
            success = deadline & (distance < 0.5) & ~collision & ~fall & ~outside
            values = dict(success=success, collision=collision, fall=fall, outside=outside,
                          timeout=deadline & ~success, distance=distance,
                          duration=self.episode_length_buf[ids] * self.step_dt)
            for name, value in values.items():
                self.episode_results[name][ids] = value.float()
        super()._reset_idx(env_ids)
        if values is not None:
            for name, value in values.items():
                self.extras['log']['BasicNav/' + name] = value.float().mean()

    def step(self, action):
        result = super().step(action)
        if getattr(self, "stream_render", False) and not (self.sim.has_gui() or self.sim.has_rtx_sensors()):
            self.sim.render()
        return result
