import torch
from stage2.inference import OnlineVOP, heading_scan
from .nav_env import Go2NavigationEnv


class VOPNavigationEnv(Go2NavigationEnv):
    def get_vop(self):
        # ObservationManager probes term dimensions during environment construction.
        if not hasattr(self, 'online_vop'):
            if not self.cfg.vop_checkpoint:
                raise ValueError('VOP-Nav requires --vop_checkpoint')
            self.online_vop = OnlineVOP(self.cfg.vop_checkpoint, self.num_envs,
                                        self.device, self.step_dt)
        sensor = self.scene['lidar']
        hit = sensor.data.ray_hits_w  # materialize the current lazy sensor buffers
        ranges = torch.nan_to_num((hit-sensor._ray_starts_w).norm(dim=-1),
                                   nan=6., posinf=6.).clamp(.12, 6.)
        scan = heading_scan(ranges, sensor._ray_directions_w,
                            self.scene['robot'].data.root_quat_w)
        self.online_vop.update(scan, self._sim_step_counter)
        return self.online_vop

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, 'online_vop'):
            self.online_vop.reset(env_ids)
