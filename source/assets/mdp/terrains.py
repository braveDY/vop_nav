from __future__ import annotations

import numpy as np
import trimesh
from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh
from isaaclab.terrains import TerrainGenerator
from isaaclab.utils import configclass


def _terrain_mesh(heights: np.ndarray, cfg: HfTerrainBaseCfg):
    padded = np.pad(heights, ((0, 1), (0, 1)), mode="edge")
    vertices, triangles = convert_height_field_to_mesh(
        padded, cfg.horizontal_scale, cfg.vertical_scale, cfg.slope_threshold
    )
    center_x, center_y = (int(size / (2 * cfg.horizontal_scale)) for size in cfg.size)
    half_width = int(0.5 / cfg.horizontal_scale)
    origin_z = heights[center_x - half_width:center_x + half_width, center_y - half_width:center_y + half_width].max()
    origin = np.array([cfg.size[0] / 2, cfg.size[1] / 2, origin_z * cfg.vertical_scale])
    return [trimesh.Trimesh(vertices=vertices, faces=triangles)], origin


def navigation_terrain(difficulty: float, cfg):
    level = min(9, int(difficulty * 10))
    shape = tuple(int(size / cfg.horizontal_scale) for size in cfg.size)
    amplitude = cfg.max_height * level / 9.0
    heights = np.zeros(shape)
    if cfg.mode == 'rough':
        heights = np.random.uniform(-amplitude, amplitude, shape)
    elif cfg.mode == 'blocks' and amplitude:
        for _ in range(20):
            x = np.random.randint(0, shape[0] - 2)
            y = np.random.randint(0, shape[1] - 2)
            heights[x:x + 2, y:y + 2] = np.random.choice([-amplitude, amplitude])
    x = np.arange(shape[0]) * cfg.horizontal_scale - cfg.size[0] / 2
    y = np.arange(shape[1]) * cfg.horizontal_scale - cfg.size[1] / 2
    heights[(abs(x) > 4.0), :] = 0.0
    heights[:, (abs(y) > 2.5)] = 0.0
    return _terrain_mesh(np.rint(heights / cfg.vertical_scale).astype(np.int16), cfg)


@configclass
class NavigationTerrainCfg(HfTerrainBaseCfg):
    function = navigation_terrain
    mode: str = 'flat'
    max_height: float = 0.07


class NavigationTerrainGenerator(TerrainGenerator):
    """Use column midpoints; the installed SDK's +.001 overflows at 1024 columns."""
    def _generate_curriculum_terrains(self):
        configs = list(self.cfg.sub_terrains.values())
        proportions = np.array([cfg.proportion for cfg in configs])
        cumulative = np.cumsum(proportions / proportions.sum())
        for col in range(self.cfg.num_cols):
            kind = int(np.searchsorted(cumulative, (col + 0.5) / self.cfg.num_cols))
            for row in range(self.cfg.num_rows):
                lower, upper = self.cfg.difficulty_range
                difficulty = lower + (upper - lower) * (row + self.np_rng.uniform()) / self.cfg.num_rows
                mesh, origin = self._get_terrain_mesh(difficulty, configs[kind])
                self._add_sub_terrain(mesh, origin, row, col, configs[kind])
