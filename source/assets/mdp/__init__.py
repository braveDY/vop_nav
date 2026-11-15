"""MDP terms loaded on demand so tensor helpers do not initialize Isaac Lab."""
from importlib import import_module


_TERMS = {
    'heading_linear_velocity': 'observations',
    'vop_intervals': 'observations',
    'vop_velocity_reward': 'rewards',
    'ArenaGoalCommandCfg': 'commands',
    'navigation_curriculum': 'curriculum',
    'episode_time_left': 'observations',
    'lidar_ranges': 'observations',
    'optical_depth_image': 'observations',
    'position_target': 'rewards',
    'stalled': 'rewards',
    'feet_collision': 'rewards',
    'airborne': 'rewards',
    'torque_soft_limits': 'rewards',
    'basic_agility': 'rewards',
    'basic_failure': 'rewards',
    'heading_target': 'rewards',
    'stand_at_goal': 'rewards',
    'reset_navigation_scene': 'scene',
    'navigation_time_out': 'terminations',
    'obstacle_clearance': 'terminations',
    'obstacle_collision': 'terminations',
    'outside_navigation_area': 'terminations',
}


def __getattr__(name: str) -> object:
    if name == '__all__':
        native = import_module('isaaclab.envs.mdp')
        public = getattr(native, '__all__', [key for key in vars(native) if not key.startswith('_')])
        value = list(dict.fromkeys([*public, *_TERMS]))
    elif name in {'geometry', 'reward_math', 'terrains'}:
        value = import_module(f'{__name__}.{name}')
    elif name in _TERMS:
        value = getattr(import_module(f'{__name__}.{_TERMS[name]}'), name)
    elif name.startswith('_'):
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    else:
        try:
            value = getattr(import_module('isaaclab.envs.mdp'), name)
        except AttributeError as error:
            raise AttributeError(f'module {__name__!r} has no attribute {name!r}') from error
    globals()[name] = value
    return value
