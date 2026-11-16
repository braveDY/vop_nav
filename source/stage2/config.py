from dataclasses import dataclass
import math
from pathlib import Path


def input_config(control_dt, ranges):
    return dict(history=5, rays=40, order='oldest_to_current',
                control_dt_s=control_dt, range_m=list(ranges),
                lidar='raw geometric ranges resampled to heading bins [-pi,pi)',
                history_alignment='each scan in its own heading frame; no ego-motion compensation')


def label_config(horizon, maximum, radius):
    return dict(horizon_s=horizon, maximum_speed_mps=maximum, directions=360,
                geometry='rounded_box', empty_interval=[0., 0.], robot_radius_m=radius,
                angle_start_rad=-math.pi, angle_step_rad=math.pi / 180)


def validate_contract(saved, *, control_dt, ranges, radius, maximum, range_atol=0.):
    inputs, labels = saved['input_config'], saved['label_config']
    expected = input_config(control_dt, ranges)
    if any(inputs.get(k) != v for k, v in expected.items() if k not in ('control_dt_s', 'range_m')):
        raise ValueError('unsupported checkpoint LiDAR input definition')
    if (not math.isclose(inputs['control_dt_s'], control_dt)
            or len(inputs['range_m']) != len(ranges)
            or any(not math.isclose(a, b, rel_tol=0., abs_tol=range_atol)
                   for a, b in zip(inputs['range_m'], ranges))):
        raise ValueError('control period or LiDAR range differs from checkpoint')
    expected = label_config(labels['horizon_s'], maximum, radius)
    for key, value in expected.items():
        actual = labels.get(key)
        matches = math.isclose(actual, value) if isinstance(value, float) else actual == value
        if not matches:
            raise ValueError(f'checkpoint VO label definition differs: {key}')


@dataclass
class ModelConfig:
    channels: tuple = (24, 48, 64)
    hidden: int = 256
    maximum: float = 4.5

    def __post_init__(self):
        self.channels = tuple(self.channels)
        if len(self.channels) != 3 or min(self.channels) <= 0 or self.hidden <= 0:
            raise ValueError('three positive channel sizes and positive hidden size required')
        if not math.isfinite(self.maximum) or self.maximum <= 0:
            raise ValueError('positive finite maximum required')


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    alpha: float = .2
    seed: int = 123
    workers: int = 2
    num_threads: int = 4
    device: str = 'cuda:0'
    tensorboard: bool = False

    def __post_init__(self):
        if min(self.epochs, self.batch_size, self.num_threads) <= 0 or self.workers < 0 or self.seed < 0:
            raise ValueError('invalid epochs, batch size, threads, workers or seed')
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('learning_rate must be finite and positive')
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0 or not 0 <= self.alpha <= 1:
            raise ValueError('invalid weight decay or loss weight')


PROJECT = Path(__file__).resolve().parents[2]
CONFIGS = dict(collect='collect', train_vop='train', compare='compare')


def read_yaml(path):
    from pathlib import Path
    import yaml
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = PROJECT / path
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f'YAML must contain a mapping: {path}')
    return value


def merge_config(base, values, prefix=''):
    import copy
    result = copy.deepcopy(base)
    for key, value in values.items():
        if key not in base:
            raise ValueError(f'Unknown configuration field: {prefix}{key}')
        if isinstance(base[key], dict) and isinstance(value, dict):
            result[key] = merge_config(base[key], value, prefix + key + '.') if base[key] else value
        else:
            expected = base[key]
            if expected is not None:
                valid = (isinstance(value, (float, int)) and not isinstance(value, bool)
                         if isinstance(expected, float) else type(value) is type(expected))
                if not valid:
                    raise ValueError(f'Invalid type for configuration field: {prefix}{key}')
            result[key] = value
    return result


def parse_config(kind, argv=None):
    import argparse
    from datetime import datetime
    import os
    from pathlib import Path
    default = PROJECT / 'configs' / 'stage2' / (CONFIGS[kind] + '.yaml')
    parser = argparse.ArgumentParser(description=f'Run {kind} from a YAML configuration')
    parser.add_argument('--config', type=Path, default=default)
    parser.add_argument('--device')
    parser.add_argument('--run-dir', dest='run_dir', type=Path)
    parser.add_argument('--name')
    if kind == 'collect':
        parser.add_argument('--num-envs', dest='num_envs', type=int)
    if kind == 'train_vop':
        parser.add_argument('--epochs', type=int)
    overrides = vars(parser.parse_args(argv))
    selected = overrides.pop('config')
    try:
        base = read_yaml(default)
        data = merge_config(base, read_yaml(selected))
        data.update({key: value for key, value in overrides.items() if value is not None})
        if not isinstance(data.get('name', ''), str) or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in data.get('name', '')):
            raise ValueError('name can only contain English letters, numbers, _ and -')
        host_root = os.environ.get('HOST_PROJECT_ROOT', '')
        for key in ('checkpoint', 'dataset', 'input', 'run_dir', 'prepared_dir'):
            value = data.get(key)
            if value is None:
                continue
            value = Path(value).expanduser()
            if host_root and value.is_absolute() and value.is_relative_to(host_root):
                value = PROJECT / value.relative_to(host_root)
            data[key] = (PROJECT / value).resolve() if not value.is_absolute() else value.resolve()
        if data.get('run_dir') is None:
            directory = dict(collect='stage2/datasets', train_vop='stage2/training',
                             compare='stage2/comparison')[kind]
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            data['run_dir'] = PROJECT / 'logs' / directory / (stamp + ('_' + data['name'] if data['name'] else ''))
        for key in ('num_envs', 'epochs', 'batch_size', 'num_threads'):
            if key in data and (type(data[key]) is not int or data[key] <= 0):
                raise ValueError(f'{key} must be a positive integer')
        data['_kind'] = kind
        return argparse.Namespace(**data)
    except (OSError, TypeError, KeyError, ValueError) as error:
        parser.error(str(error))


def plain_config(value):
    from pathlib import Path
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: plain_config(item) for key,item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_config(item) for item in value]
    return value


def save_config(args, directory):
    from pathlib import Path
    import yaml
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    kind = args._kind
    fields = read_yaml(PROJECT / 'configs' / 'stage2' / (CONFIGS[kind] + '.yaml'))
    values = {key: plain_config(getattr(args, key)) for key in fields}
    (directory / 'resolved.yaml').write_text(yaml.safe_dump(values, sort_keys=False))
