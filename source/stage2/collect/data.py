import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from ..config import input_config, label_config
from ..progress import print_progress
from ..geometry import heading_lidar, make_labels


def check_block(data, num_envs, steps, previous=None):
    shapes = {
        'global_step': (steps,), 'episode_id': (steps, num_envs),
        'episode_step': (steps, num_envs), 'history_valid': (steps, num_envs),
        'lidar_ranges': (steps, num_envs, 40), 'lidar_hit_valid': (steps, num_envs, 40),
        'lidar_origin_w': (steps, num_envs, 3),
        'lidar_directions_w': (steps, num_envs, 40, 3),
        'policy_lidar_ranges': (steps, num_envs, 40),
        'robot_root_state_w': (steps, num_envs, 13),
        'joint_positions': (steps, num_envs, 12), 'joint_velocities': (steps, num_envs, 12),
        'goal_position_w': (steps, num_envs, 3), 'actions': (steps, num_envs, 12),
        'obstacle_positions_env': (steps, num_envs, 22, 2),
        'obstacle_velocities_w': (steps, num_envs, 22, 2),
        'obstacle_active': (steps, num_envs, 22),
        'done_after_step': (steps, num_envs), 'truncated_after_step': (steps, num_envs),
        'terminated_after_step': (steps, num_envs), 'reward': (steps, num_envs),
        'terminal_success': (steps, num_envs), 'terminal_collision': (steps, num_envs),
        'terminal_fall': (steps, num_envs), 'terminal_outside': (steps, num_envs),
        'env_origins_w': (num_envs, 3), 'curriculum_level': (num_envs,),
        'wall_root_state_w': (num_envs, 2, 13),
    }
    for name, shape in shapes.items():
        assert name in data, f'missing {name}'
        value = data[name]
        assert value.shape == shape, (name, value.shape, shape)
        if value.dtype.kind == 'f':
            assert np.isfinite(value).all(), f'nonfinite {name}'
    for name in ('lidar_ranges', 'policy_lidar_ranges'):
        assert ((data[name] >= 0.12 - 1e-6) & (data[name] <= 6.0 + 1e-6)).all(), name
    assert np.array_equal(data['done_after_step'],
                          data['terminated_after_step'] | data['truncated_after_step'])
    assert np.array_equal(data['history_valid'], data['episode_step'] >= 4)
    # Validate the recorded scan itself, without assuming native yaw-mode rotation.
    q = data['robot_root_state_w'][..., 3:7]
    np.testing.assert_allclose(np.linalg.norm(q, axis=-1), 1.0, atol=1e-4)
    directions = data['lidar_directions_w']
    np.testing.assert_allclose(np.linalg.norm(directions, axis=-1), 1.0, atol=2e-5, rtol=0)
    np.testing.assert_allclose(directions[..., 2], 0.0, atol=2e-5, rtol=0)
    angles = np.arctan2(directions[..., 1], directions[..., 0])
    spacing = (np.roll(angles, -1, axis=-1) - angles + np.pi) % (2 * np.pi) - np.pi
    np.testing.assert_allclose(spacing, 2 * np.pi / 40, atol=2e-5, rtol=0)
    if previous is None:
        assert np.all(data['episode_id'][0] == 0)
        assert np.all(data['episode_step'][0] == 0)
        assert data['global_step'][0] == 0
    else:
        _check_transition(previous, {k: data[k][0] for k in
                          ('global_step', 'episode_id', 'episode_step')})
    assert np.all(np.diff(data['global_step']) == 1)
    for i in range(1, steps):
        _check_transition({k: data[k][i - 1] for k in
                           ('global_step', 'episode_id', 'episode_step', 'done_after_step')},
                          {k: data[k][i] for k in ('global_step', 'episode_id', 'episode_step')})
    return {k: data[k][-1].copy() for k in
            ('global_step', 'episode_id', 'episode_step', 'done_after_step')}


def _check_transition(old, new):
    done = old['done_after_step']
    assert new['global_step'] == old['global_step'] + 1
    assert np.array_equal(new['episode_id'], old['episode_id'] + done)
    expected_step = np.where(done, 0, old['episode_step'] + 1)
    assert np.array_equal(new['episode_step'], expected_step), 'episode boundary/step mismatch'


def validate_dataset(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    n, steps = manifest['num_envs'], manifest['steps_per_file']
    assert len(manifest['files']) == manifest['requested_files']
    assert len(list(directory.glob('raw_*.npz'))) == manifest['requested_files']
    previous, valid, done_count, total_bytes = None, 0, 0, 0
    levels = {}
    for record in manifest['files']:
        path = directory / record['name']
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == record['sha256'], f'checksum mismatch: {path}'
        with np.load(path, allow_pickle=False) as data:
            previous = check_block(data, n, steps, previous)
            valid += int(data['history_valid'].sum())
            done_count += int(data['done_after_step'].sum())
            for level in np.unique(data['curriculum_level']):
                mask = data['curriculum_level'] == level
                key = str(int(level))
                row = levels.setdefault(key, {'records': 0, 'completed_episodes': 0,
                                               'successes': 0, 'collisions': 0, 'falls': 0})
                row['records'] += int(mask.sum()) * steps
                row['completed_episodes'] += int(data['done_after_step'][:, mask].sum())
                for field, key2 in [('terminal_success', 'successes'),
                                    ('terminal_collision', 'collisions'), ('terminal_fall', 'falls')]:
                    row[key2] += int(data[field][:, mask].sum())
        total_bytes += path.stat().st_size
    return dict(passed=True, files=len(manifest['files']), num_envs=n, steps_per_file=steps,
                environment_steps=len(manifest['files']) * n * steps,
                history_valid_samples=valid, completed_episodes=done_count, bytes=total_bytes,
                levels=levels, labels_present=False,
                checks=['all_files_read', 'sha256', 'shapes', 'finite_values', 'range_bounds',
                        'unit_horizontal_rays', 'ray_angular_spacing',
                        'within_and_across_file_episode_continuity'])


def split_environments(levels, fraction, seed):
    """Stratify by level and keep each environment entirely in one split."""
    if not 0 < fraction < 1 or len(levels) < 2:
        raise ValueError('need at least two environments and 0 < val_fraction < 1')
    rng = np.random.default_rng(seed)
    validation = []
    for level in np.unique(levels):
        ids = rng.permutation(np.flatnonzero(levels == level))
        if len(ids) >= 2:
            validation.extend(ids[:max(1, min(len(ids) - 1, round(len(ids) * fraction)))].tolist())
        elif rng.random() < fraction:
            validation.extend(ids.tolist())
    if not validation:
        validation = [int(rng.integers(len(levels)))]
    if len(validation) == len(levels):
        validation.pop()
    validation = sorted(validation)
    return {'train': sorted(set(range(len(levels))) - set(validation)), 'val': validation}


def prepare_dataset(source, output, *, horizon=1., maximum=4.5, val_fraction=.1,
                    seed=123, shard_size=8192, label_batch_size=64, device='cpu', max_samples_per_split=None,
                    check_stop=None, on_progress=None):
    """Publish manifest.json only after all selected raw data has been prepared."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if check_stop is None:
        check_stop = lambda: None
    check_stop()
    if not (math.isfinite(horizon) and math.isfinite(maximum) and horizon > 0 and maximum > 0):
        raise ValueError('positive finite horizon and maximum required')
    if min(shard_size, label_batch_size) <= 0 or seed < 0:
        raise ValueError('positive shard/batch sizes and nonnegative seed required')
    if max_samples_per_split is not None and max_samples_per_split <= 0:
        raise ValueError('max_samples_per_split must be positive')
    if source == output or source in output.parents:
        raise ValueError('prepared data must be outside the raw collection')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'output must be empty: {output}')
    raw_bytes = (source / 'manifest.json').read_bytes()
    raw = json.loads(raw_bytes)
    if len(raw['files']) != raw['requested_files'] or len(list(source.glob('raw_*.npz'))) != raw['requested_files']:
        raise ValueError('raw collection is incomplete; wait for collection to finish')
    acceptance = source / 'acceptance.json'
    if not acceptance.is_file() or not json.loads(acceptance.read_text()).get('passed'):
        raise ValueError('raw collection acceptance missing or failed; wait for successful collection validation')
    if raw.get('obstacle_orientation_wxyz') != [1., 0., 0., 0.]:
        raise ValueError('full-boundary labels require the recorded axis-aligned obstacles')
    output.mkdir(parents=True, exist_ok=True)
    package = Path(__file__).resolve().parents[1]
    info = dict(schema_version=1, source=str(source), source_manifest_sha256=hashlib.sha256(raw_bytes).hexdigest(),
                implementation_sha256={name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                                       for name in ('collect/data.py', 'geometry.py', 'config.py')},
                label_config=label_config(horizon, maximum, raw['robot_collision_proxy_radius_m']),
                input_config=input_config(raw['control_dt_s'], raw['lidar_range_m']),
                seed=seed, val_fraction=val_fraction, max_samples_per_split=max_samples_per_split,
                subset=max_samples_per_split is not None, splits={}, shards=[], raw_files_verified=[])
    counts = {'train': 0, 'val': 0}
    previous, tail, tail_episode = None, None, None
    origins = levels = walls = None
    started = time.monotonic()
    written_bytes = 0
    for record in raw['files']:
        file_started = time.monotonic()
        previous_samples = sum(counts.values())
        check_stop()
        path = (source / record['name']).resolve()
        if path.parent != source or path.suffix != '.npz':
            raise ValueError(f'invalid raw block path: {path}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f'checksum mismatch: {path}')
        with np.load(path, allow_pickle=False) as archive:
            block = {key: archive[key] for key in archive.files}
        previous = check_block(block, raw['num_envs'], raw['steps_per_file'], previous)
        if levels is None:
            origins, levels, walls = (block[k].copy() for k in ('env_origins_w', 'curriculum_level', 'wall_root_state_w'))
            info['splits'] = split_environments(levels, val_fraction, seed)
        elif not all(np.array_equal(a, b) for a, b in ((origins, block['env_origins_w']),
                      (levels, block['curriculum_level']), (walls, block['wall_root_state_w']))):
            raise ValueError('environment geometry/levels changed between files')
        scans = heading_lidar(block)
        tail_length = 0 if tail is None else len(tail)
        history = scans if tail is None else np.concatenate((tail, scans))
        episodes = block['episode_id'] if tail is None else np.concatenate((tail_episode, block['episode_id']))
        for split, ids in info['splits'].items():
            times, envs = np.nonzero(block['history_valid'] & np.isin(np.arange(raw['num_envs']), ids)[None])
            if max_samples_per_split is not None:
                remaining = max(0, max_samples_per_split - counts[split])
                times, envs = times[:remaining], envs[:remaining]
            for start in range(0, len(times), shard_size):
                check_stop()
                t, e = times[start:start + shard_size], envs[start:start + shard_size]
                end = tail_length + t
                if (end < 4).any():
                    raise ValueError('valid sample has insufficient history')
                window = np.stack([history[end - 4 + k, e] for k in range(5)], 1)
                if not all(np.array_equal(episodes[end - 4 + k, e], block['episode_id'][t, e]) for k in range(5)):
                    raise ValueError('five-frame window crosses an episode boundary')
                directory = output / f'shard_{len(info["shards"]):06d}'
                directory.mkdir()
                np.save(directory / 'history.npy', window, allow_pickle=False)
                identity = np.stack((e, block['episode_id'][t, e], block['global_step'][t]), -1)
                np.save(directory / 'sample_ids.npy', identity, allow_pickle=False)
                target = np.lib.format.open_memmap(directory / 'targets.npy', mode='w+', dtype=np.float32,
                                                   shape=(len(t), 360, 2))
                for offset in range(0, len(t), label_batch_size):
                    check_stop()
                    stop = offset + label_batch_size
                    target[offset:stop] = make_labels(block, raw, t[offset:stop], e[offset:stop],
                                                      horizon=horizon, maximum=maximum, device=device)
                target.flush()
                del target
                info['shards'].append(dict(path=directory.name, split=split, samples=len(t),
                    sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob('*.npy')}))
                counts[split] += len(t)
                written_bytes += sum(p.stat().st_size for p in directory.glob('*.npy'))
        tail, tail_episode = history[-4:].copy(), episodes[-4:].copy()
        info['raw_files_verified'].append(record)
        progress = dict(source_file=path.name, completed_files=len(info['raw_files_verified']),
                        requested_files=raw['requested_files'], samples=counts.copy())
        (output / 'progress.json').write_text(json.dumps(progress, indent=2))
        if on_progress is not None:
            on_progress(progress)
        now = time.monotonic()
        print_progress('Label generation', len(info['raw_files_verified']), raw['requested_files'], now - started, {
            'Source file': path.name,
            'Throughput': f'{(sum(counts.values()) - previous_samples) / (now - file_started):.0f} samples/s',
            'Train samples': counts['train'],
            'Val samples': counts['val'],
            'Completed shards': len(info['shards']),
            'File time': f'{now - file_started:.3f}s',
            'Written prepared data': f'{written_bytes / 2**30:.3f} GiB',
        })
        if max_samples_per_split is not None and all(n >= max_samples_per_split for n in counts.values()):
            break
    if not all(counts.values()):
        raise ValueError(f'both splits need valid five-frame samples: {counts}')
    info['samples'] = counts
    check_stop()
    temporary = output / 'manifest.json.tmp'
    temporary.write_text(json.dumps(info, indent=2))
    temporary.replace(output / 'manifest.json')
    print('PREPARATION COMPLETE: ' + json.dumps(counts), flush=True)
    return info
