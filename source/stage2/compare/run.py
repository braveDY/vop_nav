"""Export raw BasicNav recordings into a browser-only, offline replay folder."""
import base64
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from ..geometry import heading_lidar, make_labels
from ..metrics import batch_metrics


def yaw(quaternion):
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def project_block(block, selected):
    origin = block['env_origins_w'][selected]
    robot = block['robot_root_state_w'][:, selected]
    q = robot[..., 3:7]
    w, x, y, z = np.moveaxis(q, -1, 0)
    attitude = np.stack((np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
                         np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))), axis=-1)
    hit = (block['lidar_origin_w'][:, selected, None, :2]
           + block['lidar_ranges'][:, selected, :, None]
           * block['lidar_directions_w'][:, selected, :, :2] - origin[None, :, None, :2])
    result = dict(robot_xy=robot[..., :2] - origin[None, :, :2], robot_yaw=yaw(q),
                  robot_height=robot[..., 2] - origin[None, :, 2], robot_attitude=attitude,
                  robot_velocity=robot[..., 7:9],
                  goal_xy=block['goal_position_w'][:, selected, :2] - origin[None, :, :2],
                  lidar_origin=block['lidar_origin_w'][:, selected, :2] - origin[None, :, :2],
                  lidar_xy=hit, lidar_valid=block['lidar_hit_valid'][:, selected],
                  obstacle_xy=block['obstacle_positions_env'][:, selected],
                  obstacle_velocity=block['obstacle_velocities_w'][:, selected],
                  obstacle_active=block['obstacle_active'][:, selected])
    for name in ('episode_id', 'episode_step', 'history_valid', 'done_after_step',
                 'terminal_success', 'terminal_collision', 'terminal_fall', 'terminal_outside'):
        result[name] = block[name][:, selected]
    return result


def encode_array(value):
    if value.dtype.kind == 'b':
        dtype, name = np.dtype('u1'), 'u8'
    elif value.dtype.kind in 'iu':
        dtype, name = np.dtype('<u4'), 'u32'
    else:
        dtype, name = np.dtype('<f4'), 'f32'
    data = np.ascontiguousarray(value, dtype=dtype)
    if data.dtype.kind == 'f' and not np.isfinite(data).all():
        raise ValueError('Nonfinite replay coordinates')
    return dict(dtype=name, shape=list(data.shape), data=base64.b64encode(data.tobytes()).decode('ascii'))


def export_recording(source, output, env_ids=None, *, checkpoint=None, device='cpu', batch_size=256, label_batch_size=16, horizon=1., maximum=4.5):
    if source is None:
        raise ValueError('Set input in the comparison YAML file')
    source, output = Path(source).resolve(), Path(output).resolve()
    if (source / 'raw').is_dir():
        source = source / 'raw'
    manifest = json.loads((source / 'manifest.json').read_text())
    selected = np.arange(manifest['num_envs']) if env_ids is None else np.asarray(env_ids, dtype=int)
    if not len(selected) or len(np.unique(selected)) != len(selected):
        raise ValueError('Select at least one environment, with no duplicates')
    if (selected < 0).any() or (selected >= manifest['num_envs']).any():
        raise ValueError(f"Environment IDs must be in 0..{manifest['num_envs'] - 1}")
    if not manifest['files']:
        raise ValueError('The manifest contains no recordings')
    from ..geometry import make_labels
    if label_batch_size <= 0 or not (np.isfinite(horizon) and horizon > 0 and np.isfinite(maximum) and maximum > 0):
        raise ValueError('positive finite VO parameters and label batch size required')
    label_config = dict(horizon_s=horizon, maximum_speed_mps=maximum, geometry='rounded_box',
                        robot_radius_m=manifest['robot_collision_proxy_radius_m'], directions=360)
    predictor = None
    if checkpoint:
        predictor = ReplayPredictor(checkpoint, manifest, device, batch_size, label_batch_size)
        label_config = predictor.info['label_config']
    if output.exists() and any(p.name not in ('compare.log', 'resolved.yaml') for p in output.iterdir()):
        raise FileExistsError(f'Output must be empty: {output}')
    if output == source or source in output.parents:
        raise ValueError('Keep the exported viewer separate from the source dataset')
    output.mkdir(parents=True, exist_ok=True)
    blocks, global_steps, boundaries = [], [], []
    reference_origins = reference_levels = reference_walls = None
    previous_episode = previous_step = previous_done = None
    for record in manifest['files']:
        path = source / record['name']
        if path.parent != source or path.suffix != '.npz':
            raise ValueError(f'Invalid source block path: {path}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f'Checksum mismatch: {path}')
        with np.load(path, allow_pickle=False) as block:
            frame_numbers = block['global_step'].copy()
            if np.any(np.diff(frame_numbers) != 1) or (global_steps and frame_numbers[0] != global_steps[-1][-1] + 1):
                raise ValueError('Recording files must have consecutive global steps')
            current = project_block(block, selected)
            if previous_episode is not None:
                if not np.array_equal(current['episode_id'][0], previous_episode + previous_done):
                    raise ValueError('Episode discontinuity between files')
                if not np.array_equal(current['episode_step'][0], np.where(previous_done, 0, previous_step + 1)):
                    raise ValueError('Episode-step discontinuity between files')
            previous_episode = current['episode_id'][-1].copy()
            previous_step = current['episode_step'][-1].copy()
            previous_done = current['done_after_step'][-1].copy()
            if predictor:
                current.update(predictor.predict_block(block, selected))
            else:
                steps = len(frame_numbers)
                reference = np.zeros((steps, len(selected), 360, 2), dtype=np.float32)
                times, columns = np.indices((steps, len(selected))).reshape(2, -1)
                for offset in range(0, len(times), label_batch_size):
                    t, e = times[offset:offset + label_batch_size], columns[offset:offset + label_batch_size]
                    reference[t, e] = make_labels(block, manifest, t, selected[e],
                        horizon=horizon, maximum=maximum, device=device)
                current['vo_reference'] = reference
            origins = block['env_origins_w'][selected].copy()
            levels = block['curriculum_level'][selected].copy()
            walls = block['wall_root_state_w'][selected].copy()
            if reference_origins is None:
                reference_origins, reference_levels, reference_walls = origins, levels, walls
            elif not all(np.array_equal(a, b) for a, b in
                         ((origins, reference_origins), (levels, reference_levels), (walls, reference_walls))):
                raise ValueError('Environment geometry changes between files')
            boundaries.append(dict(name=path.name, start=sum(len(x) for x in global_steps), count=len(frame_numbers)))
            blocks.append(current)
            global_steps.append(frame_numbers)
        print(f'Read {path.name}: {len(frame_numbers)} frames × {len(selected)} environments', flush=True)
    fields = {name: np.concatenate([block[name] for block in blocks], axis=0) for name in blocks[0]}
    del blocks
    frame_numbers = np.concatenate(global_steps)
    data_dir = output / 'data'
    data_dir.mkdir()
    entries = []
    for column, env_id in enumerate(selected):
        walls = reference_walls[column]
        wall_xy = walls[:, :2] - reference_origins[column, :2]
        wall_states = [dict(x=float(xy[0]), y=float(xy[1]), yaw=float(angle),
                            size=manifest['wall_geometry']['size_m'][:2])
                       for xy, angle in zip(wall_xy, yaw(walls[:, 3:7]))]
        payload = dict(frames=len(frame_numbers), level=int(reference_levels[column]), walls=wall_states,
                       fields={name: encode_array(value[:, column]) for name, value in fields.items()})
        filename = f'data/env_{int(env_id):04d}.js'
        text = json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        (output / filename).write_text(f'window.VOP_REPLAY_LOAD({int(env_id)},{text});\n')
        entries.append(dict(id=int(env_id), level=int(reference_levels[column]), file=filename))
    viewer_manifest = dict(version=1, frames=len(frame_numbers), dt=manifest['control_dt_s'],
        duration_s=len(frame_numbers) * manifest['control_dt_s'], global_steps=frame_numbers.tolist(),
        environments=entries, blocks=boundaries, obstacle_specs=manifest['obstacle_specs'],
        vo_config=label_config,
        robot_radius=manifest.get('robot_collision_proxy_radius_m', .35),
        source_checkpoint=Path(manifest['checkpoint']).name,
        source_sha256=manifest['checkpoint_sha256'], labels_present=manifest['labels_present'])
    if predictor:
        viewer_manifest['model'] = predictor.info
    assets = Path(__file__).resolve().parent / 'replay'
    for name in ('index.html', 'viewer.css', 'comparison.js', 'vo_view.js', 'viewer.js'):
        shutil.copyfile(assets / name, output / name)
    (output / 'manifest.js').write_text('window.VOP_REPLAY_MANIFEST=' +
        json.dumps(viewer_manifest, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + ';\n')
    summary = dict(environments=len(entries), frames=len(frame_numbers),
                   duration_s=viewer_manifest['duration_s'], source=str(source), index=str(output / 'index.html'))
    if predictor:
        summary['model'] = predictor.info
    (output / 'export.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


class ReplayPredictor:
    def __init__(self, checkpoint, manifest, device='cpu', batch_size=256, label_batch_size=16):
        from ..model import load_model
        from ..config import validate_contract
        if min(batch_size, label_batch_size) <= 0:
            raise ValueError('inference and label batch sizes must be positive')
        checkpoint = Path(checkpoint).resolve()
        self.model, saved = load_model(checkpoint, device)
        self.model.eval()
        inputs, labels = saved['input_config'], saved['label_config']
        validate_contract(saved, control_dt=manifest['control_dt_s'], ranges=manifest['lidar_range_m'],
                          radius=manifest['robot_collision_proxy_radius_m'],
                          maximum=self.model.config.maximum, range_atol=1e-6)
        if manifest.get('obstacle_orientation_wxyz') != [1., 0., 0., 0.]:
            raise ValueError('full-boundary inference requires axis-aligned recorded obstacles')
        self.device, self.batch_size, self.label_batch_size = device, batch_size, label_batch_size
        self.alpha = saved['train_config']['alpha']
        self.manifest = manifest
        self.tail = self.tail_episode = None
        self.samples, self.totals = 0, {}
        self.info = dict(checkpoint=checkpoint.name, checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            epoch=saved['epoch'], input_config=inputs, label_config=labels,
            training_dataset_fingerprint=saved['dataset_fingerprint'],
            training_mode='offline', device=device,
            comparison='selected recording; validation split membership not established')

    @torch.inference_mode()
    def predict_block(self, block, selected):
        """Return aligned [time,env,...] predictions; never pad an invalid history."""
        names = ('robot_root_state_w', 'lidar_ranges', 'lidar_directions_w',
                 'obstacle_positions_env', 'obstacle_velocities_w', 'obstacle_active',
                 'episode_id', 'episode_step', 'history_valid', 'done_after_step')
        local = {key: block[key][:, selected] for key in names}
        for key in ('env_origins_w', 'wall_root_state_w'):
            local[key] = block[key][selected]
        scans = heading_lidar(local)
        steps, environments = scans.shape[:2]
        if not np.array_equal(local['history_valid'], local['episode_step'] >= 4):
            raise ValueError('invalid history flags')
        if steps > 1:
            done = local['done_after_step'][:-1]
            if (not np.array_equal(local['episode_id'][1:], local['episode_id'][:-1] + done)
                    or not np.array_equal(local['episode_step'][1:], np.where(done, 0, local['episode_step'][:-1] + 1))):
                raise ValueError('episode discontinuity within raw block')
        prefix = 0 if self.tail is None else len(self.tail)
        history = scans if self.tail is None else np.concatenate((self.tail, scans))
        episodes = local['episode_id'] if self.tail_episode is None else np.concatenate((self.tail_episode, local['episode_id']))
        times, envs = np.nonzero(local['history_valid'])
        ends = prefix + times
        if (ends < 4).any() or any(not np.array_equal(episodes[ends - 4 + k, envs], local['episode_id'][times, envs]) for k in range(5)):
            raise ValueError('five-frame inference history is missing or crosses an episode')
        predictions = np.zeros((steps, environments, 360, 2), dtype=np.float32)
        reference = np.zeros_like(predictions)
        labels = self.info['label_config']
        all_times, all_envs = np.indices((steps, environments)).reshape(2, -1)
        for offset in range(0, len(all_times), self.label_batch_size):
            t, e = all_times[offset:offset + self.label_batch_size], all_envs[offset:offset + self.label_batch_size]
            reference[t, e] = make_labels(local, self.manifest, t, e, horizon=labels['horizon_s'],
                                         maximum=labels['maximum_speed_mps'], device=self.device)
        for offset in range(0, len(times), self.batch_size):
            t, e, end = times[offset:offset + self.batch_size], envs[offset:offset + self.batch_size], ends[offset:offset + self.batch_size]
            window = np.stack([history[end - 4 + k, e] for k in range(5)], 1)
            prediction = self.model(torch.from_numpy(window).to(self.device))
            if not torch.isfinite(prediction).all():
                raise ValueError('nonfinite model prediction')
            predictions[t, e] = prediction.cpu().numpy()
            metrics = batch_metrics(prediction, torch.from_numpy(reference[t, e]).to(self.device), self.alpha)
            self.samples += len(t)
            for key, value in metrics.items():
                self.totals[key] = self.totals.get(key, 0.) + value * len(t)
        self.tail, self.tail_episode = history[-4:].copy(), episodes[-4:].copy()
        self.info['metrics'] = {k: v / self.samples for k, v in self.totals.items()} | {'samples': self.samples}
        return dict(model_intervals=predictions, vo_reference=reference, model_valid=local['history_valid'])


def main():
    from ..config import parse_config, save_config
    args = parse_config('compare')
    import torch
    if args.num_threads <= 0:
        raise ValueError('num_threads must be positive')
    torch.set_num_threads(args.num_threads)
    export_recording(args.input, args.run_dir, args.env_ids, checkpoint=args.checkpoint, device=args.device,
                     batch_size=args.batch_size, label_batch_size=args.label_batch_size, horizon=args.horizon, maximum=args.maximum)
    save_config(args, args.run_dir)


if __name__ == '__main__':
    main()
