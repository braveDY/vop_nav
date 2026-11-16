"""Collect frozen BasicNav rollouts with synchronized truth and explicit episode boundaries."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

from ..progress import print_progress
from .data import check_block, validate_dataset

def read_policy_config(checkpoint):
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    config = json.loads((checkpoint.parent / 'config.json').read_text())
    if config['arguments']['task'] != 'Unitree-Go2-VOP-BasicNav':
        raise ValueError('Collection requires a BasicNav PPO checkpoint')
    return config


def _name(value):
    return value.__name__ if callable(value) else value.rsplit(':', 1)[-1]


def validate_policy_config(training, cfg):
    expected = training['env']
    actual = cfg.to_dict()

    def same(label, left, right):
        # JSON normalizes tuples in native config objects to the saved list format.
        if json.dumps(left, sort_keys=True) != json.dumps(right, sort_keys=True):
            raise ValueError(f'Checkpoint input/action mismatch: {label}: {left!r} != {right!r}')

    for sensor_name, class_name, fields in (
        ('ray_caster', 'MultiMeshRayCasterCamera',
         ('ray_alignment', 'max_distance', 'offset', 'pattern_cfg', 'data_types', 'depth_clipping_behavior')),
        ('lidar', 'MultiMeshRayCaster', ('ray_alignment', 'max_distance', 'offset', 'pattern_cfg')),
    ):
        saved, current = expected['scene'][sensor_name], actual['scene'][sensor_name]
        if _name(saved['class_type']) != class_name:
            raise ValueError(f'{sensor_name} requires native {class_name}; legacy sensors are unsupported')
        for field in fields:
            left, right = saved[field], current[field]
            if field == 'pattern_cfg':
                left = {k: v for k, v in left.items() if k != 'func'}
                right = {k: v for k, v in right.items() if k != 'func'}
            same(f'{sensor_name}.{field}', left, right)

    for group in ('policy', 'critic'):
        saved, current = expected['observations'][group], actual['observations'][group]
        saved_terms = [key for key, value in saved.items() if isinstance(value, dict) and 'func' in value]
        current_terms = [key for key, value in current.items() if isinstance(value, dict) and 'func' in value]
        same(f'{group}.observation_order', saved_terms, current_terms)
        for field in ('enable_corruption', 'concatenate_terms'):
            same(f'{group}.{field}', saved[field], current[field])
        for term in saved_terms:
            same(f'{group}.{term}.func', _name(saved[term]['func']), _name(current[term]['func']))
            for field in ('scale', 'clip', 'noise', 'modifiers', 'history_length', 'flatten_history_dim'):
                same(f'{group}.{term}.{field}', saved[term].get(field), current[term].get(field))
            saved_params, current_params = saved[term]['params'], current[term]['params']
            same(f'{group}.{term}.params', list(saved_params), list(current_params))
            for parameter, value in saved_params.items():
                other = current_params[parameter]
                if isinstance(value, dict):
                    # Native managers resolve IDs in saved configs; names define the contract.
                    for field in ('name', 'joint_names', 'body_names', 'preserve_order'):
                        same(f'{group}.{term}.{parameter}.{field}', value[field], other[field])
                else:
                    same(f'{group}.{term}.{parameter}', value, other)

    saved, current = expected['actions']['joint_position'], actual['actions']['joint_position']
    same('action.class_type', _name(saved['class_type']), _name(current['class_type']))
    for field in ('asset_name', 'joint_names', 'preserve_order', 'scale', 'use_default_offset', 'clip'):
        same(f'action.{field}', saved[field], current[field])
    same('simulation.dt', expected['sim']['dt'], actual['sim']['dt'])
    same('simulation.decimation', expected['decimation'], actual['decimation'])


def load_frozen_policy(checkpoint, policy_config, device):
    import torch
    from loco_rl.modules import BasicNavActorCritic

    from loco_rl.modules.basic_nav_actor_critic import OBS_DIM
    model = BasicNavActorCritic(OBS_DIM, OBS_DIM, 12, **policy_config).to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    model.eval()
    model.requires_grad_(False)
    return model, int(saved['iter'])


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--run_dir', type=Path, required=True)
    parser.add_argument('--num_envs', type=int, default=1024)
    parser.add_argument('--files', type=int, default=10)
    parser.add_argument('--steps_per_file', type=int, default=48)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--levels', type=int, nargs='+', default=[0, 3, 6, 9])
    parser.add_argument('--geometry_check', action='store_true')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if min(args.num_envs, args.files, args.steps_per_file) <= 0 or any(x not in range(10) for x in args.levels):
        parser.error('positive dimensions and levels 0..9 required')
    args.checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    training = read_policy_config(args.checkpoint)
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.run_dir / 'collection.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (args.run_dir / 'manifest.json').exists() or list(args.run_dir.glob('raw_*.npz')):
        raise FileExistsError('Refusing to overwrite an existing collection')

    def write_json(path, payload):
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(payload, indent=2, default=str))
        temporary.replace(path)

    started = time.monotonic()

    def progress(status, **extra):
        payload = dict(status=status, pid=os.getpid(), updated_utc=datetime.now(timezone.utc).isoformat(),
                       elapsed_s=time.monotonic() - started)
        payload.update(extra)
        write_json(args.run_dir / 'state.json', payload)

    progress('starting', arguments=vars(args))
    app = None
    stopping = False
    completed_files = 0

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def check_stop():
        if stopping or (args.run_dir / 'stop_requested.json').exists():
            raise InterruptedError('Collection stop requested; incomplete block was not published')

    exit_code = 0
    env = None
    try:
        check_stop()
        app = AppLauncher(args).app
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        check_stop()
        import numpy as np
        import torch
        from isaaclab.managers import SceneEntityCfg
        from envs.go2.nav_env import Go2NavigationEnv
        from envs.go2.nav_env_cfg import Go2EnvCfg
        from assets.mdp.scene import obstacle_state
        from assets.mdp.observations import lidar_ranges
        from loco_rl.env.wrapper import NavigationVecEnvWrapper

        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cfg = Go2EnvCfg()
        cfg.rewards = {}
        cfg.curriculum = None
        cfg.scene.num_envs = args.num_envs
        cfg.seed = args.seed
        cfg.sim.device = args.device
        validate_policy_config(training, cfg)
        write_json(args.run_dir / 'config.json', dict(env=cfg.to_dict(), arguments=vars(args),
                                                   policy=training['train']['policy']))
        env = Go2NavigationEnv(cfg)
        check_stop()
        wrapper = NavigationVecEnvWrapper(env)
        from loco_rl.modules.basic_nav_actor_critic import OBS_DIM, STATE_DIM, LIDAR_DIM, DEPTH_SHAPE
        assert (wrapper.num_obs, wrapper.num_actions) == (OBS_DIM, 12)
        levels = torch.tensor(args.levels, device=env.device).repeat_interleave(
            (args.num_envs + len(args.levels) - 1) // len(args.levels))[:args.num_envs]
        terrain = env.scene.terrain
        terrain.terrain_levels[:] = levels
        terrain.env_origins[:] = terrain.terrain_origins[levels, terrain.terrain_types]
        wrapper.reset()
        check_stop()
        robot = env.scene['robot']
        scene = obstacle_state(env)
        scan = env.scene['lidar']
        assert scan.num_rays == LIDAR_DIM and env.scene['ray_caster'].num_rays == DEPTH_SHAPE[0] * DEPTH_SHAPE[1]
        model, iteration = load_frozen_policy(args.checkpoint, training['train']['policy'], env.device)
        original_parameters = {name: value.detach().clone() for name, value in model.state_dict().items()}
        checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()

        def cpu(tensor):
            return tensor.detach().cpu().numpy().copy()

        with torch.inference_mode():
            geometry = None
            if args.geometry_check:
                # Independent 90-degree heading test: a cylinder 2m along world +Y.
                ids = torch.arange(args.num_envs, device=env.device)
                scene.active[:] = False
                scene.active[:, 0] = True
                scene.positions[:, 0] = torch.tensor([-4., 2.], device=env.device)
                scene.velocities[:] = 0.
                for index in range(len(scene.assets)):
                    scene._write(index, ids)
                pose = robot.data.root_state_w[:, :7].clone()
                pose[:, :3] = env.scene.env_origins + torch.tensor([-4., 0., .4], device=env.device)
                pose[:, 3:] = torch.tensor([2 ** -.5, 0., 0., 2 ** -.5], device=env.device)
                robot.write_root_pose_to_sim(pose)
                robot.write_root_velocity_to_sim(torch.zeros(args.num_envs, 6, device=env.device))
                env.scene.write_data_to_sim()
                env.sim.forward()
                scan.update(env.step_dt, force_recompute=True)
                ranges = (scan.data.ray_hits_w - scan._ray_starts_w).norm(dim=-1)
                # Use actual world rays: native yaw-mode direction handling varies by SDK.
                center = env.scene.env_origins[:, :2] + scene.positions[:, 0]
                delta = center[:, None] - scan._ray_starts_w[..., :2]
                directions = scan._ray_directions_w
                torch.testing.assert_close(directions.norm(dim=-1),
                    torch.ones_like(ranges), atol=1e-6, rtol=0.)
                torch.testing.assert_close(directions[..., 2],
                    torch.zeros_like(ranges), atol=1e-6, rtol=0.)
                projection = (delta * directions[..., :2]).sum(-1)
                ray_index = projection.argmax(-1, keepdim=True)
                along = projection.gather(1, ray_index).squeeze(1)
                discriminant = along.square() - delta.square().sum(-1).gather(1, ray_index).squeeze(1) + scene.radii[0].square()
                assert (discriminant >= 0.).all() and (along > 0.).all()
                expected = along - discriminant.sqrt()
                measured = ranges.gather(1, ray_index).squeeze(1)
                torch.testing.assert_close(measured, expected, atol=.06, rtol=0.)
                geometry = dict(passed=True, yaw_degrees=90, ray_index=cpu(ray_index[:, 0]).tolist(),
                                front_cylinder_range_m=cpu(measured).tolist())
                write_json(args.run_dir / 'geometry_check.json', geometry)
                wrapper.reset()
                check_stop()

            manifest = dict(schema_version=1, requested_files=args.files, num_envs=args.num_envs,
                steps_per_file=args.steps_per_file, control_dt_s=env.step_dt, seed=args.seed,
                checkpoint=str(args.checkpoint), checkpoint_sha256=checkpoint_hash, iteration=iteration,
                policy_mode='deterministic_frozen_actor; original observation corruption retained',
                policy_depth=f'native pinhole camera, {DEPTH_SHAPE[0]}x{DEPTH_SHAPE[1]}, optical-axis Z depth in metres',
                policy_lidar='native single-frame 40-ray yaw-mode policy sensor',
                lidar='same lidar sensor as policy; actual ray-origin Euclidean ranges',
                lidar_direction_reference='per-frame world directions recorded in lidar_directions_w; no assumed body-frame angles',
                lidar_range_m=[.12, 6.], quaternion_order='wxyz',
                obstacle_positions='environment-local XY; add env_origins_w for world coordinates',
                obstacle_velocities='world-axis absolute planar velocities, m/s',
                obstacle_specs=cfg.scene.terrain.obstacle_specs,
                obstacle_orientation_wxyz=[1., 0., 0., 0.],
                joint_names=robot.joint_names,
                action_joint_names=list(cfg.actions.joint_position.joint_names),
                action_scale=.25, robot_collision_proxy_radius_m=.35,
                source_sha256={str(path.relative_to(Path(__file__).resolve().parents[3])):
                    hashlib.sha256(path.read_bytes()).hexdigest()
                    for base in ('source', 'scripts')
                    for path in (Path(__file__).resolve().parents[3] / base).rglob('*.py')},
                wall_geometry=dict(names=['wall_left', 'wall_right'], size_m=[10.4, .2, 2.]),
                time_alignment='state t is copied before action/step; done_after_step describes t->t+1',
                history='current plus previous 4 frames, same env_id and episode_id; preserve across files',
                reward_definition='zero: collection has no training reward terms',
                labels_present=False, geometry_check=geometry, files=[])
            write_json(args.run_dir / 'manifest.json', manifest)
            episode = torch.zeros(args.num_envs, dtype=torch.int64, device=env.device)
            origins = cpu(env.scene.env_origins)
            walls = cpu(torch.stack([env.scene[name].data.root_state_w for name in ('wall_left', 'wall_right')], 1))
            previous = None
            max_pose_error = max_velocity_error = 0.
            progress('collecting', completed_files=0, requested_files=args.files)
            collection_started = time.monotonic()
            written_bytes = 0
            for file_index in range(args.files):
                file_started = time.monotonic()
                frames = {}
                for offset in range(args.steps_per_file):
                    check_stop()
                    global_step = file_index * args.steps_per_file + offset
                    # Reuse the scan that produced obs_buf. The environment already
                    # advances sensors; updating here recasts against a cached observation.
                    hit = scan.data.ray_hits_w
                    distances = (hit - scan._ray_starts_w).norm(dim=-1)
                    observation = wrapper.get_observations()
                    policy_scan = lidar_ranges(env, SceneEntityCfg('lidar'))
                    torch.testing.assert_close(observation[:, STATE_DIM:STATE_DIM + LIDAR_DIM],
                                               policy_scan, atol=1e-5, rtol=0.)
                    actual_position = torch.stack([asset.data.root_pos_w[:, :2] for asset in scene.assets], 1)
                    actual_velocity = torch.stack([asset.data.root_lin_vel_w[:, :2] for asset in scene.assets], 1)
                    pose_error = float((actual_position - scene.positions - env.scene.env_origins[:, None, :2]).abs().max())
                    velocity_error = float((actual_velocity - scene.velocities).abs().max())
                    max_pose_error = max(max_pose_error, pose_error)
                    max_velocity_error = max(max_velocity_error, velocity_error)
                    assert pose_error < .002 and velocity_error < .002, (pose_error, velocity_error)
                    for asset in scene.assets:
                        expected_q = torch.tensor([1., 0., 0., 0.], device=env.device)
                        assert (asset.data.root_quat_w - expected_q).abs().max() < 1e-5
                    actions = model.act_inference(observation)
                    assert torch.isfinite(actions).all()
                    sample = dict(global_step=np.asarray(global_step, dtype=np.int64),
                        episode_id=cpu(episode), episode_step=cpu(env.episode_length_buf).astype(np.int32),
                        history_valid=cpu(env.episode_length_buf >= 4),
                        lidar_ranges=cpu(torch.nan_to_num(distances, nan=6., posinf=6.).clamp(.12, 6.)),
                        lidar_hit_valid=cpu(torch.isfinite(hit).all(-1) & (distances <= 6.)),
                        lidar_origin_w=cpu(scan._ray_starts_w[:, 0]),
                        lidar_directions_w=cpu(scan._ray_directions_w),
                        policy_lidar_ranges=cpu(policy_scan), robot_root_state_w=cpu(robot.data.root_state_w),
                        joint_positions=cpu(robot.data.joint_pos), joint_velocities=cpu(robot.data.joint_vel),
                        goal_position_w=cpu(env.command_manager.get_term('goal').target_pos_w), actions=cpu(actions),
                        obstacle_positions_env=cpu(scene.positions), obstacle_velocities_w=cpu(scene.velocities),
                        obstacle_active=cpu(scene.active))
                    _, _, reward, done, _ = wrapper.step(actions)
                    truncated = env.reset_time_outs
                    sample.update(done_after_step=cpu(done), truncated_after_step=cpu(truncated),
                                  terminated_after_step=cpu(env.reset_terminated), reward=cpu(reward))
                    for name in ('success', 'collision', 'fall', 'outside'):
                        sample['terminal_' + name] = cpu(done & env.episode_results[name].bool())
                    episode += done
                    for name, value in sample.items():
                        frames.setdefault(name, []).append(value)
                sampled = time.monotonic()
                arrays = {name: np.stack(values) for name, values in frames.items()}
                arrays.update(env_origins_w=origins, curriculum_level=cpu(levels), wall_root_state_w=walls)
                previous = check_block(arrays, args.num_envs, args.steps_per_file, previous)
                assert np.array_equal(cpu(env.scene.env_origins), origins)
                path = args.run_dir / f'raw_{file_index:04d}.npz'
                temporary = path.with_suffix('.npz.tmp')
                with temporary.open('xb') as stream:
                    np.savez(stream, **arrays)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(path)
                manifest['files'].append(dict(name=path.name, bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(), first_step=int(arrays['global_step'][0]),
                    last_step=int(arrays['global_step'][-1])))
                write_json(args.run_dir / 'manifest.json', manifest)
                completed_files = file_index + 1
                progress('collecting', completed_files=file_index + 1, requested_files=args.files,
                         records=(file_index + 1) * args.num_envs * args.steps_per_file)
                now = time.monotonic()
                written_bytes += path.stat().st_size
                print_progress('Collection', completed_files, args.files, now - collection_started, {
                    'Environments': args.num_envs,
                    'Throughput': f'{args.num_envs * args.steps_per_file / (now - file_started):.0f} records/s',
                    'Sampling time': f'{sampled - file_started:.3f}s',
                    'Validation / saving time': f'{now - sampled:.3f}s',
                    'Collected records': completed_files * args.num_envs * args.steps_per_file,
                    'Written raw data': f'{written_bytes / 2**30:.3f} GiB',
                    'Output file': path.name,
                })
            for name, value in model.state_dict().items():
                assert torch.equal(value, original_parameters[name]), f'policy changed: {name}'
            assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == checkpoint_hash
            result = validate_dataset(args.run_dir)
            check_stop()
            result.update(policy_unchanged=True, checkpoint_unchanged=True,
                          max_obstacle_pose_error_m=max_pose_error, max_obstacle_velocity_error_mps=max_velocity_error,
                          elapsed_s=time.monotonic() - started)
            write_json(args.run_dir / 'acceptance.json', result)
            progress('raw_completed', **result)
            print('RAW COLLECTION COMPLETE: ' + json.dumps(result), flush=True)
    except InterruptedError as error:
        exit_code = 130
        progress('stopped', completed_files=completed_files, requested_files=args.files, error=str(error))
        print(f'RAW COLLECTION STOPPED: {completed_files}/{args.files} complete files retained', flush=True)
    except BaseException as error:
        exit_code = 1
        traceback.print_exc()
        progress('failed', error=repr(error))
    finally:
        if env is not None:
            env.close()
        if app is not None and exit_code:
            app.app.post_quit(exit_code)
        if app is not None:
            app.close()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
