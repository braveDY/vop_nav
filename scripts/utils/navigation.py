from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import random
import signal
import time

ROOT = Path(__file__).resolve().parents[2]



def parse_arguments(kind, argv=None):
    import argparse
    import os
    parser = argparse.ArgumentParser(description=f'Run navigation {kind} with native Python configuration')
    parser.add_argument('--task', default='Unitree-Go2-VOP-BasicNav' + ('-Play' if kind == 'play' else ''))
    parser.add_argument('--checkpoint', type=Path, required=kind != 'train')
    parser.add_argument('--vop_checkpoint', type=Path)
    parser.add_argument('--num_envs', type=int, default=4 if kind == 'evaluate' else None)
    parser.add_argument('--device')
    if kind != 'evaluate':
        parser.add_argument('--seed', type=int)
    parser.add_argument('--run_dir', type=Path)
    parser.add_argument('--name', default='')
    parser.add_argument('--kit_args', default='')
    if kind == 'train':
        parser.add_argument('--max_iterations', type=int)
        parser.add_argument('--deadline')
        parser.add_argument('--chunk_iterations', type=int, default=25)
        parser.add_argument('--low_memory', action='store_true')
    else:
        parser.add_argument('--memory_reserve_gib', type=float, default=4.)
        if kind == 'play':
            parser.add_argument('--scenario', choices=('Training','Forest','Office','Square','Slow'), default='Training')
            parser.add_argument('--level', type=int, choices=range(10))
            parser.add_argument('--steps', type=int, default=0)
        else:
            parser.add_argument('--scenarios', nargs='+', choices=('Training','Forest','Office','Square','Slow'), default=['Forest','Office','Square','Slow'])
            parser.add_argument('--levels', nargs='+', type=int, choices=range(10), default=[0,3,6,9])
            parser.add_argument('--seeds', nargs='+', type=int, default=[123,456])
            parser.add_argument('--episodes_per_env', type=int, default=2)
    args = parser.parse_args(argv)
    tasks = ('Unitree-Go2-VOP-BasicNav','Unitree-Go2-VOP-Nav')
    allowed = tuple(task + '-Play' for task in tasks) if kind == 'play' else tasks
    if args.task not in allowed:
        parser.error('task must be one of: ' + ', '.join(allowed))
    if ('VOP-Nav' in args.task) != bool(args.vop_checkpoint):
        parser.error('VOP-Nav requires --vop_checkpoint; BasicNav must leave it unset')
    for field in ('num_envs','max_iterations','chunk_iterations','episodes_per_env'):
        value = getattr(args, field, None)
        if value is not None and value <= 0:
            parser.error(f'{field} must be positive')
    if getattr(args, 'seed', None) is not None and not 0 <= args.seed < 2**32:
        parser.error('seed must be in [0, 2**32)')
    if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.name):
        parser.error('name can only contain English letters, numbers, _ and -')
    for field in ('checkpoint','vop_checkpoint','run_dir'):
        value = getattr(args, field)
        if value is None:
            continue
        value = value.expanduser()
        host = os.environ.get('HOST_PROJECT_ROOT')
        if host and value.is_absolute() and value.is_relative_to(host):
            value = ROOT / value.relative_to(host)
        setattr(args, field, (ROOT / value).resolve())
    if args.run_dir is None:
        directory = 'navigation' if kind == 'train' else kind
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        args.run_dir = ROOT / 'logs' / directory / (stamp + ('_' + args.name if args.name else ''))
    args.headless, args.livestream = True, 0
    if kind == 'play':
        if args.scenario != 'Training' and args.level is not None:
            parser.error('--level only applies to --scenario Training')
        args.level = 9 if args.level is None else args.level
        if args.steps < 0:
            parser.error('steps must be nonnegative')
        args.real_time = True
        local = os.environ.get('RUN_MODE','local') == 'local'
        args.headless, args.livestream = not local, 0 if local else 2
        if not local and os.environ.get('REMOTE_STREAM_IP'):
            args.kit_args += (' --/app/livestream/publicEndpointAddress=' + os.environ['REMOTE_STREAM_IP']
                              + ' --/app/livestream/port=' + os.environ.get('LIVESTREAM_PORT','49100'))
    if kind == 'train' and args.deadline:
        try:
            deadline = datetime.fromisoformat(args.deadline.replace('Z','+00:00'))
            if deadline.tzinfo is None:
                raise ValueError('timezone required')
        except ValueError:
            parser.error('deadline must be an ISO time with a timezone')
    return args


def save_arguments(args, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'arguments.json').write_text(json.dumps(vars(args), indent=2, default=str))


def launcher_arguments(args):
    return {key: value for key, value in vars(args).items() if value is not None}

def _set_seed(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_low_memory(cfg):
    cfg.scene.num_envs = 1
    cfg.sim.physx.gpu_max_rigid_contact_count = 2**18
    cfg.sim.physx.gpu_max_rigid_patch_count = 2**15
    cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**16
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**16
    cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**16
    cfg.sim.physx.gpu_heap_capacity = 2**25
    cfg.sim.physx.gpu_temp_buffer_capacity = 2**23


def create_env(args):
    import gymnasium as gym
    import envs
    from loco_rl.env.wrapper import NavigationVecEnvWrapper
    spec = gym.spec(args.task)
    module, name = spec.kwargs['env_cfg_entry_point'].split(':')
    cfg = getattr(importlib.import_module(module), name)()
    if hasattr(cfg, 'vop_checkpoint'):
        if not args.vop_checkpoint:
            raise ValueError('VOP-Nav requires --vop_checkpoint')
        path = args.vop_checkpoint.expanduser()
        cfg.vop_checkpoint = str((ROOT / path).resolve() if not path.is_absolute() else path.resolve())
    elif args.vop_checkpoint:
        raise ValueError('vop_checkpoint requires a VOP-Nav task')
    if args.device is not None:
        cfg.sim.device = args.device
    if args.num_envs is not None:
        if args.num_envs <= 0:
            raise ValueError('num_envs must be positive')
        cfg.scene.num_envs = args.num_envs
    if args.low_memory:
        if args.num_envs not in (None, 1):
            raise ValueError('low_memory needs num_envs: 1')
        configure_low_memory(cfg)
    module, name = spec.kwargs['train_cfg_entry_point'].split(':')
    train_cfg = getattr(importlib.import_module(module), name)()
    if args.seed is None:
        args.seed = train_cfg.seed
    _set_seed(args.seed)
    cfg.seed = args.seed
    args.device = cfg.sim.device
    native = gym.make(args.task, cfg=cfg).unwrapped
    args.num_envs = cfg.scene.num_envs
    native.stream_render = bool(getattr(args, 'livestream', 0) > 0)
    train_cfg.seed = args.seed
    if args.max_iterations is not None:
        train_cfg.runner.max_iterations = args.max_iterations
    run = args.run_dir
    save_arguments(args, run)
    train_config=train_cfg.to_dict()
    (run / 'config.json').write_text(json.dumps({'env': native.cfg.to_dict(),
        'train': train_config, 'arguments': vars(args),
        'vop_metadata': native.get_vop().metadata if hasattr(native, 'get_vop') else None}, indent=2, default=str))
    (run / 'physics_metadata.json').write_text(json.dumps({
        'environment_class': type(native).__name__, 'num_envs': native.num_envs,
        'body_names': native.scene['robot'].body_names,
        'perception': type(native.scene.sensors['ray_caster']).__name__ if 'ray_caster' in native.scene.sensors else None,
        'managers': {n: type(getattr(native, n + '_manager')).__name__ for n in
                     ('action', 'observation', 'reward', 'termination', 'command', 'event', 'curriculum')}
    }, indent=2))
    print(f'[Navigation] Log directory: {run}', flush=True)
    return native, NavigationVecEnvWrapper(native), train_config, run


def load_runner(env, config, run, checkpoint=None):
    from loco_rl.runners import OnPolicyRunner
    import zipfile
    if checkpoint:
        with zipfile.ZipFile(checkpoint.expanduser().resolve()) as z:
            if any('/code/' in n for n in z.namelist()):
                raise ValueError('Training requires a PPO checkpoint, not a TorchScript policy export.')
    runner = OnPolicyRunner(env, config, log_dir=str(run), device=env.device)
    if checkpoint:
        runner.load(str(checkpoint.expanduser().resolve()))
    return runner


def learn(runner, env, wrapper, config, run, args, app):
    if not args.deadline:
        iterations = args.max_iterations if args.max_iterations is not None else config['runner']['max_iterations']
        runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=True)
        return

    import torch

    torch.set_num_threads(4)
    deadline = datetime.fromisoformat(args.deadline.replace('Z', '+00:00'))
    requested_stop = False
    started = time.monotonic()

    def stop(signum, frame):
        nonlocal requested_stop
        requested_stop = True

    def state(status, **extra):
        payload = dict(status=status, updated_utc=datetime.now(timezone.utc).isoformat(),
                       elapsed_s=time.monotonic() - started, deadline=args.deadline,
                       graceful_signal_handler_active=signal.getsignal(signal.SIGTERM) is stop,
                       iteration=runner.current_learning_iteration,
                       checkpoint=str(run / f'model_{runner.current_learning_iteration}.pt'),
                       curriculum_mean=env.scene.terrain.terrain_levels.float().mean().item(),
                       curriculum_max=int(env.scene.terrain.terrain_levels.max()), **extra)
        temporary = run / 'train_state.tmp'
        temporary.write_text(json.dumps(payload, indent=2))
        temporary.replace(run / 'train_state.json')

    # Register after AppLauncher, which installs its own signal handlers.
    previous_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        if args.checkpoint:
            curriculum_file = args.checkpoint.with_suffix('.curriculum.pt')
            levels = torch.load(curriculum_file, map_location=env.device, weights_only=True)
            if levels.shape != env.scene.terrain.terrain_levels.shape:
                raise ValueError('Curriculum snapshot environment count does not match')
            env.scene.terrain.terrain_levels[:] = levels
            env.scene.terrain.env_origins[:] = env.scene.terrain.terrain_origins[levels, env.scene.terrain.terrain_types]
            wrapper.reset()
        limit = args.max_iterations if args.max_iterations is not None else config['runner']['max_iterations']
        first = not bool(args.checkpoint)
        state('training')
        while (not requested_stop and app.is_running() and datetime.now(timezone.utc) < deadline
               and runner.current_learning_iteration < limit):
            count = min(args.chunk_iterations, limit - runner.current_learning_iteration)
            runner.learn(count, init_at_random_ep_len=first)
            first = False
            if not all(torch.isfinite(value).all() for value in runner.alg.actor_critic.parameters()):
                raise FloatingPointError('Nonfinite BasicNav model parameter')
            levels_path = (run / f'model_{runner.current_learning_iteration}.pt').with_suffix('.curriculum.pt')
            temporary = levels_path.with_suffix('.tmp')
            torch.save(env.scene.terrain.terrain_levels.cpu(), temporary)
            temporary.replace(levels_path)
            state('training')
        state('ready_acceptance', stopped_by_signal=requested_stop)
    except BaseException as error:
        state('failed', error=repr(error))
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
