from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import random
import signal
import time

ROOT = Path(__file__).resolve().parents[2]


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
    apply_values(cfg, args.environment)
    if hasattr(cfg, 'vop_checkpoint'):
        if not args.vop_checkpoint:
            raise ValueError('VOP-Nav requires vop_checkpoint in YAML')
        path = args.vop_checkpoint.expanduser()
        cfg.vop_checkpoint = str((ROOT / path).resolve() if not path.is_absolute() else path.resolve())
    elif args.vop_checkpoint:
        raise ValueError('vop_checkpoint requires a VOP-Nav task')
    cfg.sim.device = args.device
    if args.num_envs is not None:
        if args.num_envs <= 0:
            raise ValueError('num_envs must be positive')
        cfg.scene.num_envs = args.num_envs
    if args.low_memory:
        if args.num_envs not in (None, 1):
            raise ValueError('low_memory needs num_envs: 1')
        configure_low_memory(cfg)
    if args.experiment_name is None:
        args.experiment_name = 'vop_nav' if hasattr(cfg, 'vop_checkpoint') else 'vop_basic'
    _set_seed(args.seed)
    cfg.seed = args.seed
    native = gym.make(args.task, cfg=cfg).unwrapped
    native.stream_render = bool(getattr(args, 'livestream', 0) > 0)
    module, name = spec.kwargs['train_cfg_entry_point'].split(':')
    train_cfg = getattr(importlib.import_module(module), name)()
    apply_values(train_cfg, args.agent)
    train_cfg.seed = args.seed
    if args.run_dir:
        run = args.run_dir.expanduser()
        if not run.is_absolute():
            run = ROOT / run
    else:
        run = ROOT / 'logs' / 'navigation' / args.experiment_name / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + (args.name or 'train'))
    run.mkdir(parents=True, exist_ok=True)
    from stage2.config import save_config
    save_config(args, run)
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


def apply_values(target, values):
    for key, value in values.items():
        if isinstance(target, dict):
            if key not in target:
                raise ValueError(f'Unknown configuration key: {key}')
            current = target[key]
        else:
            if not hasattr(target, key):
                raise ValueError(f'Unknown configuration field: {key}')
            current = getattr(target, key)
        if isinstance(value, dict) and current is not None:
            apply_values(current, value)
        elif isinstance(target, dict):
            target[key] = value
        else:
            setattr(target, key, tuple(value) if isinstance(current, tuple) and isinstance(value, list) else value)


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
