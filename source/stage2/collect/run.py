"""Run frozen-policy collection, then prepare the accepted data for VOP training."""
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from ..config import parse_config, save_config


def validate_arguments(args):
    if args.num_envs < 2 or min(args.files, args.steps_per_file) <= 0 or args.files * args.steps_per_file < 5:
        raise ValueError('Need at least 2 environments and 5 steps per environment. files and steps_per_file must each be > 0')
    if args.seed < 0 or min(args.shard_size, args.label_batch_size) <= 0:
        raise ValueError('seed must be >= 0. shard_size and label_batch_size must be integers > 0')
    if not (math.isfinite(args.horizon) and args.horizon > 0 and math.isfinite(args.maximum) and args.maximum > 0):
        raise ValueError('horizon and maximum must be > 0, not NaN or infinity')
    if not 0 < args.val_fraction < 1:
        raise ValueError('val_fraction must be > 0 and < 1')


def validate_directories(run, prepared):
    if run == prepared or run in prepared.parents or prepared in run.parents:
        raise ValueError('Raw and training data folders must be separate. Neither can be inside the other')
    if prepared.exists() and (not prepared.is_dir() or any(prepared.iterdir())):
        raise FileExistsError(f'Training data folder must be empty or new: {prepared}')


def dataset_directories(run, prepared=None):
    run = Path(run).resolve()
    raw = (run / 'raw').resolve()
    prepared = Path(prepared).resolve() if prepared is not None else (run / 'prepared').resolve()
    if raw.parent != run:
        raise ValueError('raw must be directly inside the run folder. It cannot link to another folder')
    if prepared == run or prepared in run.parents:
        raise ValueError('Training data folder cannot be the run folder or contain it')
    if run in prepared.parents and prepared.relative_to(run).parts[0] in (
            'config.json', 'state.json', 'collect.log', 'pipeline.lock', 'stop_requested.json'):
        raise ValueError('Training data folder cannot use a path reserved for run files or logs')
    validate_directories(raw, prepared)
    return raw, prepared


def run_pipeline(args, raw_command):
    """Wait for the simulator to exit before importing torch and generating labels."""
    run = Path(args.run_dir).resolve()
    validate_arguments(args)
    raw, prepared = dataset_directories(run, args.prepared_dir)
    run.mkdir(parents=True, exist_ok=True)
    with (run / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(path.name not in ('pipeline.lock', 'collect.log', 'stop_requested.json', 'resolved.yaml') for path in run.iterdir()):
            raise FileExistsError(f'Refusing to reuse an existing collection: {run}')
        save_config(args, run)
        child = None
        stopping = False
        phase = 'collecting'
        started = time.monotonic()
        payload = {}

        def write_json(path, value):
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(value, indent=2, default=str))
            temporary.replace(path)

        def state(status, **extra):
            payload.update(status=status, phase=phase, pid=os.getpid(), raw_dir=str(raw), prepared_dir=str(prepared),
                           elapsed_s=time.monotonic() - started,
                           updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
            write_json(run / 'state.json', payload)

        def forward_stop():
            if raw.is_dir():
                write_json(raw / 'stop_requested.json', {'pid': os.getpid()})
            if child is not None and child.poll() is None:
                try:
                    child.send_signal(signal.SIGTERM)
                except ProcessLookupError:
                    pass

        def request_stop(signum, frame):
            nonlocal stopping
            stopping = True
            forward_stop()

        def check_stop():
            if stopping or (run / 'stop_requested.json').exists():
                raise InterruptedError('Collection/preparation stop requested')

        previous_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            write_json(run / 'config.json', dict(schema_version=1, arguments=vars(args),
                       raw_dir=str(raw), prepared_dir=str(prepared)))
            state('starting', arguments=vars(args))
            check_stop()
            raw.mkdir()
            child = subprocess.Popen(raw_command, stdin=subprocess.DEVNULL)
            if stopping:
                forward_stop()
            while True:
                try:
                    code = child.wait(timeout=1.)
                except subprocess.TimeoutExpired:
                    if not stopping and (run / 'stop_requested.json').exists():
                        request_stop(signal.SIGTERM, None)
                    child_state = raw / 'state.json'
                    if child_state.exists():
                        state('collecting', collection=json.loads(child_state.read_text()))
                    continue
                break
            child_state = raw / 'state.json'
            if child_state.exists():
                state('collecting', collection=json.loads(child_state.read_text()))
            check_stop()
            if code:
                raise RuntimeError(f'Raw collection exited with code {code}; preparation was not started')
            acceptance = json.loads((raw / 'acceptance.json').read_text())
            if not acceptance.get('passed'):
                raise ValueError('Raw collection acceptance failed; preparation was not started')
            phase = 'preparing'
            state('preparing', raw_acceptance=acceptance)
            print(f'PREPARATION START: {raw} -> {prepared}', flush=True)
            import torch
            from .data import prepare_dataset

            torch.set_num_threads(4)
            info = prepare_dataset(raw, prepared, horizon=args.horizon, maximum=args.maximum,
                val_fraction=args.val_fraction, seed=args.seed, shard_size=args.shard_size,
                label_batch_size=args.label_batch_size, device=args.device, check_stop=check_stop,
                on_progress=lambda progress: state('preparing', preparation=progress))
            state('completed', samples=info['samples'])
            print(f'COLLECT COMPLETE: training dataset {prepared}', flush=True)
            return 0
        except InterruptedError as error:
            state('stopped', error=str(error))
            print(f'COLLECT STOPPED during {phase}; complete raw blocks retained', flush=True)
            return 130
        except Exception as error:
            state('failed', error=repr(error))
            traceback.print_exc()
            return 1
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)


def validate_checkpoint(args):
    from .simulator import read_policy_config
    if args.checkpoint is None:
        raise ValueError('Set checkpoint in the collection YAML file')
    read_policy_config(args.checkpoint)


def main():
    args = parse_config('collect')
    validate_checkpoint(args)
    simulator_options = []
    args.run_dir = args.run_dir.resolve()
    try:
        raw, args.prepared_dir = dataset_directories(args.run_dir, args.prepared_dir)
    except (OSError, ValueError) as error:
        raise SystemExit(f'Collect startup failed: {error}')
    command = [sys.executable, '-u', '-m', 'stage2.collect.simulator',
               '--checkpoint', args.checkpoint, '--run_dir', str(raw),
               '--num_envs', str(args.num_envs), '--files', str(args.files),
               '--steps_per_file', str(args.steps_per_file), '--seed', str(args.seed),
               '--device', args.device, '--kit_args', args.kit_args, '--livestream', str(args.livestream), '--levels', *map(str, args.levels), *simulator_options]
    for flag in ('headless', 'geometry_check'):
        if getattr(args, flag):
            command.append('--' + flag)
    try:
        code = run_pipeline(args, command)
    except (OSError, ValueError) as error:
        raise SystemExit(f'Collect startup failed: {error}')
    sys.exit(code)


if __name__ == "__main__":
    main()
