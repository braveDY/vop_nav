import argparse
import ctypes
import fcntl
import json
import os
from pathlib import Path
import re
import select
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
LIBC = ctypes.CDLL(None, use_errno=True)
KINDS = ('train', 'collect', 'train_vop', 'compare', 'play', 'evaluate')


def pidfd_open(pid):
    handle = LIBC.pidfd_open(pid, 0)
    if handle < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return handle


def pidfd_signal(handle, sig):
    if LIBC.pidfd_send_signal(handle, sig, None, 0) < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def task_processes(kind, project=PROJECT):
    project = Path(project).resolve()
    found, parents = [], {}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if not (proc / 'comm').read_text().strip().startswith('python'):
                continue
            if (proc / 'cwd').resolve() != project:
                continue
            argv = [os.fsdecode(part) for part in (proc / 'cmdline').read_bytes().split(b'\0') if part]
            if not argv or not Path(argv[0]).name.startswith('python'):
                continue
            matched = False
            if kind in ('train', 'play', 'evaluate'):
                expected = project / 'scripts' / (kind + '.py')
                matched = any(not part.startswith('-') and (project / part).resolve() == expected for part in argv[1:])
            elif '-m' in argv:
                i = argv.index('-m')
                command = {'train_vop': 'train', 'collect': 'collect', 'compare': 'compare'}.get(kind)
                matched = bool(command and argv[i+1:i+3] == ['stage2', command])
                if kind == 'collect' and argv[i+1:i+2] == ['stage2.collect.simulator']:
                    matched = True
            if kind == 'webrtc_udp_bridge':
                expected = project / 'scripts/utils/streaming.py'
                matched = 'bridge' in argv and any((project / arg).resolve() == expected for arg in argv[1:] if not arg.startswith('-'))
            if not matched:
                continue
            pid = int(proc.name)
            found.append(pid)
            parents[pid] = int(next(line.split()[1] for line in (proc / 'status').read_text().splitlines() if line.startswith('PPid:')))
        except (OSError, RuntimeError, ValueError, StopIteration):
            continue
    # Select supervisors rather than their collector or forked DataLoader workers.
    return sorted(pid for pid in found if parents.get(pid) not in found)


def stop_task_processes(kind, project=PROJECT, timeout=10.0, kill_timeout=5.0):
    handles = {}
    graceful = kind in ("collect", "train_vop", "evaluate")
    try:
        for pid in task_processes(kind, project):
            try:
                handle = pidfd_open(pid)
            except ProcessLookupError:
                continue
            if pid not in task_processes(kind, project):
                os.close(handle)
                continue
            handles[pid] = handle
            if kind in ('collect', 'train_vop', 'evaluate', 'play'):
                proc = Path('/proc') / str(pid)
                argv = [os.fsdecode(part) for part in (proc / 'cmdline').read_bytes().split(b'\0') if part]
                if kind == 'play':
                    graceful = True
                run_dir = next((value.split('=', 1)[1] for value in argv if value.startswith('--run-dir=')), None)
                if run_dir is None:
                    flag = next((v for v in ('--run-dir', '--run_dir') if v in argv), None)
                    if flag:
                        run_dir = argv[argv.index(flag) + 1]
                if run_dir:
                    directory = (Path(project) / run_dir).resolve()
                    if directory.is_relative_to(Path(project).resolve()) and directory.is_dir():
                        # Persist the request while native scene initialization delays signal handling.
                        (directory / 'stop_requested.json').write_text(json.dumps({'pid': pid}))
        for handle in handles.values():
            try:
                pidfd_signal(handle, signal.SIGTERM)
            except ProcessLookupError:
                pass

        def wait_for_exit(seconds):
            remaining = dict(handles)
            deadline = time.monotonic() + seconds
            while remaining:
                for pid, handle in list(remaining.items()):
                    poller = select.poll()
                    poller.register(handle, select.POLLIN)
                    if poller.poll(0):
                        del remaining[pid]
                if not remaining or time.monotonic() >= deadline:
                    break
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))
            return remaining

        remaining = wait_for_exit(timeout)
        if remaining:
            if graceful:
                print(f"Stop request sent. {kind} is still starting, saving data or cleaning up. Not killed. PID: {', '.join(map(str, remaining))}", flush=True)
                return False
            print(f"Stop timed out. Killing {kind}. PID: {', '.join(map(str, remaining))}", file=sys.stderr, flush=True)
            for handle in remaining.values():
                try:
                    pidfd_signal(handle, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            remaining = wait_for_exit(kill_timeout)
            if remaining:
                raise RuntimeError(f"Process did not stop. PID: {', '.join(map(str, remaining))}")
    finally:
        for handle in handles.values():
            os.close(handle)
    return True


def latest_progress(path):
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 131072))
            text = stream.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    records = re.split(r"Learning (?:iteration|epoch)\s+", text)[1:]
    for record in reversed(records):
        iteration = re.match(r"(\d+/\d+)", record)
        elapsed = re.search(r"Time elapsed:\s*([\d:]+)", record)
        eta = re.search(r"ETA:\s*([\d:]+)", record)
        if iteration and elapsed and eta:
            return iteration.group(1), elapsed.group(1), eta.group(1)
    return None


def tensorboard_status():
    try:
        pid = int((PROJECT / ".ops/tensorboard.pid").read_text().strip())
        proc = Path("/proc") / str(pid)
        argv = (proc / "cmdline").read_bytes().split(b"\0")
        if (proc / "cwd").resolve(strict=True) == PROJECT and b"tensorboard.main" in argv:
            return pid
    except PermissionError:
        return "unknown"
    except (OSError, ValueError, RuntimeError):
        pass
    return None


def record_path(kind, project=PROJECT):
    return Path(project) / '.ops' / (kind + '.json')


def latest_log(kind, project=PROJECT):
    project = Path(project).resolve()
    candidates = list((project / 'logs').rglob(kind + '.log'))
    marker = record_path(kind, project)
    if marker.is_file():
        try:
            candidates.append(Path(json.loads(marker.read_text())['log_file']))
        except (ValueError, KeyError):
            pass
    candidates = [p for p in candidates if p.is_file() and p.resolve().is_relative_to(project)]
    return max(candidates, key=lambda p: p.stat().st_mtime_ns, default=None)


def launch(kind, arguments):
    from stage2.config import parse_config, plain_config, read_yaml, CONFIGS
    import yaml
    args = parse_config(kind, arguments)
    required = dict(collect='checkpoint', train_vop='dataset', compare='input', play='checkpoint', evaluate='checkpoint')
    for key in (required[kind],) if kind in required else ():
        if getattr(args, key) is None or not getattr(args, key).exists():
            raise ValueError(f'Set {key} to an existing path in the YAML file')
    for key in ('checkpoint', 'vop_checkpoint'):
        path = getattr(args, key, None)
        if path is not None and not path.is_file():
            raise ValueError(f'{key} file does not exist: {path}')
    if kind in ('train', 'play', 'evaluate'):
        vop = 'VOP-Nav' in args.task
        if vop != bool(args.vop_checkpoint):
            raise ValueError('VOP-Nav needs vop_checkpoint; BasicNav must leave it empty')
    run = args.run_dir
    if not run.is_relative_to(PROJECT) or run == PROJECT:
        raise ValueError('Run folder must be inside this project')
    ops = PROJECT / '.ops'
    ops.mkdir(exist_ok=True)
    with (ops / (kind + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = task_processes(kind)
        if active:
            raise RuntimeError(f'{kind} is already running (PID: {active})')
        if run.exists():
            raise FileExistsError(f'Run folder must be new: {run}')
        run.mkdir(parents=True)
        values = plain_config(vars(args))
        allowed = read_yaml(PROJECT / 'configs' / (CONFIGS[kind] + '.yaml'))
        snapshot = {key:values[key] for key in allowed}
        effective = run / 'resolved.yaml'
        effective.write_text(yaml.safe_dump(snapshot, sort_keys=False))
        command = [sys.executable, '-u']
        if kind in ('collect', 'train_vop', 'compare'):
            command += ['-m', 'stage2', 'train' if kind == 'train_vop' else kind]
        else:
            command += [str(PROJECT / 'scripts' / (kind + '.py'))]
        command += ['--config', str(effective), '--run-dir', str(run)]
        log = run / (kind + '.log')
        with log.open('xb') as output:
            worker = subprocess.Popen(command, cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=output,
                stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        marker = record_path(kind)
        marker.write_text(json.dumps(dict(pid=worker.pid, run_dir=str(run), log_file=str(log),
            started=time.time(), config=values), indent=2))
        deadline = time.monotonic() + 5
        while worker.pid not in task_processes(kind):
            code = worker.poll()
            if code is not None:
                if code:
                    raise RuntimeError(f'{kind} exited with code {code}. Check {log}')
                print(f'{kind} completed. Output: {run}')
                return worker
            if time.monotonic() > deadline:
                raise RuntimeError(f'Cannot confirm startup. Check PID {worker.pid} and {log}')
            time.sleep(.05)
        print(f'{kind}: RUNNING | PID: {worker.pid}')
        print(f'Run folder: {run}')
        print(f'Log file: {log}')
        return worker


def status():
    for kind in KINDS:
        pids = task_processes(kind)
        print(f"{kind}: {'RUNNING' if pids else 'STOPPED'}")
        for pid in pids:
            proc = Path('/proc') / str(pid)
            try:
                data = json.loads(record_path(kind).read_text())
                run = Path(data['run_dir'])
                elapsed = max(0, int(time.time()-data['started']))
                print(f'  PID: {pid} | Time used: {elapsed//3600:02d}:{elapsed//60%60:02d}:{elapsed%60:02d}')
                state = run / ('train_state.json' if kind == 'train' else 'state.json')
                if state.is_file():
                    value = json.loads(state.read_text())
                    fields = [f'{k}: {value[k]}' for k in ('status','phase','epoch','epochs','iteration','completed_episodes','expected_episodes') if k in value]
                    print('  ' + ' | '.join(fields))
                    phase = value.get('preparation') if value.get('phase') == 'preparing' else value.get('collection')
                    if phase and 'completed_files' in phase:
                        completed, total = phase['completed_files'], phase['requested_files']
                        print(f'  Progress: {completed}/{total} files')
                    if 'completed_episodes' in value:
                        print(f"  Progress: {value['completed_episodes']}/{value['expected_episodes']} episodes")
                if kind in ('train', 'train_vop'):
                    progress = latest_progress(Path(data['log_file']))
                    if progress:
                        print(f'  Progress: {progress[0]} | Time used: {progress[1]} | ETA: {progress[2]}')
                print(f"  Log file: {data['log_file']}")
            except (OSError, ValueError, KeyError):
                print(f'  PID: {pid} | Process details not available')
    pid = tensorboard_status()
    print(f"tensorboard: {'UNKNOWN' if pid == 'unknown' else 'RUNNING' if pid else 'STOPPED'}")
    if pid:
        print(f'  PID: {pid}')


def main():
    parser = argparse.ArgumentParser(description='Launch and manage project tasks')
    parser.add_argument('action', choices=('launch','status','stop','log-file','active-pids','compare-dir'))
    parser.add_argument('--kind', choices=(*KINDS,'webrtc_udp_bridge'), default='train')
    args, remaining = parser.parse_known_args()
    try:
        if args.action == 'launch':
            launch(args.kind, remaining)
        elif remaining:
            parser.error('Unknown arguments: ' + ' '.join(remaining))
        elif args.action == 'status':
            status()
        elif args.action == 'active-pids':
            print(' '.join(map(str,task_processes(args.kind))))
        elif args.action == 'log-file':
            path = latest_log(args.kind)
            if path:
                print(path)
        elif args.action == 'compare-dir':
            data = json.loads(record_path('compare').read_text())
            run = Path(data['run_dir']).resolve()
            if not run.is_relative_to(PROJECT) or not (run/'index.html').is_file() or not (run/'export.json').is_file() or task_processes('compare'):
                raise RuntimeError('Comparison export is not complete')
            print(run)
        elif not stop_task_processes(args.kind, timeout=30. if args.kind in ('collect','train_vop','evaluate') else 10.):
            raise SystemExit(2)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f'Task command failed: {error}\n')


if __name__ == '__main__':
    main()
