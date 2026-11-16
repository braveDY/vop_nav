import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import signal

ROOT = Path(__file__).resolve().parents[2]
OUTCOMES = {0: 'timeout', 1: 'success', 2: 'collision', 3: 'fall', 4: 'outside'}


def dependency_path():
    binary_dir = ROOT / '.deps/rvo2'
    if binary_dir.is_dir(): sys.path.insert(0, str(binary_dir))


def preflight(unseen=True, reserve=4.):
    if not 0 <= reserve < float('inf'):
        raise ValueError('--memory_reserve_gib must be finite and nonnegative')
    if unseen:
        dependency_path()
        try:
            import rvo2
        except ImportError as error:
            raise RuntimeError('RVO dependency missing. Install Python-RVO2 manually in the selected environment.') from error
    memory = Path('/sys/fs/cgroup/memory')
    if (memory / 'memory.limit_in_bytes').exists():
        stats = dict(line.split() for line in (memory / 'memory.stat').read_text().splitlines())
        available = (int((memory / 'memory.limit_in_bytes').read_text())
                     - int((memory / 'memory.usage_in_bytes').read_text())
                     + int(stats.get('total_inactive_file', 0)))
        if available < reserve * 2**30:
            raise RuntimeError(f'Insufficient container RAM: {available/2**30:.2f} GiB available')


def resolved(path):
    path = path.expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def load_policy(args, wrapper):
    import torch
    from loco_rl.modules import BasicNavActorCritic, VOPNavActorCritic
    checkpoint = resolved(args.checkpoint)
    config = json.loads((checkpoint.parent / 'config.json').read_text())['train']
    classes = dict(BasicNavActorCritic=BasicNavActorCritic, VOPNavActorCritic=VOPNavActorCritic)
    name = config['runner']['policy_class_name']
    expected = 'VOPNavActorCritic' if args.vop_checkpoint else 'BasicNavActorCritic'
    if name != expected:
        raise ValueError(f'Checkpoint policy is {name}; selected configuration expects {expected}')
    model = classes[name](wrapper.num_obs, wrapper.num_privileged_obs, 12,
                              **config['policy']).to(wrapper.device)
    saved = torch.load(checkpoint, map_location=wrapper.device, weights_only=False)
    model.load_state_dict(saved['model_state_dict'])
    model.eval().requires_grad_(False)
    return model


def action(model, observations):
    import torch
    result = model.act_inference(observations)
    if not torch.isfinite(result).all(): raise FloatingPointError('Nonfinite evaluation action')
    return result


def summarize(rows, **labels):
    if not rows: return dict(**labels, episodes=0)
    result = dict(**labels, episodes=len(rows))
    for key in ['success', 'deadline_success', 'strict_deadline_success', 'collision', 'fall',
                'outside', 'timeout', 'ever_collision', 'ever_arrived', 'ever_tilted',
                'ever_base_contact', 'distance', 'duration']:
        result[key] = sum(row[key] for row in rows) / len(rows)
    result.update({'first_'+name: sum(row['first_outcome'] == code for row in rows)/len(rows)
                   for code, name in OUTCOMES.items()})
    return result


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str))
    temporary.replace(path)


class RunRecords:
    def __init__(self, path, args):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        if any(p.name not in ('evaluate.log', 'play.log', 'launch.json', 'exit_code', 'stop_requested.json', 'arguments.json')
               for p in self.path.iterdir()):
            raise FileExistsError(f'Result directory is not empty: {self.path}')
        # Exclusive creation also prevents two Python processes from claiming one run.
        with (self.path / 'config.json').open('x') as stream:
            json.dump(vars(args), stream, indent=2, default=str)
        self.started = time.time()
        self.rows = []
        self.groups = {}
        self.expected = 0
        self.environments = {}
        self.state('starting')

    def state(self, status, **details):
        atomic_json(self.path / 'state.json', dict(status=status, pid=os.getpid(),
                    completed_episodes=len(self.rows), expected_episodes=self.expected,
                    elapsed_s=time.time()-self.started, updated_unix=time.time(), **details))

    def metadata(self, args, env, model):
        from envs.go2.evaluation.layouts import ASSUMPTIONS, SCENARIOS
        key = f'{env.cfg.evaluation_scenario}:{env.cfg.evaluation_level}'
        self.environments[key] = env.cfg.to_dict()
        data = dict(checkpoint_sha256=file_sha256(resolved(args.checkpoint)),
                    vop_sha256=file_sha256(resolved(args.vop_checkpoint)) if args.vop_checkpoint else None,
                    assumptions=ASSUMPTIONS, scenarios=SCENARIOS,
                    environments=self.environments,
                    metrics_version=2, metric_sampling_s=env.step_dt, primary_success='first_outcome == success; distance < 0.5m',
                    event_priority=list(OUTCOMES.values())[2:]+['success'],
                    collision='0.35m circular body footprint; geometric overlap; may precede physical contact',
                    arrival_behavior='continue to deadline; falls/outside still end episode',
                    observation_noise=False, pushes=False,
                    seed_protocol='per (seed, env_id, episode)',
                    code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [ROOT/'scripts/evaluate.py', ROOT/'scripts/utils/evaluation.py', *sorted((ROOT/'source/envs/go2/evaluation').glob('*.py'))]},
                    rvo_dependency=json.loads((ROOT/'.deps/rvo2/install.json').read_text())
                    if (ROOT/'.deps/rvo2/install.json').exists() else None)
        atomic_json(self.path / 'metadata.json', data)

    def append(self, row):
        self.rows.append(row)
        with (self.path / 'episodes.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')

    def summary(self):
        summaries = []
        for key, complete in self.groups.items():
            scenario, level = key
            rows = [r for r in self.rows if (r['scenario'], r['level']) == key]
            summaries.append(summarize(rows, scenario=scenario, level=level, complete=complete))
        atomic_json(self.path / 'summary.json', summaries)
        if summaries:
            with (self.path / 'summary.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in summaries for k in row)))
                writer.writeheader(); writer.writerows(summaries)
        return summaries


def print_table(summaries):
    print('Scenario   Level Episodes Complete FirstSafe Deadline StrictDeadline Fall EverCollision', flush=True)
    for row in summaries:
        values = [f'{100*row[key]:6.1f}%' if key in row else '     --'
                  for key in ('success', 'deadline_success', 'strict_deadline_success', 'fall', 'ever_collision')]
        print(f'{row["scenario"]:<10} {str(row["level"]):>5} '
              f'{row["episodes"]:>8} {str(row["complete"]):>8} ' + ' '.join(values), flush=True)


def play(args, app):
    from datetime import datetime
    import torch
    from envs.go2.evaluation.environment import EvaluationEnv, make_config
    from loco_rl.env.wrapper import NavigationVecEnvWrapper
    from .navigation import _set_seed, save_arguments
    preflight(args.scenario != 'Training', args.memory_reserve_gib)
    path=resolved(args.run_dir) if args.run_dir else ROOT/'logs/play'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    records=RunRecords(path,args)
    env=None; stopped=False
    def stop(signum, frame):
        nonlocal stopped
        stopped=True
    previous={sig:signal.signal(sig,stop) for sig in (signal.SIGTERM,signal.SIGINT)}
    try:
        seed = args.seed if args.seed is not None else 123
        _set_seed(seed)
        cfg=make_config(args.scenario,args.num_envs,args.device,seed,resolved(args.vop_checkpoint) if args.vop_checkpoint else None,args.level,task=args.task)
        args.num_envs, args.device, args.seed = cfg.scene.num_envs, cfg.sim.device, seed
        save_arguments(args,path)
        env=EvaluationEnv(cfg); env.stream_render=bool(args.livestream>0)
        wrapper=NavigationVecEnvWrapper(env)
        env.prepare(args.seed,args.level); wrapper.reset(); env.rows.clear(); env.record_episodes=True
        model=load_policy(args,wrapper); records.metadata(args,env,model)
        step=0
        records.groups[(args.scenario,args.level if args.scenario=='Training' else None)]=False
        with torch.no_grad():
            while not stopped and not (path/"stop_requested.json").exists() and app.is_running() and (not args.steps or step<args.steps):
                start=time.perf_counter()
                wrapper.step(action(model,wrapper.get_observations()))
                rows=list(env.rows)
                for row in rows:
                    row.update(scenario=args.scenario,level=args.level if args.scenario=='Training' else None,
                               seed=args.seed)
                    records.append(row)
                env.rows.clear(); step+=1
                if step%100==0: records.state('running',steps=step,scenario=args.scenario)
                if rows:
                    print(f'[PLAY] completed={len(records.rows)} first_success={sum(r["success"] for r in records.rows)}/{len(records.rows)}',flush=True)
                if args.real_time: time.sleep(max(0.,env.step_dt-(time.perf_counter()-start)))
        records.summary(); records.state('stopped' if stopped or not args.steps or step<args.steps else 'completed',steps=step)
        print(f'[PLAY] FINISHED: {path}',flush=True)
    except BaseException as error:
        records.summary(); records.state('failed',error=str(error)); raise
    finally:
        if env is not None: env.close()
        for sig,handler in previous.items(): signal.signal(sig,handler)


def configure_scenes(args):
    from envs.go2.evaluation.layouts import SCENARIOS
    selected = [args.scenario] if hasattr(args, 'scenario') else args.scenarios
    if not selected or any(name not in ('Training', *SCENARIOS) for name in selected):
        raise ValueError('Select a valid scene name')
    levels = [args.level] if hasattr(args, 'level') else args.levels
    if not levels or any(level not in range(10) for level in levels):
        raise ValueError('Scene levels must be in 0..9')
    if 'VOP-Nav' in args.task and not args.vop_checkpoint:
        raise ValueError('Set vop_checkpoint for VOP-Nav')
    if 'VOP-Nav' not in args.task and args.vop_checkpoint:
        raise ValueError('BasicNav does not use a VOP-Net checkpoint')
