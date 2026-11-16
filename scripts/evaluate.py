"""Navigation evaluation with durable per-episode records."""
from datetime import datetime
from pathlib import Path
import signal
import sys
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.evaluation import (ROOT, RunRecords, action, configure_scenes, load_policy,
                              preflight, resolved, print_table)


def main():
    from utils.navigation import parse_arguments, save_arguments, launcher_arguments
    args = parse_arguments('evaluate')
    from isaaclab.app import AppLauncher
    if any(len(set(v)) != len(v) for v in (args.seeds,args.scenarios,args.levels)):
        raise ValueError('duplicate seeds, scenarios or levels')
    if any(not 0 <= s < 2**32 for s in args.seeds): raise ValueError('seeds must be in [0, 2**32)')
    path = resolved(args.run_dir) if args.run_dir else ROOT/'logs/evaluation'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    records = RunRecords(path, args)
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, stop)
    app = None
    env = None
    try:
        preflight(any(s != 'Training' for s in args.scenarios), args.memory_reserve_gib)
        app = AppLauncher(launcher_arguments(args)).app
        # AppLauncher can change handlers; install ours again before scene creation.
        for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, stop)
        import torch
        from envs.go2.evaluation.environment import EvaluationEnv, make_config
        from loco_rl.env.wrapper import NavigationVecEnvWrapper
        from utils.navigation import _set_seed
        torch.set_num_threads(4)
        configure_scenes(args)
        cases = [(s,l) for s in args.scenarios for l in (args.levels if s == 'Training' else [None])]
        initial = make_config(args.scenarios[0],args.num_envs,args.device,args.seeds[0],args.vop_checkpoint,args.levels[0])
        args.num_envs, args.device = initial.scene.num_envs, initial.sim.device
        save_arguments(args,path)
        records.expected=len(cases)*len(args.seeds)*args.num_envs*args.episodes_per_env
        for scenario, level in cases:
            if stopped or (path/'stop_requested.json').exists(): break
            _set_seed(args.seeds[0])
            cfg = make_config(scenario,args.num_envs,args.device,args.seeds[0],resolved(args.vop_checkpoint) if args.vop_checkpoint else None,level or 0)
            args.num_envs = cfg.scene.num_envs
            records.state('initializing', scenario=scenario, level=level)
            from isaacsim.core.utils.stage import create_new_stage
            create_new_stage()
            env=EvaluationEnv(cfg)
            wrapper=NavigationVecEnvWrapper(env)
            model=load_policy(args,wrapper)
            records.metadata(args,env,model)
            try:
                group=(scenario,level); records.groups[group]=False
                if stopped or (path/'stop_requested.json').exists(): break
                for seed in args.seeds:
                    if stopped or (path/'stop_requested.json').exists(): break
                    _set_seed(seed); env.prepare(seed,level); wrapper.reset(); env.rows.clear(); env.record_episodes=True
                    counts=[0]*args.num_envs
                    records.state('running',scenario=scenario,level=level,seed=seed)
                    print(f'[EVAL] {scenario} level={level} seed={seed}',flush=True)
                    with torch.no_grad():
                        for step in range((env.max_episode_length+2)*(args.episodes_per_env+1)):
                            if stopped or (path/'stop_requested.json').exists() or not app.is_running():
                                stopped=True; break
                            wrapper.step(action(model,wrapper.get_observations()))
                            for row in env.rows:
                                i=row['env_id']
                                if counts[i] < args.episodes_per_env:
                                    row.update(scenario=scenario,level=level,seed=seed)
                                    records.append(row); counts[i]+=1
                            env.rows.clear()
                            if step%100==0:
                                records.state('running',scenario=scenario,level=level,seed=seed)
                            if step%500==0: print(f'[EVAL] progress {sum(counts)}/{args.num_envs*args.episodes_per_env}',flush=True)
                            if min(counts)>=args.episodes_per_env: break
                    if not stopped and sum(counts)!=args.num_envs*args.episodes_per_env:
                        raise RuntimeError('Incomplete evaluation group')
                expected=len(args.seeds)*args.num_envs*args.episodes_per_env
                rows=[r for r in records.rows if (r['scenario'],r['level'])==group]
                records.groups[group]=len(rows)==expected
                for summary in records.summary():
                    if (summary['scenario'],summary['level'])==group:
                        print_table([summary])
            finally:
                env.close(); env=None
        records.summary()
        complete=len(records.rows)==records.expected and not stopped
        records.state('completed' if complete else 'stopped')
        print(f'[EVAL] {"COMPLETED" if complete else "STOPPED"}: {path}',flush=True)
        return 0 if complete else 130
    except BaseException as error:
        records.summary(); records.state('failed',error=str(error)); traceback.print_exc()
        return 1
    finally:
        if env is not None: env.close()
        if app is not None: app.close()


if __name__=='__main__': sys.exit(main())
