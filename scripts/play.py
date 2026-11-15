import sys
import traceback

from stage2.config import parse_config, save_config
from utils.evaluation import configure_scenes, play, preflight


def main():
    args = parse_config('play')
    if not args.checkpoint:
        raise ValueError('Set checkpoint in the play YAML file')
    if args.steps < 0 or args.num_envs < 1:
        raise ValueError('steps must be nonnegative and num_envs positive')
    from isaaclab.app import AppLauncher
    preflight(args.scenario != 'Training', args.memory_reserve_gib)
    app = AppLauncher(vars(args).copy()).app
    code = 0
    try:
        configure_scenes(args)
        save_config(args,args.run_dir)
        play(args,app)
    except Exception:
        traceback.print_exc()
        code = 1
    finally:
        if code:
            app.app.post_quit(code)
        app.close()
    return code


if __name__ == '__main__':
    sys.exit(main())
