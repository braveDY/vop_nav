import sys
import traceback

from utils.navigation import parse_arguments, launcher_arguments
from utils.evaluation import configure_scenes, play, preflight


def main():
    args = parse_arguments('play')
    from isaaclab.app import AppLauncher
    preflight(args.scenario != 'Training', args.memory_reserve_gib)
    app = AppLauncher(launcher_arguments(args)).app
    code = 0
    try:
        configure_scenes(args)
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
