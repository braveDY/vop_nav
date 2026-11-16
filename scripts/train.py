import sys
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.navigation import create_env, load_runner, learn, parse_arguments, launcher_arguments
args = parse_arguments('train')
from isaaclab.app import AppLauncher
app = AppLauncher(launcher_arguments(args)).app
exit_code = 0
try:
    env, wrapper, config, run = create_env(args)
    try:
        runner = load_runner(wrapper, config, run, args.checkpoint)
        learn(runner, env, wrapper, config, run, args, app)
        if runner.writer:
            runner.writer.flush()
            runner.writer.close()
        print(f'[Navigation] TRAINING FINISHED: {run}', flush=True)
    finally:
        env.close()
except Exception:
    traceback.print_exc()
    exit_code = 1
finally:
    if exit_code:
        app.app.post_quit(exit_code)
    app.close()
sys.exit(exit_code)
