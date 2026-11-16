# Runtime profiles

By default, `source env.sh` selects `local.env`. Prepare IsaacLab and its dependencies first:

```bash
export LOCAL_CONDA_ROOT="$HOME/miniconda3"
export LOCAL_CONDA_ENV="$LOCAL_CONDA_ROOT/envs/isaaclab"
source env.sh
install_env
```

For SSH with Conda, copy `remote.env.example` to `<name>.env`, replace the host and absolute paths, then run `source env.sh --server <name>`. Personal `.env` profiles are ignored by Git.

For SSH with Docker, set `REMOTE_RUNTIME="docker"`, `REMOTE_HOST`, the host project path `REMOTE_DIR`, `CONTAINER_NAME`, `CONTAINER_DIR`, `CONTAINER_PYTHON`, and `REMOTE_PYTHON="${CONTAINER_PYTHON}"`. Prepare the container and project mount manually. Set `REMOTE_STREAM_IP` to the address reachable by the playback client.

`sync_code --dry-run` previews synchronization; `sync_code` updates only `configs/`, `source/`, `scripts/`, optional `tests/`, and `env.sh`. The private top-level `record/` repository is outside that scope. Synchronization deletes removed files within the selected code directories; logs and checkpoints are outside those directories.

See the root Chinese guide for training, playback, evaluation, and task management.
