# Runtime profiles

`env.sh` defaults to `local.env`. Set `LOCAL_CONDA_ROOT` and `LOCAL_CONDA_ENV`
before sourcing it if Conda is installed elsewhere. The environment must already
contain IsaacLab and the required third-party dependencies.

For a remote Conda runtime:

```bash
cp scripts/env/servers/remote.env.example scripts/env/servers/remote.env
# Edit remote.env: SSH alias, project path, Conda paths, and streaming settings.
source env.sh --server remote
sync_code --dry-run
sync_code
install_env
```

`sync_code` updates the selected remote project and deletes source files removed
locally. Review the dry run and use a dedicated remote project directory.
Runtime profiles are sourced as shell code; only use trusted files.

For Docker, create a private profile with `CONNECTION_MODE="ssh"`,
`REMOTE_HOST`, `REMOTE_RUNTIME="docker"`, `REMOTE_DIR` (host project path),
`CONTAINER_NAME`, `CONTAINER_DIR` (project mount path), `CONTAINER_PYTHON`,
`REMOTE_PYTHON="${CONTAINER_PYTHON}"`, `REMOTE_STREAM_IP`, `TB_PORT`, and
`LIVESTREAM_PORT`. Use paths and ports for your existing container.

Custom `*.env` profiles are ignored by Git; the generic `local.env` is included.
