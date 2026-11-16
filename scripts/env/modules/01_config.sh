#!/usr/bin/env bash

if [[ -z "${PROJECT_ROOT:-}" ]]; then
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
export PROJECT_ROOT

SERVER="${SERVER:-local}"
_ACTIVE_SERVER="${SERVER}"

_load_server_profile() {
    # Clear settings from the previous server before loading this profile.
    unset CONNECTION_MODE REMOTE_HOST REMOTE_RUNTIME REMOTE_DIR
    unset CONTAINER_NAME CONTAINER_DIR CONTAINER_PYTHON REMOTE_PYTHON
    unset REMOTE_CONDA_ROOT REMOTE_CONDA_ENV REMOTE_LD_LIBRARY_PATH
    unset REMOTE_STREAM_IP LIVESTREAM_PORT REMOTE_IP TB_PORT

    local server_cfg="${PROJECT_ROOT}/scripts/env/servers/${_ACTIVE_SERVER}.env"
    if [[ -f "${server_cfg}" ]]; then
        # shellcheck source=/dev/null
        source "${server_cfg}"
    else
        _err "Server config file not found: ${server_cfg}"
        local available_servers
        available_servers=$(find "${PROJECT_ROOT}/scripts/env/servers" -name "*.env" -not -name "template.env" 2>/dev/null | xargs -n1 basename 2>/dev/null | sed 's/\.env$//' | tr '\n' ' ')
        _info "Available server configs: [ ${available_servers}]"
        return 1
    fi
}

_detect_remote_connection() {
    _load_server_profile || return 1

    CONNECTION_MODE="${CONNECTION_MODE:-ssh}"
    if [[ "${CONNECTION_MODE}" != ssh && "${CONNECTION_MODE}" != local ]]; then
        _err "CONNECTION_MODE must be ssh or local"
        return 1
    fi
    export CONNECTION_MODE
    if [[ "${CONNECTION_MODE}" == local ]]; then
        export REMOTE_HOST=localhost
        _CONNECTION_DESC="Local mode (${REMOTE_RUNTIME})"
    elif [[ -n "${REMOTE_HOST_OVERRIDE:-}" ]]; then
        export REMOTE_HOST="${REMOTE_HOST_OVERRIDE}"
        _CONNECTION_DESC="User setting (${REMOTE_HOST})"
    else
        export REMOTE_HOST="${REMOTE_HOST:-${_ACTIVE_SERVER}}"
        _CONNECTION_DESC="${_ACTIVE_SERVER} (${REMOTE_HOST} / ${REMOTE_RUNTIME:-conda})"
    fi

    local address
    if [[ "${CONNECTION_MODE}" == local ]]; then
        address=127.0.0.1
    else
        address=$(ssh -G "${REMOTE_HOST}" 2>/dev/null | awk '$1 == "hostname" {print $2; exit}')
    fi
    # The public stream address may differ from the SSH address.
    if [[ -n "${REMOTE_STREAM_IP:-}" ]]; then
        REMOTE_IP="${REMOTE_STREAM_IP}"
    else
        REMOTE_IP="${address:-${REMOTE_HOST}}"
    fi
    export REMOTE_IP

    if [[ "${REMOTE_RUNTIME}" != "conda" && "${REMOTE_RUNTIME}" != "docker" ]]; then
        _err "REMOTE_RUNTIME must be conda or docker (current: ${REMOTE_RUNTIME})"
        return 1
    fi

    TB_PORT="${TB_PORT:-16086}"
    export TB_PORT
    if [[ -n "${LIVESTREAM_PORT:-}" ]]; then
        export LIVESTREAM_PORT
    fi
    return 0
}

_detect_remote_connection || return 1 2>/dev/null || exit 1

detect_remote() {
    _info "Reloading server config..."
    _detect_remote_connection
    echo -e "🔗 Connection: \033[1;36m${_CONNECTION_DESC}\033[0m"
}

_show_env_help() {
    cat <<'HELP'
  train / collect / train_vop / play / evaluate / compare
      Navigation uses --task and native Python configs; stage two uses YAML.
      Use --help for task options.
      Add --log to follow output. Compare runs on a remote server.
  download [new-local-folder]  Download the latest complete comparison page
  status                     Show tasks, progress, logs and GPU information
  log <kind>                 Follow the latest task log
  stop <kind>                Stop all tasks of that kind
  sync_code [--dry-run]      Manually sync code and YAML to the selected server
  install_env                Install project packages only
  tb / stop tb               Start or stop TensorBoard and its SSH tunnel
  enter / detect_remote      Open the runtime shell or reload the server profile
HELP
}

env_help() { _show_env_help; }
