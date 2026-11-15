#!/usr/bin/env bash

_ssh_command() {
    if [[ "${CONNECTION_MODE:-ssh}" == local ]]; then
        "$@"
    else
        ssh -o ConnectTimeout=10 "${REMOTE_HOST}" "$(_quote_args "$@")"
    fi
}

# Run a host shell; keep the remote PTY for logs and interactive sessions.
_host_shell() {
    if [[ "${CONNECTION_MODE:-ssh}" == local ]]; then
        bash -c "$1"
    else
        ssh -t -o ConnectTimeout=10 "${REMOTE_HOST}" "$1"
    fi
}

_runtime_dir() {
    if [[ "${REMOTE_RUNTIME}" == "docker" ]]; then
        printf '%s' "${CONTAINER_DIR}"
    else
        printf '%s' "${REMOTE_DIR}"
    fi
}

_remote_exec() {
    local command="$1" script runtime_dir
    runtime_dir=$(_runtime_dir)
    local project_pp="${runtime_dir}:${runtime_dir}/source"
    script="cd $(_quote_args "${runtime_dir}") && export TZ=Asia/Singapore PYTHONUNBUFFERED=1 TMPDIR=\${HOME}/isaac_sim_local/tmp"
    if [[ "${CONNECTION_MODE:-ssh}" == local ]]; then
        script="cd $(_quote_args "${runtime_dir}") && export TZ=Asia/Singapore PYTHONUNBUFFERED=1 TMPDIR=\${TMPDIR:-/tmp}"
    fi
    script+=" PUBLIC_IP=$(_quote_args "${REMOTE_IP}") HOST_PROJECT_ROOT=$(_quote_args "${REMOTE_DIR}") RUN_MODE=$(_quote_args "${CONNECTION_MODE:-ssh}")"
    script+=" REMOTE_STREAM_IP=$(_quote_args "${REMOTE_STREAM_IP:-${REMOTE_IP}}") LIVESTREAM_PORT=$(_quote_args "${LIVESTREAM_PORT:-49100}")"
    script+=" && ${command}"
    if [[ "${REMOTE_RUNTIME}" == "docker" ]]; then
        script="export PYTHONPATH=\"${project_pp}\${PYTHONPATH:+:\${PYTHONPATH}}\" && ${script}"
        _ssh_command docker exec -w "${runtime_dir}" "${CONTAINER_NAME}" bash -c "${script}"
    else
        local conda_env_setup="export CONDA_NO_PLUGINS=true && source $(_quote_args "${REMOTE_CONDA_ROOT}/etc/profile.d/conda.sh") && conda activate $(_quote_args "${REMOTE_CONDA_ENV}")"
        conda_env_setup+=" && _torch_lib=\$(ls -d \${CONDA_PREFIX}/lib/python*/site-packages/torch/lib 2>/dev/null | head -n 1) && [[ -n \${_torch_lib} ]] && export LD_LIBRARY_PATH=\"\${_torch_lib}:\${LD_LIBRARY_PATH:-}\""
        if [[ -n "${REMOTE_LD_LIBRARY_PATH:-}" ]]; then
            conda_env_setup+=" && export LD_LIBRARY_PATH=\"${REMOTE_LD_LIBRARY_PATH}:\${LD_LIBRARY_PATH:-}\""
        fi
        conda_env_setup+=" && _conda_sp=\$(ls -d \${CONDA_PREFIX}/lib/python*/site-packages 2>/dev/null | head -n 1)"
        conda_env_setup+=" && export PYTHONPATH=\"${project_pp}\${_conda_sp:+:\${_conda_sp}}\${PYTHONPATH:+:\${PYTHONPATH}}\""
        script="${conda_env_setup} && ${script}"
        _ssh_command bash -c "${script}"
    fi
}

_remote_launch() {
    local log_file="$1" command="$2"
    _remote_exec "mkdir -p $(_quote_args "$(dirname "${log_file}")") && { nohup bash -c $(_quote_args "exec ${command}") > $(_quote_args "${log_file}") 2>&1 < /dev/null & launch_pid=\$!; sleep 0.2; kill -0 \"\${launch_pid}\" 2>/dev/null; }"
}

_ensure_runtime_running() {
    if [[ "${REMOTE_RUNTIME}" == "conda" ]]; then
        _ssh_command bash -c "test -f $(_quote_args "${REMOTE_CONDA_ROOT}/etc/profile.d/conda.sh") && test -x $(_quote_args "${REMOTE_PYTHON}")" || {
            _err "Conda environment not available: ${REMOTE_CONDA_ENV}. Check the Conda path: ${REMOTE_CONDA_ROOT}"
            return 1
        }
        return 0
    fi
    local running
    running=$(_ssh_command docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null) || return 1
    if [[ "${running}" != "true" ]]; then
        _warn "Container ${CONTAINER_NAME} is not running. Starting it..."
        _ssh_command docker start "${CONTAINER_NAME}" >/dev/null || {
            _err "Cannot start container ${CONTAINER_NAME}. Check Docker on the server."
            return 1
        }
        sleep 1
    fi
}
