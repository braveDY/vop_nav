#!/usr/bin/env bash

_stop_remote_tb() {
    _remote_exec 'if [[ -f .ops/tensorboard.pid ]]; then
        pid=$(cat .ops/tensorboard.pid)
        if [[ ${pid} =~ ^[0-9]+$ ]] && [[ $(readlink "/proc/${pid}/cwd") == "$(pwd)" ]] && ps -p "${pid}" -o args= | grep -q "[t]ensorboard.main"; then
            kill "${pid}" 2>/dev/null || true
            for attempt in {1..50}; do
                kill -0 "${pid}" 2>/dev/null || break
                [[ $(ps -p "${pid}" -o stat=) == Z* ]] && break
                sleep 0.1
            done
            if kill -0 "${pid}" 2>/dev/null && [[ $(ps -p "${pid}" -o stat=) != Z* ]]; then
                echo "TensorBoard for this project did not stop (PID: ${pid})" >&2
                exit 1
            fi
        fi
        rm -f .ops/tensorboard.pid
    fi'
}

stop_tb() (
    cd "${PROJECT_ROOT}" || return 1
    mkdir -p .ops || return 1
    # Use fd 9 so background processes can close it and avoid holding the lock.
    exec 9>".ops/tensorboard_tunnel_${TB_PORT}.lock" || return 1
    flock -n 9 || { _err "Another TensorBoard command is running for this project."; return 1; }
    # Local mode may still have SSH tunnels from a previous server profile.
    python3 scripts/utils/streaming.py tb stop --port "${TB_PORT}" || return 1
    _stop_remote_tb || return 1
    if ! python3 scripts/utils/streaming.py tb port-free --port "${TB_PORT}"; then
        _err "TensorBoard stopped on ${REMOTE_HOST}, but local port ${TB_PORT} is still in use. Check with ss -ltnp 'sport = :${TB_PORT}'."
        return 1
    fi
    _info "Stopped TensorBoard on ${REMOTE_HOST} and its local tunnel on port ${TB_PORT} for this project."
)

tb() (
    local tunnel_pid=""
    if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--stop" ) ]]; then
        _err "Usage: tb to start. Use stop_tb to stop"
        return 1
    fi
    if [[ "${1:-}" == "--stop" ]]; then
        stop_tb
        return $?
    fi
    cd "${PROJECT_ROOT}" || return 1
    mkdir -p .ops || return 1
    # Use fd 9 so background processes can close it and avoid holding the lock.
    exec 9>".ops/tensorboard_tunnel_${TB_PORT}.lock" || return 1
    flock -n 9 || { _err "Another TensorBoard command is running for this project."; return 1; }
    # Local mode may still have SSH tunnels from a previous server profile.
    python3 scripts/utils/streaming.py tb stop --port "${TB_PORT}" || return 1
    _stop_remote_tb || return 1
    if ! python3 - "${TB_PORT}" <<'PY'
import socket, sys
sock = socket.socket()
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock.bind(('127.0.0.1', int(sys.argv[1])))
except OSError:
    sys.exit(1)
finally:
    sock.close()
PY
    then
        _err "Local port ${TB_PORT} is used by another program or project. That process was left running. Check with ss -ltnp 'sport = :${TB_PORT}'."
        return 1
    fi
    _ensure_runtime_running || return 1
    if ! _ssh_command python3 -c 'import socket,sys; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); s.bind(("127.0.0.1",int(sys.argv[1]))); s.close()' "${TB_PORT}"; then
        _err "Server port ${TB_PORT} is not available. TensorBoard not started."
        return 1
    fi
    local target_dir="$(_runtime_dir)/logs" command bind_flag
    bind_flag="--host=127.0.0.1"
    [[ "${REMOTE_RUNTIME}" == "docker" ]] && bind_flag="--bind_all"
    command=$(_quote_args "${REMOTE_PYTHON}" -m tensorboard.main --logdir "${target_dir}" --port "${TB_PORT}" "${bind_flag}")
    _info "Starting TensorBoard on ${REMOTE_HOST} (${target_dir})..."
    ( exec 9>&-; _remote_exec "mkdir -p .ops && { nohup ${command} > .ops/tensorboard.log 2>&1 < /dev/null & tb_pid=\$!; printf '%s\\n' \"\${tb_pid}\" > .ops/tensorboard.pid; for attempt in {1..100}; do child=\$(pgrep -P \"\${tb_pid}\" -f tensorboard.main | head -n 1); if [[ -n \"\${child}\" ]]; then printf '%s\\n' \"\${child}\" > .ops/tensorboard.pid; break; fi; if [[ \$(readlink /proc/\${tb_pid}/exe 2>/dev/null) == *python* ]] && ps -p \"\${tb_pid}\" -o args= | grep -q '[t]ensorboard.main'; then break; fi; kill -0 \"\${tb_pid}\" 2>/dev/null || exit 1; sleep 0.1; done; }" ) || {
        _err "TensorBoard failed to start. Check .ops/tensorboard.log on the server."
        return 1
    }

    trap '[[ -n "${tunnel_pid}" ]] && kill "${tunnel_pid}" 2>/dev/null || true; if [[ "${CONNECTION_MODE:-ssh}" != local ]]; then python3 scripts/utils/streaming.py tb stop --port "${TB_PORT}" >/dev/null 2>&1 || true; fi; _stop_remote_tb >/dev/null 2>&1 || true' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    if [[ "${CONNECTION_MODE:-ssh}" != local ]]; then
        tunnel_pid=$(exec 9>&-; python3 scripts/utils/streaming.py tb launch --port "${TB_PORT}" --host "${REMOTE_HOST}") || return 1
    fi
    local ready=false
    for attempt in {1..60}; do
        if [[ -n "${tunnel_pid}" ]]; then
            kill -0 "${tunnel_pid}" 2>/dev/null || { _err "SSH tunnel failed to start."; return 1; }
        fi
        _remote_exec 'pid=$(cat .ops/tensorboard.pid); kill -0 "$pid" 2>/dev/null && [[ $(readlink "/proc/$pid/cwd") == "$(pwd)" ]] && ps -p "$pid" -o args= | grep -q "[t]ensorboard.main"' >/dev/null 2>&1 || {
            _err "TensorBoard failed to start. Check .ops/tensorboard.log on the server."
            return 1
        }
        if python3 - "${TB_PORT}" "${target_dir}" <<'PY'
import json, sys, urllib.request
try:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f'http://127.0.0.1:{sys.argv[1]}/data/environment', timeout=1) as response:
        if json.load(response).get('data_location') != sys.argv[2]:
            sys.exit(1)
except Exception:
    sys.exit(1)
PY
        then
            ready=true
            break
        fi
        sleep 0.5
    done
    [[ "${ready}" == true ]] || { _err "TensorBoard start timed out or its log folder does not match."; return 1; }
    trap - EXIT INT TERM
    _info "TensorBoard: http://localhost:${TB_PORT}. Use stop_tb to stop."
)

status() {
    _info "Server: ${REMOTE_HOST} | Runtime: ${REMOTE_RUNTIME} | Project: ${REMOTE_DIR}"
    _ssh_command bash -s -- "${REMOTE_RUNTIME}" "${CONTAINER_NAME:-}" "${REMOTE_DIR}" "${REMOTE_PYTHON}" <<'EOF' || return 1
        runtime="$1" container="$2" project="$3" python="$4"
        if [[ "${runtime}" == "docker" ]]; then
            docker ps -a --filter "name=^/${container}$" --format 'Container: {{.Names}} | Status: {{.Status}} | Ports: {{.Ports}}'
        else
            if [[ -x "${python}" ]]; then "${python}" --version; else echo "Conda Python not found: ${python}"; fi
            if [[ -d "${project}" ]]; then echo "Project folder exists"; else echo "Project code not synced. Run sync_code"; fi
            if command -v rsync >/dev/null; then echo "rsync is installed"; else echo "rsync is not installed. Install it manually"; fi
        fi
        if command -v nvidia-smi >/dev/null; then
            nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "GPU information not available"
        fi
EOF
    _process_exec status || {
        _err "Cannot check project processes. Status unknown for train, play, collect, train_vop, evaluate and tensorboard."
        return 1
    }
}

enter() {
    local command
    _info "Opening a terminal on ${REMOTE_HOST} (${REMOTE_RUNTIME})..."
    if [[ "${REMOTE_RUNTIME}" == "docker" ]]; then
        command=$(_quote_args docker exec -it -w "${CONTAINER_DIR}" "${CONTAINER_NAME}" bash)
    else
        command="source $(_quote_args "${REMOTE_CONDA_ROOT}/etc/profile.d/conda.sh") && conda activate $(_quote_args "${REMOTE_CONDA_ENV}") && _torch_lib=\$(ls -d \${CONDA_PREFIX}/lib/python*/site-packages/torch/lib 2>/dev/null | head -n 1) && [[ -n \${_torch_lib} ]] && export LD_LIBRARY_PATH=\"\${_torch_lib}:\${LD_LIBRARY_PATH:-}\" && { if [[ -d $(_quote_args "${REMOTE_DIR}") ]]; then cd $(_quote_args "${REMOTE_DIR}"); fi; } && exec bash -i"
        command=$(_quote_args bash -c "${command}")
    fi
    _host_shell "${command}"
}
