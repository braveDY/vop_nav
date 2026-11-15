#!/usr/bin/env bash

_launch_task() {
    local kind="$1" follow=false arg
    shift
    local arguments=()
    for arg in "$@"; do
        if [[ "$arg" == --log ]]; then follow=true; else arguments+=("$arg"); fi
    done
    for arg in "${arguments[@]}"; do
        if [[ "$arg" == --help || "$arg" == -h ]]; then
            PYTHONPATH="${PROJECT_ROOT}/source${PYTHONPATH:+:${PYTHONPATH}}" python3 -c 'from stage2.config import parse_config; import sys; parse_config(sys.argv[1],sys.argv[2:])' "$kind" --help
            return $?
        fi
    done
    if [[ "$kind" == compare && "${CONNECTION_MODE:-ssh}" == local ]]; then
        _err "Comparison export runs on the server. Select a remote server first."
        return 1
    fi
    if [[ "$kind" == play && "${CONNECTION_MODE:-ssh}" == local && -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
        _err "Local playback needs a desktop display (DISPLAY or WAYLAND_DISPLAY)."
        return 1
    fi
    _ensure_runtime_running || return 1
    _process_exec launch --kind "$kind" "${arguments[@]}" || return 1
    if [[ "$kind" == play && "${CONNECTION_MODE:-ssh}" != local ]]; then
        _start_play_stream || { _err "Playback started, but the stream tunnel failed. Use stop play to stop it."; return 1; }
    fi
    if [[ "$follow" == true ]]; then log "$kind"; fi
}

train() { _launch_task train "$@"; }
collect() { _launch_task collect "$@"; }
train_vop() { _launch_task train_vop "$@"; }
compare() { _launch_task compare "$@"; }
play() { _launch_task play "$@"; }
evaluate() { _launch_task evaluate "$@"; }

log() {
    local path
    path=$(_process_exec log-file --kind "${1:-train}") || return 1
    [[ -n "$path" ]] || { _err "No log found for ${1:-train}."; return 1; }
    _info "Following $path. Ctrl+C stops log viewing only."
    _remote_exec "tail -n 50 -f $(_quote_args "$path")"
}

stop() {
    [[ $# == 1 ]] || { _err "Usage: stop <train|collect|train_vop|compare|play|evaluate>"; return 1; }
    _process_exec stop --kind "$1" || return 1
    if [[ "$1" == play && "${CONNECTION_MODE:-ssh}" != local ]]; then
        _process_exec stop --kind webrtc_udp_bridge || return 1
        _stop_play_stream
    fi
}

download() {
    [[ "${CONNECTION_MODE:-ssh}" != local ]] || { _err "Select a remote server first."; return 1; }
    [[ $# -le 1 ]] || { _err "Usage: download [new-local-folder]"; return 1; }
    local remote_path destination host_path
    remote_path=$(_process_exec compare-dir) || return 1
    host_path="${REMOTE_DIR}/${remote_path#"$(_runtime_dir)/"}"
    destination="${1:-${PROJECT_ROOT}/exports/$(basename "$remote_path")}"
    [[ ! -e "$destination" ]] || { _err "Destination must be new: $destination"; return 1; }
    mkdir -p "$destination" || return 1
    rsync -az -s "${REMOTE_HOST}:${host_path}/" "${destination}/" || return 1
    _info "Downloaded: ${destination}/index.html"
}

_start_play_stream() {
    local stream_port="${LIVESTREAM_PORT:-49100}"
    local stream_display_ip="${REMOTE_STREAM_IP:-${REMOTE_IP}}"
    if [[ "$stream_display_ip" == 127.0.0.1 || "$stream_display_ip" == localhost ]]; then
        local tunnel_key="${REMOTE_HOST}_${stream_port}"
        local tunnel_file="/tmp/.webrtc_${tunnel_key//[^a-zA-Z0-9]/_}.pid"
        local bridge_pid_file="/tmp/.webrtc_bridge_${tunnel_key//[^a-zA-Z0-9]/_}.pid"
        if [[ -f "${tunnel_file}" ]]; then
            local old_pid
            old_pid=$(cat "${tunnel_file}" 2>/dev/null || true)
            if [[ "${old_pid}" =~ ^[0-9]+$ ]] && tr '\0' ' ' < "/proc/${old_pid}/cmdline" 2>/dev/null | grep -Fq "${stream_port}:localhost:${stream_port}"; then
                kill "${old_pid}" 2>/dev/null || true
            fi
            rm -f "${tunnel_file}"
        fi
        if [[ -f "${bridge_pid_file}" ]]; then
            local old_bpid
            old_bpid=$(cat "${bridge_pid_file}" 2>/dev/null || true)
            if [[ "${old_bpid}" =~ ^[0-9]+$ ]] && tr '\0' ' ' < "/proc/${old_bpid}/cmdline" 2>/dev/null | grep -Fq "${PROJECT_ROOT}/scripts/utils/streaming.py"; then
                kill "${old_bpid}" 2>/dev/null || true
            fi
            rm -f "${bridge_pid_file}"
        fi

        _info "Setting up the local SSH tunnel for WebRTC (ports: ${stream_port}, 8011, 47997)..."
        _process_exec stop --kind webrtc_udp_bridge || return 1
        _remote_launch "logs/webrtc_remote_bridge.log" "$(_quote_args "${REMOTE_PYTHON}" "scripts/utils/streaming.py" bridge remote)" || return 1
        sleep 0.5
        if ssh -f -N -o ExitOnForwardFailure=yes -o ConnectTimeout=10 \
            -L "${stream_port}:localhost:${stream_port}" \
            -L "8011:localhost:8011" \
            -L "47997:localhost:47997" "${REMOTE_HOST}"; then
            local tunnel_pid
            tunnel_pid=$(pgrep -f "ssh.*-L.*${stream_port}:localhost:${stream_port}" | head -n 1)
            [[ -n "${tunnel_pid}" ]] && printf '%s\n' "${tunnel_pid}" > "${tunnel_file}"

            nohup python3 "${PROJECT_ROOT}/scripts/utils/streaming.py" bridge local </dev/null >/tmp/webrtc_local_bridge.log 2>&1 &
            local bridge_pid=$!
            printf '%s\n' "${bridge_pid}" > "${bridge_pid_file}"
            _info "\033[1;32m✓ SSH tunnel and UDP bridge are ready.\033[0m Connect with Isaac Sim WebRTC Streaming Client at: \033[1;36m127.0.0.1\033[0m"
        else
            _err "SSH tunnel setup failed. Check local ports ${stream_port} and 8011."
            return 1
        fi
    else
        _info "Connect the Isaac Sim WebRTC client to ${stream_display_ip}:${stream_port}."
    fi
}

_stop_play_stream() {
    local stream_port="${LIVESTREAM_PORT:-49100}"
    local tunnel_key="${REMOTE_HOST}_${stream_port}"
    local tunnel_file="/tmp/.webrtc_${tunnel_key//[^a-zA-Z0-9]/_}.pid"
    local bridge_pid_file="/tmp/.webrtc_bridge_${tunnel_key//[^a-zA-Z0-9]/_}.pid"
    if [[ -f "${tunnel_file}" ]]; then
        local tunnel_pid
        tunnel_pid=$(cat "${tunnel_file}" 2>/dev/null || true)
        if [[ "${tunnel_pid}" =~ ^[0-9]+$ ]]; then
            if tr '\0' ' ' < "/proc/${tunnel_pid}/cmdline" 2>/dev/null | grep -Fq "${stream_port}:localhost:${stream_port}"; then
                kill "${tunnel_pid}" 2>/dev/null || true
            fi
        fi
        rm -f "${tunnel_file}"
    fi
    if [[ -f "${bridge_pid_file}" ]]; then
        local bridge_pid
        bridge_pid=$(cat "${bridge_pid_file}" 2>/dev/null || true)
        if [[ "${bridge_pid}" =~ ^[0-9]+$ ]]; then
            if tr '\0' ' ' < "/proc/${bridge_pid}/cmdline" 2>/dev/null | grep -Fq "${PROJECT_ROOT}/scripts/utils/streaming.py"; then
                kill "${bridge_pid}" 2>/dev/null || true
            fi
        fi
        rm -f "${bridge_pid_file}"
    fi

}
