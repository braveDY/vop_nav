#!/usr/bin/env bash

sync_code() {
    if [[ "${CONNECTION_MODE:-ssh}" == local ]]; then
        _info "Local mode uses ${PROJECT_ROOT}. No code sync needed."
        return 0
    fi
    local dry_run=""
    if [[ "${1:-}" == "--dry-run" || "${1:-}" == "-n" ]]; then
        dry_run="--dry-run"
        _info "Dry run: show changes only..."
    fi

    local project_root="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"

    _info "Syncing changed code files to ${REMOTE_HOST}:${REMOTE_DIR}..."

    command -v rsync >/dev/null || { _err "rsync is not installed locally."; return 1; }
    _ssh_command bash -c 'command -v rsync >/dev/null' || {
        _err "rsync is not installed on the server. Install it manually."
        return 1
    }
    if [[ -z "${dry_run}" ]]; then
        _info "Will update the selected server files and delete files removed locally. Logs and checkpoints are kept."
        _ssh_command mkdir -p "${REMOTE_DIR}" || return 1
    fi

    local sources=(
        "${project_root}/configs"
        "${project_root}/source"
        "${project_root}/scripts"
        "${project_root}/env.sh"
    )
    [[ -d "${project_root}/tests" ]] && sources+=("${project_root}/tests")

    local rsync_output
    rsync_output=$(rsync -az -s -O --no-owner --no-group --delete ${dry_run} \
        --out-format="%i|%n" \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='*.egg-info' \
        --exclude='.git' \
        --exclude='.vscode' \
        --exclude='docs' \
        --exclude='logs' \
        "${sources[@]}" \
        "${REMOTE_HOST}:${REMOTE_DIR}/" 2>&1) || {
            _err "Code sync failed: ${rsync_output}"
            return 1
        }

    local changed_files
    changed_files=$(echo "${rsync_output}" | awk -F'|' '
        /^.f/ {print $2}
        /^\*deleting/ && $2 !~ /\/$/ {print "[Deleted] " $2}
    ')

    if [[ -n "${changed_files}" ]]; then
        _info "Changed files:"
        while IFS= read -r file; do
            [[ -n "${file}" ]] && echo -e "  \033[1;32m↑\033[0m ${file}"
        done <<< "${changed_files}"
    else
        _info "Code is up to date. No files need an update."
    fi

    if [[ -z "${dry_run}" ]]; then
        _info "Code sync done."
    else
        _info "Dry run done."
    fi
}

install_env() {
    _ensure_runtime_running || return 1
    _info "Installing project packages only. Prepare third-party dependencies manually."
    _remote_exec "$(_quote_args "${REMOTE_PYTHON}" -m pip install --no-deps --no-build-isolation -e source/assets -e source/loco_rl -e source/envs -e source/stage2)" || return 1
    _info "Project packages installed."
}
