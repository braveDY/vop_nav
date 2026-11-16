#!/usr/bin/env bash

_project_checkpoint_completion() {
    local current="$1"
    if [[ "${CONNECTION_MODE:-ssh}" == local ]]; then
        compopt -o filenames 2>/dev/null || true
        COMPREPLY=( $(compgen -f -- "$current") )
        return
    fi
    # Remote candidates must show their full paths, including the run directory.
    compopt +o filenames 2>/dev/null || true
    local key="${REMOTE_HOST}|${REMOTE_DIR}|${REMOTE_RUNTIME}|${CONTAINER_DIR:-}"
    local now=${SECONDS} listing candidate prefix="" path
    if [[ "${_CHECKPOINT_CACHE_KEY:-}" != "$key" || $((now - ${_CHECKPOINT_CACHE_TIME:--15})) -ge 15 ]]; then
        local script='cd "$1" || exit; for folder in logs pretrained checkpoints; do
            if [[ -d "$folder" ]]; then
                find "$folder" -type f -name "*.pt" ! -name "*.curriculum.pt" -print
            fi
        done | sort -r'
        listing=$(ssh -o BatchMode=yes -o ConnectTimeout=3 -o ConnectionAttempts=1 \
            "${REMOTE_HOST}" "$(_quote_args bash -c "$script" bash "${REMOTE_DIR}")" 2>/dev/null) || return
        _CHECKPOINT_CACHE_FILES=()
        while IFS= read -r path; do
            [[ -n "$path" ]] && _CHECKPOINT_CACHE_FILES+=("$path")
        done <<< "$listing"
        _CHECKPOINT_CACHE_KEY="$key"
        _CHECKPOINT_CACHE_TIME=$now
    fi
    if [[ "$current" == "${REMOTE_DIR}/"* ]]; then
        prefix="${REMOTE_DIR}/"
    elif [[ "${REMOTE_RUNTIME}" == docker && "$current" == "${CONTAINER_DIR}/"* ]]; then
        prefix="${CONTAINER_DIR}/"
    elif [[ "$current" == ./* ]]; then
        prefix='./'
    fi
    for path in "${_CHECKPOINT_CACHE_FILES[@]}"; do
        candidate="${prefix}${path}"
        [[ "$candidate" == "$current"* ]] && COMPREPLY+=("$candidate")
    done
    return 0
}

_project_task_completion() {
    local current="${COMP_WORDS[COMP_CWORD]}"
    local previous="${COMP_WORDS[COMP_CWORD-1]}"
    local command="${COMP_WORDS[0]}"
    COMPREPLY=()
    if [[ "$command" == stop || "$command" == log ]]; then
        local kinds='train collect train_vop compare play evaluate'
        [[ "$command" == stop ]] && kinds+=' tb'
        COMPREPLY=( $(compgen -W "$kinds" -- "$current") )
        return
    fi
    case "$previous" in
        --task)
            local tasks='Unitree-Go2-VOP-BasicNav Unitree-Go2-VOP-Nav'
            [[ "$command" == play ]] && tasks='Unitree-Go2-VOP-BasicNav-Play Unitree-Go2-VOP-Nav-Play'
            COMPREPLY=( $(compgen -W "$tasks" -- "$current") ); return;;
        --scenario|--scenarios)
            COMPREPLY=( $(compgen -W 'Training Forest Office Square Slow' -- "$current") ); return;;
        --level|--levels)
            COMPREPLY=( $(compgen -W '0 1 2 3 4 5 6 7 8 9' -- "$current") ); return;;
        --checkpoint|--vop_checkpoint)
            _project_checkpoint_completion "$current"; return;;
        --config|--run-dir|--run_dir)
            compopt -o filenames 2>/dev/null || true
            COMPREPLY=( $(compgen -f -- "$current") ); return;;
    esac
    local options='--config --device --run-dir --name --log --help'
    case "$command" in
        train|play|evaluate)
            options='--task --checkpoint --vop_checkpoint --device --num_envs --run_dir --name --kit_args --log --help'
            case "$command" in
                train) options+=' --seed --max_iterations --deadline --chunk_iterations --low_memory';;
                play) options+=' --seed --scenario --level --steps --memory_reserve_gib';;
                evaluate) options+=' --scenarios --levels --seeds --episodes_per_env --memory_reserve_gib';;
            esac;;
        collect) options+=' --num-envs';;
        train_vop) options+=' --epochs';;
    esac
    COMPREPLY=( $(compgen -W "$options" -- "$current") )
}

complete -F _project_task_completion train collect train_vop compare play evaluate stop log

_env_source_complete() {
    COMPREPLY=()
    local current="${COMP_WORDS[COMP_CWORD]}"
    local previous="${COMP_WORDS[COMP_CWORD-1]}"
    local script="${COMP_WORDS[1]:-}"
    if [[ "$script" =~ (^|/)env\.sh$ && $COMP_CWORD -ge 2 ]]; then
        if [[ "$previous" == --server || "$previous" == -s ]]; then
            local directory profile server
            directory="$(dirname "$script")/scripts/env/servers"
            while IFS= read -r profile; do
                server="${profile##*/}"
                server="${server%.env}"
                [[ "$server" == template ]] || COMPREPLY+=("$server")
            done < <(compgen -G "${directory}/${current}*.env" | sort)
        elif [[ $COMP_CWORD -eq 2 ]]; then
            COMPREPLY=( $(compgen -W '--server' -- "$current") )
        fi
        return 0
    fi
    compopt -o default 2>/dev/null || true
}
complete -F _env_source_complete -o default source .
