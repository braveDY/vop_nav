#!/usr/bin/env bash

_project_task_completion() {
    local current="${COMP_WORDS[COMP_CWORD]}"
    if [[ "${COMP_WORDS[0]}" == stop || "${COMP_WORDS[0]}" == log ]]; then
        COMPREPLY=( $(compgen -W 'train collect train_vop compare play evaluate' -- "$current") )
    elif [[ "${COMP_WORDS[COMP_CWORD-1]}" == --config || "${COMP_WORDS[COMP_CWORD-1]}" == --run-dir ]]; then
        compopt -o filenames
        COMPREPLY=( $(compgen -f -- "$current") )
    else
        local options='--config --device --run-dir --name --log --help'
        case "${COMP_WORDS[0]}" in
            train) options+=' --num-envs --max-iterations';;
            collect|play|evaluate) options+=' --num-envs';;
            train_vop) options+=' --epochs';;
        esac
        COMPREPLY=( $(compgen -W "$options" -- "$current") )
    fi
}
complete -F _project_task_completion train collect train_vop compare play evaluate stop log
