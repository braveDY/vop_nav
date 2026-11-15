#!/usr/bin/env bash

_process_exec() {
    _remote_exec "$(_quote_args "${REMOTE_PYTHON}" scripts/utils/tasks.py "$@")"
}
