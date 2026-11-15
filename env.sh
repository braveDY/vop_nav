DEFAULT_SERVER="local"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export PROJECT_ROOT

_server_target=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --server|-s)
            _server_target="$2"
            shift 2
            ;;
        --server=*)
            _server_target="${1#*=}"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

export SERVER="${_server_target:-${DEFAULT_SERVER}}"
unset _server_target

for _env_module in "${PROJECT_ROOT}/scripts/env/modules/"[0-9]*.sh; do
    if [[ -r "${_env_module}" ]]; then
        source "${_env_module}" || return 1 2>/dev/null || exit 1
    fi
done
unset _env_module
