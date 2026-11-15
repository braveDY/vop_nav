#!/usr/bin/env bash

_info()  { echo -e "\033[1;32m[INFO]\033[0m $*"; }
_warn()  { echo -e "\033[1;33m[WARN]\033[0m $*"; }
_err()   { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }

_quote_args() { printf '%q ' "$@"; }
