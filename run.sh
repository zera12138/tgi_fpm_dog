#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BPX_PYTHON="${BPX_PYTHON:-/home/odroid/anaconda3/envs/bpx_supermimic/bin/python}"
export PYTHONPATH="$ROOT/tgi/src:$ROOT/fpm/src"
unset LD_LIBRARY_PATH
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/bpx-tgi-mpl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/bpx-tgi-cache}"
cd "$ROOT"
command="${1:-check}"
if [[ $# -gt 0 ]]; then shift; fi
case "$command" in
  check-moe) exec "$BPX_PYTHON" scripts/check_moe.py "$@" ;;
  check) exec "$BPX_PYTHON" scripts/check_core.py "$@" ;;
  preflight) exec "$BPX_PYTHON" scripts/preflight.py "$@" ;;
  newton) exec "$BPX_PYTHON" scripts/check_newton.py "$@" ;;
  assets) "$BPX_PYTHON" scripts/build_assets.py; exec "$BPX_PYTHON" scripts/convert_bpx_usd.py "$@" ;;
  train) exec "$BPX_PYTHON" tgi/scripts/train.py "$@" ;;
  export) exec "$BPX_PYTHON" tgi/scripts/export.py "$@" ;;
  evaluate) exec "$BPX_PYTHON" scripts/evaluate_bpx.py "$@" ;;
  check-export) exec "$BPX_PYTHON" scripts/check_export.py "$@" ;;
  *) echo "Usage: ./run.sh {check|check-moe|preflight|newton|assets|train|export|evaluate|check-export} [args]" >&2; exit 2 ;;
esac
