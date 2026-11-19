#!/usr/bin/env bash
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
unset LD_LIBRARY_PATH
export PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_AUDIODRIVER=dummy
export MPLCONFIGDIR=/tmp/bpx-gamepad-mpl
exec /home/odroid/anaconda3/envs/rl_lab/bin/python "$task_dir/play.py" "$@"
