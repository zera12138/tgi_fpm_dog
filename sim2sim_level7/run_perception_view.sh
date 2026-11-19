#!/usr/bin/env bash
# Interactive perception view. Mirrors run.sh: real window by default, offscreen EGL for --headless.
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
unset LD_LIBRARY_PATH
export PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_AUDIODRIVER=dummy
export MPLCONFIGDIR=/tmp/bpx-perception-mpl
for a in "$@"; do [[ "$a" == "--headless" ]] && export MUJOCO_GL=egl; done
exec /home/odroid/anaconda3/envs/rl_lab/bin/python "$task_dir/play_perception.py" "$@"
