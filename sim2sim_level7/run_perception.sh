#!/usr/bin/env bash
# Three scenarios x two mount pitches. Policy always drives on the truth map, so the
# trajectory is identical between pitches; only the perception maps differ.
set -u
export MUJOCO_GL=egl MPLCONFIGDIR=/tmp/mpl
PY=/home/odroid/anaconda3/envs/rl_lab/bin/python
cd "$(dirname "$0")"
run() {  # name spawn seconds vx panels
  echo "=== $1 spawn=$2 ${3}s vx=$4 ==="
  "$PY" verify_perception.py --pitch 15,30 --spawn "$2" --seconds "$3" \
      --command "$4" 0 0 --panels "$5" --out "./perception_out/$1" 2>&1 | grep -Ev 'EGL|OpenGL|Traceback|^\s+File|self\.|raise |cArguments|<OpenGL|err =|baseOperation|result = 0|^\)|^\t|telemetry'
}
run stairs_up 1 12 0.3 5,8,11
run stairs_down 2 8 0.25 2,5,7
run backward 1 8 -0.2 2,5,7
echo "ALL DONE"
