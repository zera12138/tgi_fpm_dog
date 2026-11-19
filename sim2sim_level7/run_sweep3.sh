#!/usr/bin/env bash
export MUJOCO_GL=egl MPLCONFIGDIR=/tmp/mpl
PY=/home/odroid/anaconda3/envs/rl_lab/bin/python
cd "$(dirname "$0")"
{ for sc in down up; do
    for v in -2.0 -1.0 0.0 1.0 2.0; do echo "$sc pitch $v"; done
  done; } | xargs -P 8 -n 3 bash -c "$PY d_sweep.py --scenario \$0 --factor \$1 --value=\$2" > sweep3.log 2>&1
echo SWEEP3_DONE
