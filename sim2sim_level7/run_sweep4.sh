#!/usr/bin/env bash
export MUJOCO_GL=egl MPLCONFIGDIR=/tmp/mpl
PY=/home/odroid/anaconda3/envs/rl_lab/bin/python
cd "$(dirname "$0")"
{ for sc in down up; do
    for v in 0.40 0.80 1.50; do echo "$sc latency $v"; done
    for v in 2.0 3.0; do echo "$sc drop $v"; done
    for v in 0.15 0.20; do echo "$sc dz $v"; done
    for v in -6.0 -4.0 4.0 6.0; do echo "$sc pitch $v"; done
    for v in 0.06 0.08 0.12; do echo "$sc dx $v"; done
  done; } | xargs -P 8 -n 3 bash -c "$PY d_sweep.py --scenario \$0 --factor \$1 --value=\$2" > sweep4.log 2>&1
echo SWEEP4_DONE
