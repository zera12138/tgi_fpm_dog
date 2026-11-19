#!/usr/bin/env bash
# Preserve the first pilot and continue its default-size network in a new log directory.
set -euo pipefail
cd /home/odroid/bpx-tgi-fpm
CHECKPOINT="${1:-/home/odroid/bpx-tgi-fpm/logs/bpx_resume_20261107_173919/last.pt}"
[[ -f "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 1; }
RUN_DIR="logs/bpx_resume_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
echo "Resuming pilot to total iteration 10000; retaining the original run. Log: $RUN_DIR"
./run.sh train \
  --resume "$CHECKPOINT" --iterations 10000 \
  --num_envs 128 --wm_batch 256 --pi_batch 128 --fpm_batch 512 \
  --warmup_steps 200 --resume_wm_only 50 \
  --compile 0 --cudagraphs 0 --async_collect 0 --amp none \
  --vram_gb 10 --log_every 25 --ckpt_every 100 \
  --log_dir "$RUN_DIR" 2>&1 | tee "$RUN_DIR/console.log"
