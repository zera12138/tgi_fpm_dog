#!/usr/bin/env bash
# 带任务记忆与相位/腾空奖励的续训（从 10000 步检查点起）
set -euo pipefail
cd /home/odroid/bpx-tgi-fpm
CKPT="${1:-logs/bpx_resume_20261110_131910/last.pt}"
RUN_DIR="logs/mem_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
echo "Resuming $CKPT with task memory to 13000 iterations. Log: $RUN_DIR"
./run.sh train \
  --resume "$CKPT" --iterations 13000 \
  --num_envs 128 --wm_batch 256 --pi_batch 128 --fpm_batch 512 \
  --warmup_steps 200 --resume_wm_only 50 \
  --compile 0 --cudagraphs 0 --async_collect 0 --amp none \
  --vram_gb 10 --log_every 25 --ckpt_every 100 \
  --mem 1 --w_phase 0.5 --w_air 0.2 \
  --log_dir "$RUN_DIR" 2>&1 | tee "$RUN_DIR/console.log"
