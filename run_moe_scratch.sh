#!/usr/bin/env bash
# 从零开始（不续训）训练 MoE + task memory，共 10000 迭代。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
log_dir="$ROOT/logs/moe_scratch_$(date +%Y%m%d_%H%M%S)_$$"
args=(train --iterations 10000 --log_dir "$log_dir"
      --policy moe --history_length 5 --mem 1 --w_phase 0.5 --w_air 0.2
      --gait_period 0.6 --stance_fraction 0.56 --gait_cmd_threshold 0.1
      --num_envs 128 --wm_hidden 1024 1024 1024 --wm_batch 256 --wm_batches 8
      --pi_batch 128 --horizon 32 --fpm_batch 512 --fpm_steps 8
      --warmup_steps 200 --ckpt_every 100 --log_every 25
      --amp none --compile 0 --cudagraphs 0 --async_collect 0 --vram_gb 10)
if [[ "${1:-}" == "--dry-run" ]]; then
    printf '%q ' "$ROOT/run.sh" "${args[@]}"; printf '\n'; exit 0
fi
mkdir -p "$log_dir"
printf 'Training logs: %s\n' "$log_dir"
"$ROOT/run.sh" "${args[@]}" 2>&1 | tee "$log_dir/console.log"
