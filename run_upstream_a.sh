#!/usr/bin/env bash
# 上游配方（4096 envs / 40k 迭代 / compile+cudagraphs+async）+ 我们的 MoE + 任务记忆
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
log_dir="$ROOT/logs/upstreamA_$(date +%Y%m%d_%H%M%S)_$$"
args=(train --iterations 40000 --log_dir "$log_dir"
      --policy moe --history_length 5 --mem 1 --w_phase 0.5 --w_air 0.2
      --gait_period 0.6 --stance_fraction 0.56 --gait_cmd_threshold 0.1
      --num_envs 4096 --vram_gb 11.0
      --wm_batches 8 --wm_batch 256 --buffer_steps 200 --utd 0.4 --horizon 32 --pi_batch 512
      --wm_only_iters 300 --fpm_bootstrap 1500 --async_collect 1 --async_after 1600
      --pit_step 0.05 --ckpt_every 1000 --log_every 50
      --amp none --compile 1 --cudagraphs 1)
if [[ "${1:-}" == "--dry-run" ]]; then printf '%q ' "$ROOT/run.sh" "${args[@]}"; printf '\n'; exit 0; fi
mkdir -p "$log_dir"
printf 'Training logs: %s\n' "$log_dir"
"$ROOT/run.sh" "${args[@]}" 2>&1 | tee "$log_dir/console.log"
