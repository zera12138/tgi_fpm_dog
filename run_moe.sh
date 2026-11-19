#!/usr/bin/env bash
# 从10000步MLP基线接入任务记忆+MoE；全部输出放在独立目录。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
baseline="$ROOT/logs/bpx_resume_20261110_131910/last.pt"
log_dir="$ROOT/logs/moe_$(date +%Y%m%d_%H%M%S)_$$"
args=(train --resume "$baseline" --iterations 13000 --log_dir "$log_dir"
      --policy moe --history_length 5 --mem 1 --w_phase 0.5 --w_air 0.2
      --gait_period 0.6 --stance_fraction 0.56 --gait_cmd_threshold 0.1
      --num_envs 128 --wm_hidden 1024 1024 1024 --wm_batch 256 --wm_batches 8
      --pi_batch 128 --horizon 32 --fpm_batch 512 --fpm_steps 8
      --warmup_steps 200 --resume_wm_only 50 --ckpt_every 100 --log_every 25
      --amp none --compile 0 --cudagraphs 0 --async_collect 0 --vram_gb 10)
if [[ "${1:-}" == "--dry-run" ]]; then
    printf '%q ' "$ROOT/run.sh" "${args[@]}"
    printf '\n'
    exit 0
fi
if [[ $# -gt 0 ]]; then
    echo "Usage: bash run_moe.sh [--dry-run]" >&2
    exit 2
fi
mkdir -p "$log_dir"
printf 'Training logs: %s\n' "$log_dir"
"$ROOT/run.sh" "${args[@]}" 2>&1 | tee "$log_dir/console.log"
