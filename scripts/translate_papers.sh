#!/usr/bin/env bash
# 用 DeepSeek 批量翻译指定论文（pdf2zh + babeldoc），输出 中文版 / 中英对照版 两个 PDF。
# 用法：./scripts/translate_papers.sh [--only NAME]
set -euo pipefail

PDF2ZH=/home/odroid/anaconda3/envs/pdf2zh/bin/pdf2zh
ROOT=/home/odroid/bpx-tgi-fpm
PROMPT="$ROOT/scripts/prompt_zh.txt"
STAGE="$ROOT/.zhout"
LIT=/home/odroid/文献

# 输入文件 | 输出文件名前缀
ITEMS=(
  "$LIT/2610.29612v1.pdf|2610.29612v1"
  "$LIT/02_四足复杂地形与巡检/05_空间记忆与视觉落脚/2610.37203v1.pdf|2610.37203v1"
  "$LIT/02_四足复杂地形与巡检/05_空间记忆与视觉落脚/2610.37806v1.pdf|2610.37806v1"
  "$LIT/02_四足复杂地形与巡检/06_落足预测与高程建图/2026_Task_Conditioned_Uncertainty_Costmaps.pdf|2026_Task_Conditioned_Uncertainty_Costmaps"
  "$LIT/02_四足复杂地形与巡检/06_落足预测与高程建图/2022_Vision_aided_Dynamic_Quadrupedal_Locomotion.pdf|2022_Vision_aided_Dynamic_Quadrupedal_Locomotion"
  "$LIT/02_四足复杂地形与巡检/06_落足预测与高程建图/2022_Elevation_Mapping_for_Locomotion_and_Navigation_GPU.pdf|2022_Elevation_Mapping_for_Locomotion_and_Navigation_GPU"
)

ONLY=""
[[ "${1:-}" == "--only" ]] && ONLY="${2:-}"

mkdir -p "$STAGE"
for item in "${ITEMS[@]}"; do
  src="${item%%|*}"; prefix="${item##*|}"
  [[ -n "$ONLY" && "$prefix" != *"$ONLY"* ]] && continue
  dir="$(dirname "$src")"
  mono="$dir/${prefix}_中文版.pdf"
  dual="$dir/${prefix}_中英对照版.pdf"
  if [[ -s "$mono" && -s "$dual" ]]; then
    echo "SKIP (exists): $prefix"
    continue
  fi
  echo "=== $(date +%H:%M:%S) translate: ${src##*/}"
  "$PDF2ZH" -s deepseek -li en -lo zh -t 8 --prompt "$PROMPT" -o "$STAGE" "$src" 2>&1 \
    | grep -v -E '^.*INFO.*use font|it/s\]$' | tail -3 || true
  base="$(basename "$src" .pdf)"
  if [[ -s "$STAGE/$base-mono.pdf" ]]; then
    mv -f "$STAGE/$base-mono.pdf" "$mono"
    mv -f "$STAGE/$base-dual.pdf" "$dual"
    echo "  -> ${prefix}_中文版.pdf ($(du -h "$mono" | cut -f1)) / _中英对照版.pdf ($(du -h "$dual" | cut -f1))"
  else
    echo "  !! FAILED: $src" >&2
  fi
done
echo "=== done $(date +%H:%M:%S)"
