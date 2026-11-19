#!/usr/bin/env bash
# 补下对话里提到、但本地文献库缺失的 3 篇 PDF。
# 用法：
#   HTTPS_PROXY=http://127.0.0.1:7890 ./scripts/fetch_papers.sh      # 走代理
#   ./scripts/fetch_papers.sh                                        # 直连（需能访问 arxiv.org）
# 目标目录可用 LIT_ROOT 覆盖，默认 /home/odroid/文献
set -euo pipefail

LIT_ROOT="${LIT_ROOT:-/home/odroid/文献}"
DEST="$LIT_ROOT/02_四足复杂地形与巡检/06_落足预测与高程建图"
PROXY="${HTTPS_PROXY:-${https_proxy:-}}"

# arXiv ID | 输出文件名
PAPERS=(
  "2605.00261|2026_Task_Conditioned_Uncertainty_Costmaps.pdf"
  "2110.00891|2022_Vision_aided_Dynamic_Quadrupedal_Locomotion.pdf"
  "2204.12876|2022_Elevation_Mapping_for_Locomotion_and_Navigation_GPU.pdf"
)

mkdir -p "$DEST"

curl_opts=(-fL --retry 3 --retry-delay 2 --connect-timeout 20 -A "Mozilla/5.0")
[[ -n "$PROXY" ]] && curl_opts+=(-x "$PROXY")

fail=0
for entry in "${PAPERS[@]}"; do
  id="${entry%%|*}"
  name="${entry##*|}"
  out="$DEST/$name"
  if [[ -s "$out" ]] && head -c 5 "$out" | grep -q '%PDF-'; then
    echo "skip (exists): $out"
    continue
  fi
  echo "fetch arXiv:$id -> $out"
  if curl "${curl_opts[@]}" -o "$out.tmp" "https://arxiv.org/pdf/$id"; then
    if head -c 5 "$out.tmp" | grep -q '%PDF-'; then
      mv "$out.tmp" "$out"
      echo "  ok $(stat -c%s "$out") bytes"
    else
      echo "  ERROR: 返回内容不是 PDF" >&2
      rm -f "$out.tmp"
      fail=1
    fi
  else
    echo "  ERROR: 下载失败（网络/代理不通）" >&2
    rm -f "$out.tmp"
    fail=1
  fi
done

# 校验元数据标题
if command -v pdfinfo >/dev/null; then
  echo "--- titles ---"
  for entry in "${PAPERS[@]}"; do
    f="$DEST/${entry##*|}"
    [[ -s "$f" ]] && { printf '%s\n' "${f##*/}"; pdfinfo "$f" 2>/dev/null | sed -n 's/^Title: *//p'; }
  done
fi

exit "$fail"
