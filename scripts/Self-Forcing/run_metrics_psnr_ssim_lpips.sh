#!/bin/bash

set -euo pipefail

pred_folder="${PRED_FOLDER:-${1:-}}"
ref_folder="${REF_FOLDER:-results/selfforcing/bf16}"
skip_frames="${SKIP_FRAMES:-0}"
device="${DEVICE:-cuda}"

if [[ -z "${pred_folder}" ]]; then
  echo "Usage: PRED_FOLDER=<pred_folder> bash scripts/Self-Forcing/run_metrics_psnr_ssim_lpips.sh"
  echo "   or: bash scripts/Self-Forcing/run_metrics_psnr_ssim_lpips.sh <pred_folder>"
  exit 1
fi

output_json="${OUTPUT_JSON:-${pred_folder}/metrics_psnr_ssim_lpips_summary.json}"
output_jsonl="${OUTPUT_JSONL:-${pred_folder}/metrics_psnr_ssim_lpips_per_video.jsonl}"

echo "Evaluating predicted videos: ${pred_folder}"
echo "Reference videos: ${ref_folder}"

export PYTHONPATH=experiments/Self-Forcing:.

python experiments/Self-Forcing/eval_psnr_ssim_lpips.py \
  --pred_folder "${pred_folder}" \
  --ref_folder "${ref_folder}" \
  --output_json "${output_json}" \
  --output_jsonl "${output_jsonl}" \
  --skip_frames "${skip_frames}" \
  --device "${device}"

echo "Metrics summary: ${output_json}"
echo "Per-video metrics: ${output_jsonl}"
