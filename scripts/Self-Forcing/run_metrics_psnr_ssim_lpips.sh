#!/bin/bash

set -euo pipefail

pred_folder="results/selfforcing/mixed_static_int1_0.25_int2_64/kc_256_vc_256_nstages_1"
ref_folder="results/selfforcing/bf16"
output_json="${pred_folder}/metrics_psnr_ssim_lpips_summary.json"
output_jsonl="${pred_folder}/metrics_psnr_ssim_lpips_per_video.jsonl"

echo "Evaluating predicted videos: ${pred_folder}"
echo "Reference videos: ${ref_folder}"

export PYTHONPATH=experiments/Self-Forcing:.

python experiments/Self-Forcing/eval_psnr_ssim_lpips.py \
  --pred_folder "${pred_folder}" \
  --ref_folder "${ref_folder}" \
  --output_json "${output_json}" \
  --output_jsonl "${output_jsonl}" \
  --skip_frames 0 \
  --device cuda

echo "Metrics summary: ${output_json}"
echo "Per-video metrics: ${output_jsonl}"
