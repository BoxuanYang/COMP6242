#!/bin/bash

set -euo pipefail

prompts_path="${PROMPTS_PATH:-assets/t2v.txt}"
ckpt_id="${CKPT_ID:-official}"
local_attn_size="${LOCAL_ATTN_SIZE:-180}"
num_output_frames="${NUM_OUTPUT_FRAMES:-180}"
ckpt_path="${CKPT_PATH:-ckpts/Self-Forcing/self_forcing_dmd.pt}"

#########################################################
# Quantization Configuration
#########################################################
quant_type="${QUANT_TYPE:-triton-nstages-kmeans-int2}"
# quant_type="triton-nstages-kmeans-int4"
cache_num_k_centroids="${CACHE_NUM_K_CENTROIDS:-256}"
cache_num_v_centroids="${CACHE_NUM_V_CENTROIDS:-256}"
kmeans_max_iters="${KMEANS_MAX_ITERS:-2}"
quant_block_size="${QUANT_BLOCK_SIZE:-64}"
num_prq_stages="${NUM_PRQ_STAGES:-1}"

mixed_bit_enabled="${MIXED_BIT_ENABLED:-true}"
mixed_schedule="${MIXED_SCHEDULE:-static_global}"
mixed_1bit_ratio="${MIXED_1BIT_RATIO:-0.25}"
mixed_low_quant_type="${MIXED_LOW_QUANT_TYPE:-triton-nstages-kmeans-int1}"
mixed_high_quant_type="${MIXED_HIGH_QUANT_TYPE:-triton-nstages-kmeans-int2}"
quant_factor="${QUANT_FACTOR:-1}"
centroid_caching_enabled="${CENTROID_CACHING_ENABLED:-false}"

# Alias for easier experiment sweeps.
if [[ -n "${MIXED_RATIO:-}" ]]; then
  mixed_1bit_ratio="${MIXED_RATIO}"
fi

extract_bit_tag() {
  local quant="$1"
  local bit
  bit=$(echo "$quant" | sed -n 's/.*int\([0-9][0-9]*\).*/int\1/p')
  if [[ -z "$bit" ]]; then
    bit=$(echo "$quant" | sed 's/[^a-zA-Z0-9._-]/_/g')
  fi
  echo "$bit"
}

low_tag=$(extract_bit_tag "$mixed_low_quant_type")
high_tag=$(extract_bit_tag "$mixed_high_quant_type")

if [ "$mixed_bit_enabled" = true ]; then
  quant_dir=mixed_${mixed_schedule}_low${low_tag}_${mixed_1bit_ratio}_high${high_tag}_${quant_block_size}/kc_${cache_num_k_centroids}_vc_${cache_num_v_centroids}_nstages_${num_prq_stages}
else
  quant_dir=${quant_type}_${quant_block_size}/kc_${cache_num_k_centroids}_vc_${cache_num_v_centroids}_nstages_${num_prq_stages}
fi

if [[ -n "${OUTPUT_FOLDER:-}" ]]; then
  output_folder="${OUTPUT_FOLDER}"
else
  output_folder="results/selfforcing/${quant_dir}"
fi

dump_kv_level="${DUMP_KV_LEVEL:-0}"

echo "Running inference with checkpoint $ckpt_path and prompts from $prompts_path"
echo "Output will be saved to $output_folder"
echo "Mixed-bit config: enabled=$mixed_bit_enabled schedule=$mixed_schedule ratio=$mixed_1bit_ratio low=$mixed_low_quant_type high=$mixed_high_quant_type"
echo "Quantization control: quant_factor=$quant_factor centroid_caching_enabled=$centroid_caching_enabled"

export PYTHONPATH=experiments/Self-Forcing:.

DUMP_KV_LEVEL="$dump_kv_level" torchrun --nproc_per_node=1 --standalone experiments/Self-Forcing/inference.py \
  --config_path experiments/Self-Forcing/configs/self_forcing_dmd.yaml \
  --checkpoint_path $ckpt_path \
  --data_path $prompts_path \
  --output_folder $output_folder \
  --num_samples 1 \
  --num_output_frames $num_output_frames \
  --local_attn_size $local_attn_size \
  --use_ema \
  --save_with_index \
  --quant_type $quant_type \
  --cache_num_k_centroids $cache_num_k_centroids \
  --cache_num_v_centroids $cache_num_v_centroids \
  --kmeans_max_iters $kmeans_max_iters \
  --quant_block_size $quant_block_size \
  --num_prq_stages $num_prq_stages \
  --mixed_bit_enabled $mixed_bit_enabled \
  --mixed_schedule $mixed_schedule \
  --mixed_1bit_ratio $mixed_1bit_ratio \
  --mixed_low_quant_type $mixed_low_quant_type \
  --mixed_high_quant_type $mixed_high_quant_type \
  --quant_factor $quant_factor \
  --centroid_caching_enabled $centroid_caching_enabled
