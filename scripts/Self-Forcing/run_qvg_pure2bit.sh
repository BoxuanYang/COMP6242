#!/bin/bash

prompts_path=assets/t2v.txt
ckpt_id=official
local_attn_size=180
num_output_frames=180
ckpt_path=ckpts/Self-Forcing/self_forcing_dmd.pt

#########################################################
# Quantization Configuration
#########################################################
quant_type="triton-nstages-kmeans-int2"
# quant_type="triton-nstages-kmeans-int4"
cache_num_k_centroids=256
cache_num_v_centroids=256
kmeans_max_iters=2
quant_block_size=64
num_prq_stages=1

mixed_bit_enabled=false
mixed_schedule="static_global"
mixed_1bit_ratio=0.25
mixed_low_quant_type="triton-nstages-kmeans-int1"
mixed_high_quant_type="triton-nstages-kmeans-int2"

if [ "$mixed_bit_enabled" = true ]; then
  quant_dir=mixed_static_int1_${mixed_1bit_ratio}_int2_${quant_block_size}/kc_${cache_num_k_centroids}_vc_${cache_num_v_centroids}_nstages_${num_prq_stages}
else
  quant_dir=${quant_type}_${quant_block_size}/kc_${cache_num_k_centroids}_vc_${cache_num_v_centroids}_nstages_${num_prq_stages}
fi
output_folder=results/selfforcing/${quant_dir}

echo "Running inference with checkpoint $ckpt_path and prompts from $prompts_path"
echo "Output will be saved to $output_folder"

export PYTHONPATH=experiments/Self-Forcing:.

DUMP_KV_LEVEL=0 torchrun --nproc_per_node=1 --standalone experiments/Self-Forcing/inference.py \
  --config_path experiments/Self-Forcing/configs/self_forcing_dmd.yaml \
  --checkpoint_path $ckpt_path \
  --data_path $prompts_path \
  --output_folder $output_folder \
  --num_samples 1 \
  --seed 0 \
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
  --mixed_high_quant_type $mixed_high_quant_type
