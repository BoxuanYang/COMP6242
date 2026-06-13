# Self-Forcing Experiment Runbook

## -1) Environment and Model Preparation

If this is the first time running Self-Forcing on this machine, complete the following steps first:

### -1.1 Enter the project directory

```bash
cd Quant-VideoGen
```

### -1.2 Install Self-Forcing dependencies, if not already installed

```bash
uv pip install -e ".[selfforcing]"
```

### -1.3 Configure Hugging Face access, if not already logged in

```bash
hf auth login
```

### -1.4 Download the Self-Forcing model and checkpoint

```bash
bash scripts/Self-Forcing/download_models.sh
```

This script will download:

- `ckpts/Self-Forcing/Wan2.1-T2V-1.3B`
- `ckpts/Self-Forcing/self_forcing_dmd.pt`

### -1.5 Quickly verify the download

```bash
ls -l ckpts/Self-Forcing/self_forcing_dmd.pt
ls -ld ckpts/Self-Forcing/Wan2.1-T2V-1.3B
```

If both paths exist, you can run the subsequent experiments directly. You do not need to download them again every time.

---

After the environment and models are fully configured, the following is the complete runbook for the Self-Forcing experiments:

- First run the BF16 baseline.
- Then run four mixed-bit groups:
  - 4bit-2bit: ratio 0.25 / 0.50
  - 2bit-1bit: ratio 0.25 / 0.50
- Compare each group against BF16 and calculate PSNR / SSIM / LPIPS.

This runbook assumes:

- The quant factor is fixed at 1, meaning quantization is triggered for every chunk.
- Centroid caching is exposed as a switch. The default examples below keep it disabled first, so the setup stays aligned with the existing baseline.

---

## 0) Parameter Modification Guide

Recommended priority:

1. **Prefer temporary overrides through environment variables**. This is recommended because it is the least likely to break the default scripts.
2. Only edit scripts or configuration files when you want the new values to become the default for future runs.

### 0.1 Generation experiment parameters, recommended: command-line environment variable overrides

Entry script:

```text
scripts/Self-Forcing/run_qvg.sh
```

Parameters that can be overridden directly before the command:

- Mixed strategy: `MIXED_RATIO`, `MIXED_LOW_QUANT_TYPE`, `MIXED_HIGH_QUANT_TYPE`
- Quantization trigger and caching: `QUANT_FACTOR`, `CENTROID_CACHING_ENABLED`
- Output directory: `OUTPUT_FOLDER`
- Video length and attention window: `NUM_OUTPUT_FRAMES`, `LOCAL_ATTN_SIZE`
- Quantization details: `QUANT_BLOCK_SIZE`, `NUM_PRQ_STAGES`, `CACHE_NUM_K_CENTROIDS`, `CACHE_NUM_V_CENTROIDS`, `KMEANS_MAX_ITERS`
- Model and input: `CKPT_PATH`, `PROMPTS_PATH`

Example, using temporary overrides without modifying files:

```bash
MIXED_RATIO=0.50 \
MIXED_LOW_QUANT_TYPE=triton-nstages-kmeans-int4 \
MIXED_HIGH_QUANT_TYPE=triton-nstages-kmeans-int2 \
QUANT_FACTOR=1 \
CENTROID_CACHING_ENABLED=true \
NUM_OUTPUT_FRAMES=180 \
LOCAL_ATTN_SIZE=180 \
uv run bash scripts/Self-Forcing/run_qvg.sh
```

If you want to modify the default values instead of passing environment variables every time, edit:

```text
scripts/Self-Forcing/run_qvg.sh
```

Modify the default-value lines inside the script, which are in the form `${VAR:-default}`.

### 0.2 Self-Forcing model / inference baseline configuration, requires YAML modification

File:

```text
experiments/Self-Forcing/configs/self_forcing_dmd.yaml
```

Common items to modify:

- `num_frame_per_block`
- `denoising_step_list`
- `guidance_scale`
- Other native Self-Forcing configuration items

Note: Changes of this type alter the experiment protocol itself. They are no longer just changes to the mixed-bit policy.

### 0.3 Evaluation parameters, PSNR / SSIM / LPIPS

Entry script:

```text
scripts/Self-Forcing/run_metrics_psnr_ssim_lpips.sh
```

Overridable parameters:

- `PRED_FOLDER`, the result directory to evaluate
- `REF_FOLDER`, the BF16 reference directory
- `SKIP_FRAMES`
- `DEVICE`
- `OUTPUT_JSON`, `OUTPUT_JSONL`

Underlying implementation script, usually no need to modify:

```text
experiments/Self-Forcing/eval_psnr_ssim_lpips.py
```

### 0.4 Advanced code-level switches, usually no need to modify

If you need to extend the algorithm implementation rather than only tune experiment parameters, the main locations are:

- Parameter injection: `experiments/Self-Forcing/inference.py`
- Mixed-bit scheduling and triggering: `experiments/Self-Forcing/pipeline/causal_inference.py`
- Compression / decompression and centroid warm-start: `quant_videogen/compress.py`, `quant_videogen/functions.py`, `quant_videogen/real/prq.py`

---

## 1) Enter the project and create the log directory

```bash
cd Quant-VideoGen
mkdir -p logs
```

Optional: quickly confirm the parameter entry points in the script

```bash
grep -nE "MIXED_RATIO|MIXED_LOW_QUANT_TYPE|MIXED_HIGH_QUANT_TYPE|QUANT_FACTOR|CENTROID_CACHING_ENABLED|OUTPUT_FOLDER" scripts/Self-Forcing/run_qvg.sh
```

---

## 2) First run the BF16 baseline

```bash
bash scripts/Self-Forcing/run_bf16.sh 2>&1 | tee logs/self_forcing_bf16.log
```

BF16 video output directory:

```text
results/selfforcing/bf16
```

---

## 3) Run three mixed-bit generation groups

Notes:

- You do not need to manually edit the script files again. Override parameters directly through environment variables.
- The quant factor is fixed at 1: `QUANT_FACTOR=1`
- Centroid caching can be enabled or disabled: `CENTROID_CACHING_ENABLED=true/false`

### 3.1 Pure 2bit quantization baseline

```bash
MIXED_BIT_ENABLED=false \
QUANT_TYPE=triton-nstages-kmeans-int2 \
time uv run bash scripts/Self-Forcing/run_qvg.sh 2>&1 | tee logs/self_forcing_qvg_int2.log
```

### 3.2 1bit-2bit, ratio=0.25

```bash
MIXED_RATIO=0.25 \
MIXED_BIT_ENABLED=true \
MIXED_LOW_QUANT_TYPE=triton-nstages-kmeans-int1 \
MIXED_HIGH_QUANT_TYPE=triton-nstages-kmeans-int2 \
QUANT_FACTOR=1 \
CENTROID_CACHING_ENABLED=false \
time uv run bash scripts/Self-Forcing/run_qvg.sh 2>&1 | tee logs/self_forcing_mixed_low1_high2_r025.log
```

Output directory:

```text
results/selfforcing/mixed_static_global_lowint1_highint2_r0.25_64/kc_256_vc_256_nstages_1
```

### 3.3 1bit-2bit, ratio=0.50

```bash
MIXED_RATIO=0.50 \
MIXED_BIT_ENABLED=true \
MIXED_LOW_QUANT_TYPE=triton-nstages-kmeans-int1 \
MIXED_HIGH_QUANT_TYPE=triton-nstages-kmeans-int2 \
QUANT_FACTOR=1 \
CENTROID_CACHING_ENABLED=false \
time uv run bash scripts/Self-Forcing/run_qvg.sh 2>&1 | tee logs/self_forcing_mixed_low1_high2_r050.log
```

Output directory:

```text
results/selfforcing/mixed_static_global_lowint1_highint2_r0.50_64/kc_256_vc_256_nstages_1
```

---

## 4) Evaluate each group against BF16 using PSNR / SSIM / LPIPS

Evaluation script:

```text
scripts/Self-Forcing/run_metrics_psnr_ssim_lpips.sh
```

Alternatively, run a single command directly:

```text
python experiments/Self-Forcing/eval_psnr_ssim_lpips.py \
  --pred_folder <pred_dir> \
  --ref_folder <ref_dir> \
  --output_json <out.json> \
  --output_jsonl <out.jsonl> \
  --device cpu
```

Example:

```text
python experiments/Self-Forcing/eval_psnr_ssim_lpips.py \
  --pred_folder results/qvg-2bit \
  --ref_folder results/bf16 \
  --output_json results/qvg-2bit/metrics_summary.json \
  --output_jsonl results/qvg-2bit/metrics_per_video.jsonl \
  --device cpu
```

### 4.1 Evaluate 1bit-2bit, ratio=0.25

```bash
PRED_FOLDER=results/selfforcing/mixed_static_global_lowint1_highint2_r0.25_64/kc_256_vc_256_nstages_1 \
REF_FOLDER=results/selfforcing/bf16 \
uv run bash scripts/Self-Forcing/run_metrics_psnr_ssim_lpips.sh
```

### 4.2 Evaluate 1bit-2bit, ratio=0.50

```bash
PRED_FOLDER=results/selfforcing/mixed_static_global_lowint1_highint2_r0.50_64/kc_256_vc_256_nstages_1 \
REF_FOLDER=results/selfforcing/bf16 \
uv run bash scripts/Self-Forcing/run_metrics_psnr_ssim_lpips.sh
```

Each `PRED_FOLDER` will contain two newly generated metric files:

```text
metrics_psnr_ssim_lpips_summary.json
metrics_psnr_ssim_lpips_per_video.jsonl
```

---

## 5) Capture GPU memory / KV cache size, then grep the values into the report table

```bash
grep -E "Total KV Cache|Per Layer|Peak Memory" .log/self_forcing_qvg_int2.log | tail -n 1
```

Each log line looks like this:

```text
Peak Memory Usage: ... | Per Layer Memory Usage: ... | Total KV Cache Memory Usage: ...
```

These correspond to peak GPU memory, per-layer KV memory, and total KV cache size, respectively.

---

## 6) Quickly check whether all results are complete

### 6.1 View all videos

```bash
find results/selfforcing -maxdepth 6 -name "*.mp4"
```

### 6.2 View mixed-bit boundaries and memory logs

```bash
grep -E "Mixed-bit schedule|Mixed-bit KV spans|Peak Memory Usage|Per Layer Memory Usage|Quantization KV Cache Time|quant_factor|centroid_caching_enabled" logs/self_forcing_mixed_*.log
```

### 6.3 Summarize the two groups of metrics

```bash
for f in \
  results/selfforcing/mixed_static_global_lowint1_highint2_r0.25_64/kc_256_vc_256_nstages_1/metrics_psnr_ssim_lpips_summary.json \
  results/selfforcing/mixed_static_global_lowint1_highint2_r0.50_64/kc_256_vc_256_nstages_1/metrics_psnr_ssim_lpips_summary.json
do
  echo "==== $f"
  cat "$f"
done
```

---

## 7) Optional: enable centroid caching and run another round

If you want to perform an ablation study with centroid caching enabled versus disabled, simply change this line in the command:

```bash
CENTROID_CACHING_ENABLED=false
```

to:

```bash
CENTROID_CACHING_ENABLED=true
```

Keep all other parameters unchanged.
