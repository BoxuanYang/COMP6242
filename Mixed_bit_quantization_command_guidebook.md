## -1) 环境与模型准备

若第一次在这台机器跑 Self-Forcing，先完成如下步骤：

### -1.1 进入项目目录

```bash
cd Quant-VideoGen
```

### -1.2 安装 Self-Forcing 依赖（若尚未安装）

```bash
uv pip install -e ".[selfforcing]"
```

### -1.3 配置 Hugging Face 访问（若尚未登录）

```bash
hf auth login
```

### -1.4 下载 Self-Forcing 模型与 checkpoint

```bash
bash scripts/Self-Forcing/download_models.sh
```

该脚本会下载：

- `ckpts/Self-Forcing/Wan2.1-T2V-1.3B`
- `ckpts/Self-Forcing/self_forcing_dmd.pt`

### -1.5 下载完成后快速校验

```bash
ls -l ckpts/Self-Forcing/self_forcing_dmd.pt
ls -ld ckpts/Self-Forcing/Wan2.1-T2V-1.3B
```

如果这两个路径存在，后续即可直接跑，不需要每次重复下载。

---

在环境和模型都已配置完成后，下面是 Self-Forcing 实验的完整执行手册：

- 先跑 BF16 baseline。
- 再跑 4 组 mixed-bit：
	- 4bit-2bit: ratio 0.25 / 0.50
	- 2bit-1bit: ratio 0.25 / 0.50
- 每组都和 BF16 对比，计算 PSNR / SSIM / LPIPS。

本手册默认：

- quant factor 固定为 1（每个 chunk 触发量化）
- centroid caching 提供开关（默认示例先关闭，便于和现有 baseline 对齐）

---

## 0) 参数修改导航

优先级建议：

1. **优先用环境变量临时覆盖**（推荐，最不容易改坏默认脚本）。
2. 只有当你希望“以后每次都用新默认值”时，再去改脚本/配置文件。

### 0.1 生成实验参数（推荐：命令行环境变量覆盖）

入口脚本：

```text
scripts/Self-Forcing/run_qvg.sh
```

可直接在命令前覆盖的参数：

- mixed 策略：`MIXED_RATIO`、`MIXED_LOW_QUANT_TYPE`、`MIXED_HIGH_QUANT_TYPE`
- 量化触发与缓存：`QUANT_FACTOR`、`CENTROID_CACHING_ENABLED`
- 输出目录：`OUTPUT_FOLDER`
- 视频长度与注意力窗口：`NUM_OUTPUT_FRAMES`、`LOCAL_ATTN_SIZE`
- 量化细节：`QUANT_BLOCK_SIZE`、`NUM_PRQ_STAGES`、`CACHE_NUM_K_CENTROIDS`、`CACHE_NUM_V_CENTROIDS`、`KMEANS_MAX_ITERS`
- 模型与输入：`CKPT_PATH`、`PROMPTS_PATH`

示例（不改文件，临时覆盖）：

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

如果你想修改“默认值”而不是每次传环境变量，就编辑：

```text
scripts/Self-Forcing/run_qvg.sh
```

里面这些默认行（形如 `${VAR:-default}`）即可。

### 0.2 Self-Forcing 模型/推理基线配置（需要改 yaml）

文件：

```text
experiments/Self-Forcing/configs/self_forcing_dmd.yaml
```

常见会改的项：

- `num_frame_per_block`
- `denoising_step_list`
- `guidance_scale`
- 其他 Self-Forcing 原生配置

说明：这类改动会改变实验协议本身，不再只是 mixed-bit 政策变化。

### 0.3 评测参数（PSNR/SSIM/LPIPS）

入口脚本：

```text
scripts/Self-Forcing/run_metrics_psnr_ssim_lpips.sh
```

可覆盖参数：

- `PRED_FOLDER`（待评测结果目录）
- `REF_FOLDER`（BF16 参考目录）
- `SKIP_FRAMES`
- `DEVICE`
- `OUTPUT_JSON`、`OUTPUT_JSONL`

底层实现脚本（通常不需要改）：

```text
experiments/Self-Forcing/eval_psnr_ssim_lpips.py
```

### 0.4 代码级高级开关（通常不用改）

如果你要扩展算法实现，而不是只调实验参数，主要位置是：

- 参数注入：`experiments/Self-Forcing/inference.py`
- mixed 调度与触发：`experiments/Self-Forcing/pipeline/causal_inference.py`
- 压缩/解压与 centroid warm-start：`quant_videogen/compress.py`、`quant_videogen/functions.py`、`quant_videogen/real/prq.py`

---

## 1) 进入项目并创建日志目录

```bash
cd Quant-VideoGen
mkdir -p logs
```

可选：快速确认脚本里的参数入口

```bash
grep -nE "MIXED_RATIO|MIXED_LOW_QUANT_TYPE|MIXED_HIGH_QUANT_TYPE|QUANT_FACTOR|CENTROID_CACHING_ENABLED|OUTPUT_FOLDER" scripts/Self-Forcing/run_qvg.sh
```

---

## 2) 先跑 BF16 baseline

```bash
bash scripts/Self-Forcing/run_bf16.sh 2>&1 | tee logs/self_forcing_bf16.log
```

BF16 视频输出目录：

```text
results/selfforcing/bf16
```

---

## 3) 跑 4 组 mixed-bit 生成

说明：

- 不需要再手改脚本文件，直接通过环境变量覆盖。
- quant factor 固定为 1：`QUANT_FACTOR=1`
- centroid caching 可开关：`CENTROID_CACHING_ENABLED=true/false`

### 3.1 4bit-2bit, ratio=0.25

```bash
MIXED_RATIO=0.25 \
MIXED_LOW_QUANT_TYPE=triton-nstages-kmeans-int4 \
MIXED_HIGH_QUANT_TYPE=triton-nstages-kmeans-int2 \
QUANT_FACTOR=1 \
CENTROID_CACHING_ENABLED=true \
uv run bash scripts/Self-Forcing/run_qvg.sh 2>&1 | tee logs/self_forcing_mixed_low4_high2_r025.log
```

输出目录：

```text
results/selfforcing/mixed_static_global_lowint4_0.25_highint2_64/kc_256_vc_256_nstages_1
```

### 3.2 4bit-2bit, ratio=0.50

```bash
MIXED_RATIO=0.50 \
MIXED_LOW_QUANT_TYPE=triton-nstages-kmeans-int4 \
MIXED_HIGH_QUANT_TYPE=triton-nstages-kmeans-int2 \
QUANT_FACTOR=1 \
CENTROID_CACHING_ENABLED=true \
uv run bash scripts/Self-Forcing/run_qvg.sh 2>&1 | tee logs/self_forcing_mixed_low4_high2_r050.log
```

输出目录：

```text
results/selfforcing/mixed_static_global_lowint4_0.50_highint2_64/kc_256_vc_256_nstages_1
```

### 3.3 2bit-1bit, ratio=0.25

```bash
MIXED_RATIO=0.25 \
MIXED_LOW_QUANT_TYPE=triton-nstages-kmeans-int2 \
MIXED_HIGH_QUANT_TYPE=triton-nstages-kmeans-int1 \
QUANT_FACTOR=1 \
CENTROID_CACHING_ENABLED=true \
uv run bash scripts/Self-Forcing/run_qvg.sh 2>&1 | tee logs/self_forcing_mixed_low2_high1_r025.log
```

输出目录：

```text
results/selfforcing/mixed_static_global_lowint2_0.25_highint1_64/kc_256_vc_256_nstages_1
```

### 3.4 2bit-1bit, ratio=0.50

```bash
MIXED_RATIO=0.50 \
MIXED_LOW_QUANT_TYPE=triton-nstages-kmeans-int2 \
MIXED_HIGH_QUANT_TYPE=triton-nstages-kmeans-int1 \
QUANT_FACTOR=1 \
CENTROID_CACHING_ENABLED=true \
uv run bash scripts/Self-Forcing/run_qvg.sh 2>&1 | tee logs/self_forcing_mixed_low2_high1_r050.log
```

输出目录：

```text
results/selfforcing/mixed_static_global_lowint2_0.50_highint1_64/kc_256_vc_256_nstages_1
```

---

## 4) 每组对 BF16 评测 PSNR / SSIM / LPIPS

评测脚本：

```text
scripts/Self-Forcing/run_metrics_psnr_ssim_lpips.sh
```

或者直接单条：

```text
python experiments/Self-Forcing/eval_psnr_ssim_lpips.py \
  --pred_folder <pred_dir> \
  --ref_folder <ref_dir> \
  --output_json <out.json> \
  --output_jsonl <out.jsonl> \
  --device cpu
```

e.g.,

```text
python experiments/Self-Forcing/eval_psnr_ssim_lpips.py \
  --pred_folder results/qvg-2bit \
  --ref_folder results/bf16 \
  --output_json results/qvg-2bit/metrics_summary.json \
  --output_jsonl results/qvg-2bit/metrics_per_video.jsonl \
  --device cpu
```

### 4.1 评测 4bit-2bit, ratio=0.25

```bash
PRED_FOLDER=results/selfforcing/mixed_static_global_lowint4_0.25_highint2_64/kc_256_vc_256_nstages_1 \
REF_FOLDER=results/selfforcing/bf16 \
uv run bash scripts/Self-Forcing/run_metrics_psnr_ssim_lpips.sh
```

### 4.2 评测 4bit-2bit, ratio=0.50

```bash
PRED_FOLDER=results/selfforcing/mixed_static_global_lowint4_0.50_highint2_64/kc_256_vc_256_nstages_1 \
REF_FOLDER=results/selfforcing/bf16 \
uv run bash scripts/Self-Forcing/run_metrics_psnr_ssim_lpips.sh
```

### 4.3 评测 2bit-1bit, ratio=0.25

```bash
PRED_FOLDER=results/selfforcing/mixed_static_global_lowint2_0.25_highint1_64/kc_256_vc_256_nstages_1 \
REF_FOLDER=results/selfforcing/bf16 \
uv run bash scripts/Self-Forcing/run_metrics_psnr_ssim_lpips.sh
```

### 4.4 评测 2bit-1bit, ratio=0.50

```bash
PRED_FOLDER=results/selfforcing/mixed_static_global_lowint2_0.50_highint1_64/kc_256_vc_256_nstages_1 \
REF_FOLDER=results/selfforcing/bf16 \
uv run bash scripts/Self-Forcing/run_metrics_psnr_ssim_lpips.sh
```

每个 PRED_FOLDER 下会新增两份指标文件：

```text
metrics_psnr_ssim_lpips_summary.json
metrics_psnr_ssim_lpips_per_video.jsonl
```

---

## 5) 快速检查结果是否齐全

### 5.1 查看所有视频

```bash
find results/selfforcing -maxdepth 6 -name "*.mp4"
```

### 5.2 查看 mixed 分界与显存日志

```bash
grep -E "Mixed-bit schedule|Mixed-bit KV spans|Peak Memory Usage|Per Layer Memory Usage|Quantization KV Cache Time|quant_factor|centroid_caching_enabled" logs/self_forcing_mixed_*.log
```

### 5.3 汇总 4 组指标

```bash
for f in \
	results/selfforcing/mixed_static_global_lowint4_0.25_highint2_64/kc_256_vc_256_nstages_1/metrics_psnr_ssim_lpips_summary.json \
	results/selfforcing/mixed_static_global_lowint4_0.50_highint2_64/kc_256_vc_256_nstages_1/metrics_psnr_ssim_lpips_summary.json \
	results/selfforcing/mixed_static_global_lowint2_0.25_highint1_64/kc_256_vc_256_nstages_1/metrics_psnr_ssim_lpips_summary.json \
	results/selfforcing/mixed_static_global_lowint2_0.50_highint1_64/kc_256_vc_256_nstages_1/metrics_psnr_ssim_lpips_summary.json
do
	echo "==== $f"
	cat "$f"
done
```

---

## 6) 可选：打开 centroid caching 再跑一轮

如果要做 ablation（开/关 centroid caching），只需把命令里的：

```bash
CENTROID_CACHING_ENABLED=false
```

改成：

```bash
CENTROID_CACHING_ENABLED=true
```

其他参数保持不变即可。