# Quant-VideoGen: Mixed-Bit KV Cache Extension

This repository is our group-project extension of the released QuantVideoGen (QVG) codebase. The original QVG pipeline applies uniform 2-bit KV-cache quantization during inference. Our work studies a narrower question: can a simple temporal mixed-bit policy improve the memory/quality trade-off by using lower precision for older KV-cache chunks and higher precision for newer ones?

The current extension is implemented for the Self-Forcing integration only. We keep the original model weights and training setup unchanged; all of our changes happen at inference time in the KV-cache compression, cache storage, experiment control, and evaluation pipeline.

## Project Focus

We compare four settings on Self-Forcing:

- `BF16`: no KV-cache quantization.
- `QVG INT2`: the original uniform 2-bit QuantVideoGen baseline.
- `Mixed-25`: the oldest 25% of frame chunks use INT1, and the remaining 75% use INT2.
- `Mixed-50`: the oldest 50% of frame chunks use INT1, and the remaining 50% use INT2.

Our default mixed-bit policy is `static_global`. Before inference starts, we compute one global chunk boundary as `floor(final_num_chunks * mixed_1bit_ratio)`. During KV-cache compression, chunks before that boundary are quantized with the low-bit type, while chunks after the boundary use the high-bit type. If one quantization range crosses the boundary, it is split into two sub-spans and each sub-span is compressed with its own quantization config.

## What We Added Beyond Original QVG

### 1. Temporal mixed-bit scheduling for Self-Forcing

Original QVG uses one quantization type for the whole KV cache. We added a scheduler that lets different temporal regions use different bit-widths.

- `experiments/Self-Forcing/inference.py` parses mixed-bit arguments and injects them into `config.quant_config`.
- `experiments/Self-Forcing/pipeline/causal_inference.py` computes the global boundary with `_configure_mixed_bit_schedule()`.
- `_get_quantization_spans()` turns one token range into one or two frame-aligned spans, each with its own `quant_config`.
- `_quantize_kv_cache_span()` compresses every span independently, so one inference run can contain both INT1 and INT2 cache segments.

### 2. A new Triton PRQ `int1` quantization path

The original release already supports low-bit PRQ variants such as INT2 and INT4. To make INT1/INT2 mixed-bit experiments possible, we extended the quantization backend with 1-bit support.

- `quant_videogen/compress.py` adds `triton-nstages-kmeans-int1` to the quantization dispatch logic.
- `quant_videogen/functions.py` and `quant_videogen/real/prq.py` allow PRQ residual quantization with `num_bits=1`.
- `quant_videogen/real/quant_pack.py` implements sign-only 1-bit residual packing with blockwise mean-absolute scaling. For INT1, each residual value is reduced to its sign and eight values are packed into one byte.
- `quant_videogen/real/accumulate.py` adds the matching unpack-and-dequantize path so INT1 payloads can be reconstructed during attention reads.

### 3. Mixed-span KV-cache storage and correct per-span dequantization

Mixed-bit scheduling only works if the cache can remember which part was compressed with which bit-width. We therefore extended the chunked cache format to store quantized spans together with their own metadata.

- `quant_videogen/kv_cache.py` allows a cache to contain multiple quantized spans, not just one uniform quantized representation.
- Every real-quantized span stores `info.quant_config` and `info.output_dtype`.
- `quant_videogen/uncompress.py` extracts the `quant_type` and bit-width from the stored span metadata, then dispatches the correct dequantization path for that span.

This is what makes INT1 and INT2 segments coexist safely in the same KV cache.

### 4. Experiment controls for reproducible sweeps

We added a more explicit experiment interface around the Self-Forcing QVG script so mixed-bit runs can be reproduced without editing Python code.

- `scripts/Self-Forcing/run_qvg.sh` now exposes:
  - `MIXED_BIT_ENABLED`
  - `MIXED_SCHEDULE`
  - `MIXED_1BIT_RATIO` / `MIXED_RATIO`
  - `MIXED_LOW_QUANT_TYPE`
  - `MIXED_HIGH_QUANT_TYPE`
  - `QUANT_FACTOR`
  - `CENTROID_CACHING_ENABLED`
- Output folders are named from the active quantization settings, which makes baseline and mixed-bit runs easier to compare.
- `QUANT_FACTOR` controls how often KV quantization is triggered in chunk units.

### 5. Optional centroid warm-start for repeated quantization

Repeated PRQ over many KV spans can rerun K-Means many times. We added an optional engineering optimization that reuses centroids from the previous span of the same layer and quantization type.

- `experiments/Self-Forcing/pipeline/causal_inference.py` keeps centroid caches keyed by `(layer_idx, quant_type)`.
- `quant_videogen/compress.py` forwards optional centroid initialization data.
- `quant_videogen/real/prq.py` accepts `init_centroids_list` and warm-starts K-Means stage by stage when shapes match.

This does not change the model architecture; it is an inference-time acceleration and stabilization option for experiment sweeps.

### 6. Evaluation scripts for video-quality comparison

The guidebook is not only about generation commands; we also added evaluation tooling so the mixed-bit policy can be measured against BF16 references.

- `scripts/Self-Forcing/run_metrics_psnr_ssim_lpips.sh` is a wrapper for evaluation runs.
- `experiments/Self-Forcing/eval_psnr_ssim_lpips.py` computes PSNR, SSIM, and LPIPS between predicted videos and BF16 outputs.
- The metric script reuses the upstream LongCat metric implementation so frame handling stays consistent with the QVG codebase.

## Scope and Limitations

- The mixed-bit scheduler is currently connected only to the Self-Forcing path.
- The implemented policy is `static_global`; it is a fixed temporal split, not a learned or adaptive bit-allocation strategy.
- We do not retrain the model or modify model weights. This is an inference-time KV-cache extension on top of QVG.

## Running the Experiments

For the full command guide, see `Mixed_bit_quantization_command_guidebook.md`.

That guidebook covers:

- environment and model setup for Self-Forcing
- BF16 baseline generation
- uniform INT2 QVG baseline generation
- mixed INT1/INT2 generation with different ratios
- PSNR / SSIM / LPIPS evaluation against BF16 outputs

## Upstream Base

This project inherits from the released QuantVideoGen codebase:

- Repository: https://github.com/svg-project/Quant-VideoGen
- Project page: https://svg-project.github.io/qvg/
- Paper: https://arxiv.org/abs/2602.02958

Most of the original model integrations, baseline KV-cache quantization framework, and environment setup come from upstream QVG. This README focuses on the additional mixed-bit design and implementation contributed in this group project.
