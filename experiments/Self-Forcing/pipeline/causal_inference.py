from typing import List, Optional
import torch
import os
import numpy as np
from tqdm import tqdm
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation

from types import SimpleNamespace
from termcolor import cprint

from quant_videogen.compress import get_quantize_fn, compress_kv_cache
from quant_videogen.kv_cache import ChunkedKVCache, offload_kv_cache_layer, onload_kv_cache_layer

class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        # Environment variable for dumping KV cache
        self.dump_kv_level = int(os.getenv("DUMP_KV_LEVEL", "0"))  # 0: off, 1: final, 2: every block
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
            
        # Quant Config using SimpleNamespace
        self.quant_config = SimpleNamespace(**args.quant_config)
        self.mixed_final_num_chunks = None
        self.mixed_1bit_boundary_chunk = None
        self.quant_factor = max(1, int(getattr(self.quant_config, "quant_factor", 1)))
        self.centroid_caching_enabled = bool(getattr(self.quant_config, "centroid_caching_enabled", False))
        self._centroid_cache_k = {}
        self._centroid_cache_v = {}

        cprint(
            f"Quant trigger interval (quant_factor): {self.quant_factor} | "
            f"centroid_caching_enabled: {self.centroid_caching_enabled}",
            "light_cyan",
        )

        self.generator.model.kv_cache_cpu_offload = getattr(self.quant_config, "kv_cache_cpu_offload", False)

    def _reset_centroid_cache(self):
        self._centroid_cache_k.clear()
        self._centroid_cache_v.clear()

    def _get_centroid_cache_key(self, layer_idx: int, quant_type: str):
        return (layer_idx, quant_type)


    def quantize_kv_cache(self, tokens_to_quantize_start: int, tokens_to_quantize_end: int, max_tokens_to_quantize: int):
        """
        Quantize a range of the KV cache.  Indices are in token space and
        must be frame-aligned.
        """
        # Check the inputs
        assert tokens_to_quantize_start <= max_tokens_to_quantize and tokens_to_quantize_end <= max_tokens_to_quantize
        cprint(
            f"Quantizing {tokens_to_quantize_end - tokens_to_quantize_start} tokens, "
            f"from {tokens_to_quantize_start} to {tokens_to_quantize_end} tokens. "
            f"The max tokens to quantize is {max_tokens_to_quantize}",
            "light_cyan",
        )

        # Record the time using torch.cuda.Event
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()

        # Do nothing if quantization type is none
        if self.quant_config.quant_type == "none":
            self._print_memory_usage(self.kv_cache1)
            return

        cprint(f"Quantizing kv cache with config:\n{self.quant_config}", "light_blue")
        quant_spans = self._get_quantization_spans(
            tokens_to_quantize_start,
            tokens_to_quantize_end,
        )

        do_offload = getattr(self.quant_config, "kv_cache_cpu_offload", False)
        cuda_device = torch.device("cuda")

        with torch.no_grad():
            for layer_idx, layer in enumerate(self.kv_cache1):
                if do_offload:
                    onload_kv_cache_layer(layer, cuda_device)

                # Quantize each configured span independently.
                for span_start, span_end, span_quant_config in quant_spans:
                    self._quantize_kv_cache_span(
                        layer,
                        layer_idx,
                        span_start,
                        span_end,
                        span_quant_config,
                    )

                if do_offload:
                    offload_kv_cache_layer(layer)

        end_time.record()
        torch.cuda.synchronize()
        duration = start_time.elapsed_time(end_time)
        cprint(f"Quantization KV Cache Time: {(duration / 1000):.2f} s", "light_cyan")

        self._print_memory_usage(self.kv_cache1)

    def _get_quantization_spans(self, start_index: int, end_index: int):
        """Return frame-aligned quantization spans and their per-span configs."""
        mixed_enabled = getattr(self.quant_config, "mixed_bit_enabled", False)
        if not mixed_enabled:
            return [(start_index, end_index, self.quant_config)]

        num_tokens = end_index - start_index
        assert num_tokens % self.frame_seq_length == 0, (
            f"Quantization range length {num_tokens} must be frame-aligned "
            f"to frame_seq_length {self.frame_seq_length}"
        )

        schedule = getattr(self.quant_config, "mixed_schedule", "static_global")
        if schedule != "static_global":
            raise ValueError(f"Unsupported mixed-bit schedule: {schedule}")

        if self.mixed_1bit_boundary_chunk is None:
            raise RuntimeError("Mixed-bit static boundary has not been configured")

        ratio = float(getattr(self.quant_config, "mixed_1bit_ratio", 0.0))
        low_quant_type = getattr(
            self.quant_config,
            "mixed_low_quant_type",
            "triton-nstages-kmeans-int1",
        )
        high_quant_type = getattr(
            self.quant_config,
            "mixed_high_quant_type",
            "triton-nstages-kmeans-int2",
        )

        spans = []
        boundary_index = self.mixed_1bit_boundary_chunk * self.frame_seq_length
        if start_index < boundary_index:
            low_bit_end = min(end_index, boundary_index)
            spans.append((
                start_index,
                low_bit_end,
                self._make_span_quant_config(low_quant_type),
            ))
        if boundary_index < end_index:
            high_bit_start = max(start_index, boundary_index)
            spans.append((
                high_bit_start,
                end_index,
                self._make_span_quant_config(high_quant_type),
            ))

        span_desc = ", ".join(
            f"[{span_start}, {span_end}) -> {span_config.quant_type}"
            for span_start, span_end, span_config in spans
        )
        cprint(
            f"Mixed-bit KV spans: {span_desc} "
            f"(schedule={schedule}, low-bit ratio target={ratio:.2f}, "
            f"boundary chunk={self.mixed_1bit_boundary_chunk}/"
            f"{self.mixed_final_num_chunks})",
            "light_cyan",
        )
        return spans

    def _configure_mixed_bit_schedule(self, final_num_chunks: int):
        if not getattr(self.quant_config, "mixed_bit_enabled", False):
            self.mixed_final_num_chunks = None
            self.mixed_1bit_boundary_chunk = None
            return

        schedule = getattr(self.quant_config, "mixed_schedule", "static_global")
        if schedule != "static_global":
            raise ValueError(f"Unsupported mixed-bit schedule: {schedule}")

        ratio = float(getattr(self.quant_config, "mixed_1bit_ratio", 0.0))
        ratio = max(0.0, min(1.0, ratio))
        self.mixed_final_num_chunks = final_num_chunks
        self.mixed_1bit_boundary_chunk = int(final_num_chunks * ratio)
        self.mixed_1bit_boundary_chunk = max(
            0,
            min(final_num_chunks, self.mixed_1bit_boundary_chunk),
        )

        cprint(
            "Mixed-bit schedule: static_global | "
            f"Final chunks: {self.mixed_final_num_chunks} | "
            f"Low-bit boundary chunk: {self.mixed_1bit_boundary_chunk} | "
            f"Low-bit range: [0, {self.mixed_1bit_boundary_chunk}) | "
            f"High-bit range: [{self.mixed_1bit_boundary_chunk}, {self.mixed_final_num_chunks})",
            "light_cyan",
        )

    def _make_span_quant_config(self, quant_type: str):
        quant_config_dict = vars(self.quant_config).copy()
        quant_config_dict["quant_type"] = quant_type
        return SimpleNamespace(**quant_config_dict)

    def _quantize_kv_cache_span(
        self,
        layer: dict,
        layer_idx: int,
        start_index: int,
        end_index: int,
        quant_config: SimpleNamespace,
    ):
        if start_index >= end_index:
            return

        quantize_fn = get_quantize_fn(quant_config.quant_type, quant_config)

        # Read the span to quantize - always full precision [B, S, H, D]
        k = layer["k"].read(start_index, end_index)
        v = layer["v"].read(start_index, end_index)

        # compress_kv_cache expects [B, H, S, D]
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()

        centroid_init_data = None
        if self.centroid_caching_enabled:
            cache_key = self._get_centroid_cache_key(layer_idx, quant_config.quant_type)
            centroid_init_data = {
                "k": self._centroid_cache_k.get(cache_key),
                "v": self._centroid_cache_v.get(cache_key),
            }

        k_quant, v_quant = compress_kv_cache(
            k,
            v,
            quant_config.quant_type,
            quant_config,
            quantize_fn,
            centroid_init_data=centroid_init_data,
        )

        if self.centroid_caching_enabled:
            cache_key = self._get_centroid_cache_key(layer_idx, quant_config.quant_type)
            if isinstance(k_quant, dict) and "centroids_list" in k_quant:
                self._centroid_cache_k[cache_key] = k_quant["centroids_list"]
            if isinstance(v_quant, dict) and "centroids_list" in v_quant:
                self._centroid_cache_v[cache_key] = v_quant["centroids_list"]

        self._print_kv_cache_mse_error(k, k_quant, v, v_quant, layer_idx)

        # Pack decompression metadata for real-quantized dicts
        if isinstance(k_quant, dict) and isinstance(v_quant, dict):
            k_quant, v_quant = self._pack_info_into_kv_cache(
                k_quant, v_quant, k.dtype, quant_config
            )

        # store_quantized handles both tensor (fake) and dict (real)
        layer["k"].store_quantized(start_index, end_index, k_quant)
        layer["v"].store_quantized(start_index, end_index, v_quant)

    
    def _print_kv_cache_mse_error(self, k, k_quant, v, v_quant, layer_idx):
        """Print the Rel L2 error between the original and quantized KV cache."""

        if isinstance(k_quant, dict) and isinstance(v_quant, dict):
            return

        k_rel_l2 = torch.norm(k - k_quant, p=2) / torch.norm(k, p=2)
        v_rel_l2 = torch.norm(v - v_quant, p=2) / torch.norm(v, p=2)
        cprint(f"Layer {layer_idx:02d} | K Rel L2: {k_rel_l2:.4f} | V Rel L2: {v_rel_l2:.4f}", "light_blue")

    def _print_memory_usage(self, kv_cache_list: list[dict]):
        """Offload all KV cache, then onload layer-by-layer to measure memory usage."""

        torch.cuda.empty_cache()
        cuda_device = torch.device("cuda")

        for layer in kv_cache_list:
            offload_kv_cache_layer(layer)

        memory_usage_before_onload = torch.cuda.memory_allocated() / 1024**2
        memory_usage = [memory_usage_before_onload]

        for layer in kv_cache_list:
            onload_kv_cache_layer(layer, cuda_device)
            memory_usage.append(torch.cuda.memory_allocated() / 1024**2)

        peak_memory_usage = max(memory_usage)
        diff_per_layer = [memory_usage[i + 1] - memory_usage[i] for i in range(len(memory_usage) - 1)]
        per_layer_memory_usage = np.median(diff_per_layer)
        total_kv_cache_memory_usage = per_layer_memory_usage * len(kv_cache_list)

        cprint(
            f"Peak Memory Usage: {peak_memory_usage:.2f} MB | Other Memory Usage: {memory_usage[0]:.2f} MB | "
            f"Per Layer Memory Usage: {per_layer_memory_usage:.2f} MB | Total KV Cache Memory Usage: {total_kv_cache_memory_usage:.2f} MB",
            "light_blue",
        )

    def _pack_info_into_kv_cache(self, k_cache, v_cache, output_dtype, quant_config=None):
        """Pack metadata into KV cache. Only when the cached value is a real quantized dict."""
        quant_config = self.quant_config if quant_config is None else quant_config
        if isinstance(k_cache, dict) and isinstance(v_cache, dict):
            k_cache["info"] = {
                "output_dtype": output_dtype,
                "quant_config": quant_config,
            }
            v_cache["info"] = {
                "output_dtype": output_dtype,
                "quant_config": quant_config,
            }
        return k_cache, v_cache
    

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        self._reset_centroid_cache()
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        self._configure_mixed_bit_schedule(final_num_chunks=num_output_frames)
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros (size depends on target length when using global attention)
        if self.kv_cache1 is None:
            # When local_attn_size == -1 we need a cache large enough for the entire sequence
            if self.local_attn_size == -1:
                # reserve all output frames (input + to-generate) plus 1 extra block for safety
                target_frames = num_output_frames + 1
            else:
                target_frames = self.local_attn_size

            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device,
                target_frames=target_frames
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache
            for layer in self.kv_cache1:
                layer["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                layer["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        for chunk_index, current_num_frames in enumerate(tqdm(all_num_frames)):
            if profile:
                block_start.record()

            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
            
            #########################################################
            # Possiply quantize the KV cache here
            #########################################################
            # When generating the first 8 chunks, we do not quantize them.

            QUANT_FACTOR = self.quant_factor
            if chunk_index < QUANT_FACTOR or chunk_index % QUANT_FACTOR != 0:
                pass
            else:
                max_tokens_to_quantize = self.kv_cache1[0]["local_end_index"].item()
                cprint(f"\nAt Chunk {chunk_index}, max tokens to quantize: {max_tokens_to_quantize} tokens", "light_cyan")            
                
                # assert tokens_to_quantize == np.sum(all_num_frames[:chunk_index]) * self.frame_seq_length
                
                # Only quantize the previous chunks
                tokens_to_quantize_start = int(np.sum(all_num_frames[:chunk_index - QUANT_FACTOR]) * self.frame_seq_length)
                tokens_to_quantize_end = int(np.sum(all_num_frames[:chunk_index]) * self.frame_seq_length)
            
                self.quantize_kv_cache(tokens_to_quantize_start, tokens_to_quantize_end, max_tokens_to_quantize)
            
            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Step 4: Decode the output
        video = self.vae.decode_to_pixel(output, use_cache=False)
        
        # video = (video * 0.5 + 0.5).clamp(0, 1)
        video.mul_(0.5).add_(0.5).clamp_(0, 1)

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        # ----------------------------------------------------------
        # Optional: dump KV / CrossAttn cache for analysis
        # ----------------------------------------------------------
        if self.dump_kv_level >= 1:
            # Only rank-0 process dumps to avoid duplicates in DDP
            if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
                dump_dir = os.path.join(os.getenv("KV_DUMP_DIR", "kv_dumps"))
                os.makedirs(dump_dir, exist_ok=True)
                # Compose filename with prompt hash & frames
                filename = f"kv_cache_frames{num_output_frames}.pt"
                torch.save({
                    "kv_cache": self.kv_cache1,
                    "crossattn_cache": getattr(self, "crossattn_cache", None),
                }, os.path.join(dump_dir, filename))
                print(f"KV cache dumped to {os.path.join(dump_dir, filename)}")

        if return_latents:
            return video, output
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device, target_frames: int | None = None):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # allocate enough tokens for entire sequence when global attention
            total_frames = target_frames if target_frames is not None else 21
            kv_cache_size = total_frames * self.frame_seq_length

        max_num_chunks = kv_cache_size // self.frame_seq_length

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": ChunkedKVCache(batch_size, self.frame_seq_length, 12, 128, max_num_chunks, dtype, device, layout="BSHD"),
                "v": ChunkedKVCache(batch_size, self.frame_seq_length, 12, 128, max_num_chunks, dtype, device, layout="BSHD"),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
