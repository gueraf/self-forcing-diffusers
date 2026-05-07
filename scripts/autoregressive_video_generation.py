"""Autoregressive Self-Forcing video generation with rolling KV caching.

This example runs chunk-wise Wan/Self-Forcing inference and keeps clean self-attention KV
states across chunks. It also supports injecting ground-truth video chunks at any chunk index:
the reference frames are VAE-encoded, written into the cache with absolute temporal positions,
and generation continues from that point onward.

Usage:
    python scripts/autoregressive_video_generation.py \
        --prompt "A cat walks on the grass, realistic" \
        --num_chunks 27 \
        --frames_per_chunk 9 \
        --output output.mp4

    python scripts/autoregressive_video_generation.py \
        --prompt "A fox runs through a forest" \
        --conditioning_video path/to/reference.mp4 \
        --conditioning_start_chunk 4 \
        --output conditioned_output.mp4
"""

import argparse
import os
import sys

import torch
from transformers import AutoTokenizer, UMT5EncoderModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from self_forcing_diffusers.model_patches import apply_self_forcing_wan_model_patches
from self_forcing_diffusers.rolling_kv import write_kv_cache
from self_forcing_diffusers.sf_inference import (
    add_sf_noise,
    build_sf_denoising_steps,
    build_sf_scheduler_tables,
    convert_sf_flow_to_x0,
    sample_sf_renoise,
)


apply_self_forcing_wan_model_patches()

from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanPipeline, WanTransformer3DModel
from diffusers.models.transformers.transformer_wan import WanKVCache
from diffusers.utils import export_to_video, load_video


def _retrieve_latents(encoder_output, sample_mode="argmax"):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latent_dist"):
        return encoder_output.latent_dist.sample()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not retrieve latents from the VAE encoder output.")


def _get_latent_stats(pipe, device):
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(device, torch.float32)
    )
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(
        device,
        torch.float32,
    )
    return latents_mean, latents_std


def _decode_latents(pipe, latents, latents_mean, latents_std):
    """Decode the full latent stack in one VAE call.

    Per-chunk decoding would reset the Wan VAE's causal feat_cache between chunks and produce
    a visible cut at every chunk boundary. Decoding once at the end keeps temporal continuity.
    """
    vae_device = next(pipe.vae.parameters()).device
    decode_latents = latents.to(device=vae_device, dtype=pipe.vae.dtype)
    latents_mean = latents_mean.to(device=vae_device, dtype=torch.float32)
    latents_std = latents_std.to(device=vae_device, dtype=torch.float32)
    decode_latents = decode_latents / latents_std + latents_mean
    with torch.no_grad():
        video = pipe.vae.decode(decode_latents, return_dict=False)[0]
    return pipe.video_processor.postprocess_video(video, output_type="pil")[0]


def _assert_valid_self_forcing_transformer(transformer):
    has_cross_attn_norm = bool(getattr(transformer.config, "cross_attn_norm", False))
    has_norm_module = transformer.blocks and not isinstance(transformer.blocks[0].norm2, torch.nn.Identity)

    if has_cross_attn_norm and has_norm_module:
        return

    raise ValueError(
        "The loaded transformer is missing Self-Forcing cross-attention norms. "
        "Re-convert the checkpoint with `scripts/convert_self_forcing_to_diffusers.py` before running autoregressive generation."
    )


def _align_self_forcing_transformer_dtype(transformer):
    runtime_device = transformer.patch_embedding.weight.device
    runtime_dtype = transformer.patch_embedding.weight.dtype

    transformer.condition_embedder.time_embedder.to(device=runtime_device, dtype=runtime_dtype)
    transformer.scale_shift_table.data = transformer.scale_shift_table.data.to(device=runtime_device, dtype=runtime_dtype)

    for block in transformer.blocks:
        block.scale_shift_table.data = block.scale_shift_table.data.to(device=runtime_device, dtype=runtime_dtype)

        if hasattr(block.norm2, "weight") and block.norm2.weight is not None:
            block.norm2.weight.data = block.norm2.weight.data.to(device=runtime_device, dtype=runtime_dtype)
        if hasattr(block.norm2, "bias") and block.norm2.bias is not None:
            block.norm2.bias.data = block.norm2.bias.data.to(device=runtime_device, dtype=runtime_dtype)


def _load_pipeline(model_id, wan_base_model_id, device, text_encoder_device=None, vae_device=None):
    transformer = WanTransformer3DModel.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    _assert_valid_self_forcing_transformer(transformer)
    _align_self_forcing_transformer_dtype(transformer)
    vae = AutoencoderKLWan.from_pretrained(wan_base_model_id, subfolder="vae", torch_dtype=torch.float32)
    vae_device = vae_device or device

    if text_encoder_device is None and vae_device == device:
        pipe = WanPipeline.from_pretrained(
            wan_base_model_id,
            vae=vae,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        pipe.scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0, num_train_timesteps=1000)
        pipe.to(device)
        return pipe, device

    text_encoder_device = text_encoder_device or device
    text_encoder_dtype = torch.float32 if text_encoder_device == "cpu" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(wan_base_model_id, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        wan_base_model_id,
        subfolder="text_encoder",
        torch_dtype=text_encoder_dtype,
    )
    text_encoder.to(text_encoder_device)

    pipe = WanPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=FlowMatchEulerDiscreteScheduler(shift=5.0, num_train_timesteps=1000),
        transformer=transformer,
    )
    pipe.transformer.to(device)
    pipe.vae.to(vae_device)
    return pipe, text_encoder_device


def _chunk_video_frames(video_frames, frames_per_chunk):
    if len(video_frames) % frames_per_chunk != 0:
        raise ValueError(
            f"`conditioning_video` must have a multiple of {frames_per_chunk} frames, but received {len(video_frames)}."
        )
    return [video_frames[idx : idx + frames_per_chunk] for idx in range(0, len(video_frames), frames_per_chunk)]


def _encode_reference_chunk(pipe, frames, height, width, latents_mean, latents_std, vae_device, transformer_device):
    reference_video = pipe.video_processor.preprocess_video(frames, height=height, width=width).to(vae_device, torch.float32)
    reference_frames = pipe.video_processor.postprocess_video(reference_video, output_type="pil")[0]
    with torch.no_grad():
        encoded = _retrieve_latents(pipe.vae.encode(reference_video.to(dtype=pipe.vae.dtype)), sample_mode="argmax")
    encoded = (encoded.to(torch.float32) - latents_mean) * latents_std
    return (
        encoded.to(device=transformer_device, dtype=pipe.transformer.dtype),
        reference_frames,
    )


def _generate_chunk_velocity(
    pipe,
    noisy_input,
    timestep,
    prompt_embeds,
    negative_prompt_embeds,
    guidance_scale,
    frame_offset,
    cond_cache,
    uncond_cache,
    overwrite_newest: bool,
):
    if timestep.ndim == 0:
        timestep = timestep.expand(noisy_input.shape[0], noisy_input.shape[2])
    elif timestep.ndim == 1:
        timestep = timestep.unsqueeze(1).expand(noisy_input.shape[0], noisy_input.shape[2])

    def run(kv_cache, encoder_hidden_states):
        prev_overwrite_newest = kv_cache.overwrite_newest
        try:
            if overwrite_newest:
                kv_cache.enable_overwrite_mode()
            else:
                kv_cache.enable_append_mode()
            return pipe.transformer(
                noisy_input,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                frame_offset=frame_offset,
                return_dict=False,
                attention_kwargs={"kv_cache": kv_cache},
            )[0]
        finally:
            kv_cache.overwrite_newest = prev_overwrite_newest

    if guidance_scale > 1.0:
        velocity_cond = run(cond_cache, prompt_embeds)
        velocity_uncond = run(uncond_cache, negative_prompt_embeds)
        return velocity_uncond + guidance_scale * (velocity_cond - velocity_uncond)

    return run(cond_cache, prompt_embeds)


def generate_autoregressive_video(
    prompt,
    negative_prompt="",
    num_chunks=27,
    frames_per_chunk=9,
    height=480,
    width=832,
    guidance_scale=1.0,
    window_size=-1,
    model_id="gueraf/Self-Forcing-diffusers",
    wan_base_model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    conditioning_video=None,
    conditioning_start_chunk=0,
    device="cuda",
    text_encoder_device=None,
    vae_device=None,
    seed=0,
):
    pipe, prompt_device = _load_pipeline(
        model_id=model_id,
        wan_base_model_id=wan_base_model_id,
        device=device,
        text_encoder_device=text_encoder_device,
        vae_device=vae_device,
    )

    denoising_steps = build_sf_denoising_steps(device=device)
    scheduler_timesteps, scheduler_sigmas = build_sf_scheduler_tables(device=device)
    do_cfg = guidance_scale > 1.0
    prompt_dtype = torch.float32 if prompt_device == "cpu" else torch.bfloat16

    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            do_classifier_free_guidance=do_cfg,
            device=prompt_device,
            dtype=prompt_dtype,
        )
        prompt_embeds = prompt_embeds.to(device=device, dtype=torch.bfloat16)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=torch.bfloat16)

    if text_encoder_device is not None and text_encoder_device != "cpu":
        pipe.text_encoder.to("cpu")
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    num_blocks = len(pipe.transformer.blocks)
    cond_cache = WanKVCache(num_blocks=num_blocks, window_size=window_size)
    uncond_cache = WanKVCache(num_blocks=num_blocks, window_size=window_size) if do_cfg else None

    p_t, _, _ = pipe.transformer.config.patch_size
    latent_h = height // pipe.vae_scale_factor_spatial
    latent_w = width // pipe.vae_scale_factor_spatial
    latent_t = (frames_per_chunk - 1) // pipe.vae_scale_factor_temporal + 1
    patch_frames_per_chunk = latent_t // p_t
    vae_device = next(pipe.vae.parameters()).device
    latents_mean, latents_std = _get_latent_stats(pipe, vae_device)

    reference_chunks = []
    if conditioning_video is not None:
        video_frames = load_video(conditioning_video)
        for chunk_frames in _chunk_video_frames(video_frames, frames_per_chunk):
            reference_chunks.append(
                _encode_reference_chunk(
                    pipe,
                    chunk_frames,
                    height,
                    width,
                    latents_mean,
                    latents_std,
                    vae_device=vae_device,
                    transformer_device=device,
                )
            )
        if conditioning_start_chunk + len(reference_chunks) > num_chunks:
            raise ValueError("`conditioning_video` would write past `num_chunks`.")

    generator = torch.Generator(device=device).manual_seed(seed)
    chunk_latents = []
    chunk_idx = 0

    while chunk_idx < num_chunks:
        if reference_chunks and chunk_idx == conditioning_start_chunk:
            conditioning_latents = [latents for latents, _ in reference_chunks]
            reference_frame_offset = chunk_idx * patch_frames_per_chunk
            write_kv_cache(
                pipe.transformer,
                conditioning_latents,
                prompt_embeds,
                cond_cache,
                frame_offset=reference_frame_offset,
                overwrite_first_chunk=False,
            )
            if do_cfg:
                write_kv_cache(
                    pipe.transformer,
                    conditioning_latents,
                    negative_prompt_embeds,
                    uncond_cache,
                    frame_offset=reference_frame_offset,
                    overwrite_first_chunk=False,
                )

            for latents, _ in reference_chunks:
                chunk_latents.append(latents)
                print(f"Injected reference chunk at index {chunk_idx}")
                chunk_idx += 1
            continue

        frame_offset = chunk_idx * patch_frames_per_chunk
        noisy_input = torch.randn(
            (1, pipe.vae.config.z_dim, latent_t, latent_h, latent_w),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )

        with torch.no_grad():
            for step_idx, timestep in enumerate(denoising_steps):
                model_timestep = timestep.expand(noisy_input.shape[0], noisy_input.shape[2])
                velocity = _generate_chunk_velocity(
                    pipe,
                    noisy_input,
                    timestep=model_timestep,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    guidance_scale=guidance_scale,
                    frame_offset=frame_offset,
                    cond_cache=cond_cache,
                    uncond_cache=uncond_cache,
                    overwrite_newest=step_idx > 0,
                )
                x0_pred = convert_sf_flow_to_x0(
                    velocity,
                    noisy_input,
                    model_timestep,
                    scheduler_timesteps,
                    scheduler_sigmas,
                )

                if step_idx < len(denoising_steps) - 1:
                    next_timestep = denoising_steps[step_idx + 1].expand(noisy_input.shape[0], noisy_input.shape[2])
                    eps = sample_sf_renoise(x0_pred, generator=generator)
                    noisy_input = add_sf_noise(
                        x0_pred,
                        eps,
                        next_timestep,
                        scheduler_timesteps,
                        scheduler_sigmas,
                    )

        write_kv_cache(
            pipe.transformer,
            x0_pred,
            prompt_embeds,
            cond_cache,
            frame_offset=frame_offset,
            overwrite_first_chunk=True,
        )
        if do_cfg:
            write_kv_cache(
                pipe.transformer,
                x0_pred,
                negative_prompt_embeds,
                uncond_cache,
                frame_offset=frame_offset,
                overwrite_first_chunk=True,
            )

        chunk_latents.append(x0_pred)
        print(f"Generated chunk {chunk_idx + 1}/{num_chunks}")
        chunk_idx += 1

    cond_cache.reset()
    if uncond_cache is not None:
        uncond_cache.reset()

    print(f"Decoding {len(chunk_latents)} chunks ...")
    full_latents = torch.cat(chunk_latents, dim=2).to(device=vae_device, dtype=torch.float32)
    return _decode_latents(pipe, full_latents, latents_mean, latents_std)


def main():
    parser = argparse.ArgumentParser(description="Autoregressive Self-Forcing video generation with rolling KV cache")
    parser.add_argument("--prompt", default="A cat walks on the grass, realistic style, high quality")
    parser.add_argument(
        "--negative_prompt",
        default="Bright tones, overexposed, static, blurred details, subtitles, worst quality, low quality",
    )
    parser.add_argument("--num_chunks", type=int, default=45)
    parser.add_argument("--frames_per_chunk", type=int, default=9)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--window_size", type=int, default=-1)
    parser.add_argument("--model_id", default="gueraf/Self-Forcing-diffusers")
    parser.add_argument("--wan_base_model_id", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--conditioning_video", type=str, default=None)
    parser.add_argument("--conditioning_start_chunk", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text_encoder_device", default=None)
    parser.add_argument("--vae_device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="autoregressive_output.mp4")
    parser.add_argument("--fps", type=int, default=16)
    args = parser.parse_args()

    frames = generate_autoregressive_video(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_chunks=args.num_chunks,
        frames_per_chunk=args.frames_per_chunk,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        window_size=args.window_size,
        model_id=args.model_id,
        wan_base_model_id=args.wan_base_model_id,
        conditioning_video=args.conditioning_video,
        conditioning_start_chunk=args.conditioning_start_chunk,
        device=args.device,
        text_encoder_device=args.text_encoder_device,
        vae_device=args.vae_device,
        seed=args.seed,
    )
    export_to_video(frames, args.output, fps=args.fps)
    print(f"Saved {args.output} ({len(frames)} frames @ {args.fps}fps)")


if __name__ == "__main__":
    main()
