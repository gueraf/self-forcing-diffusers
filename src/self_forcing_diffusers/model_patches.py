from __future__ import annotations

import torch
import torch.nn as nn


_PATCHES_APPLIED = False


def assert_valid_self_forcing_transformer(transformer):
    has_cross_attn_norm = bool(getattr(transformer.config, "cross_attn_norm", False))
    has_norm_module = transformer.blocks and not isinstance(transformer.blocks[0].norm2, torch.nn.Identity)

    if has_cross_attn_norm and has_norm_module:
        return

    raise ValueError(
        "The loaded transformer is missing Self-Forcing cross-attention norms. "
        "Re-convert the checkpoint with `scripts/convert_self_forcing_to_diffusers.py` before running validation or generation."
    )


def align_self_forcing_transformer_dtype(transformer):
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


def apply_self_forcing_wan_model_patches():
    global _PATCHES_APPLIED

    if _PATCHES_APPLIED:
        return

    from diffusers.models.transformers import transformer_wan as wan_mod

    def _wan_time_text_image_embedding_init(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: int | None = None,
        pos_embed_seq_len: int | None = None,
    ):
        super(wan_mod.WanTimeTextImageEmbedding, self).__init__()

        self.time_freq_dim = time_freq_dim
        self.timesteps_proj = wan_mod.Timesteps(
            num_channels=time_freq_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.time_embedder = wan_mod.TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = wan_mod.PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = wan_mod.WanImageEmbedding(image_embed_dim, dim, pos_embed_seq_len=pos_embed_seq_len)

    def _wan_time_text_image_embedding_forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        timestep_seq_len: int | None = None,
    ):
        half = self.time_freq_dim // 2
        timestep_fp64 = timestep.reshape(-1).to(torch.float64)
        freqs = torch.pow(
            torch.tensor(10000.0, device=timestep.device, dtype=torch.float64),
            -torch.arange(half, device=timestep.device, dtype=torch.float64).div(half),
        )
        timestep = torch.outer(timestep_fp64, freqs)
        timestep = torch.cat([torch.cos(timestep), torch.sin(timestep)], dim=1).to(dtype=encoder_hidden_states.dtype)
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image

    def _wan_rotary_pos_embed_init(
        self,
        attention_head_dim: int,
        patch_size: tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super(wan_mod.WanRotaryPosEmbed, self).__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        self.t_dim = t_dim
        self.h_dim = h_dim
        self.w_dim = w_dim

        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64
        complex_dtype = torch.complex64 if freqs_dtype == torch.float32 else torch.complex128

        def _wan_rope_params(dim: int) -> torch.Tensor:
            if dim % 2 != 0:
                raise ValueError(f"`dim` must be even for Wan rotary embeddings, but received {dim}.")

            freqs = torch.outer(
                torch.arange(max_seq_len, dtype=freqs_dtype),
                1.0
                / torch.pow(
                    torch.tensor(theta, dtype=freqs_dtype),
                    torch.arange(0, dim, 2, dtype=freqs_dtype).div(dim),
                ),
            )
            return torch.polar(torch.ones_like(freqs), freqs).to(complex_dtype)

        self.register_buffer("freqs_t_complex", _wan_rope_params(t_dim), persistent=False)
        self.register_buffer("freqs_h_complex", _wan_rope_params(h_dim), persistent=False)
        self.register_buffer("freqs_w_complex", _wan_rope_params(w_dim), persistent=False)

    def _wan_rotary_pos_embed_forward(self, hidden_states: torch.Tensor, frame_offset: int = 0):
        _, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        freqs_complex_f = self.freqs_t_complex[frame_offset : frame_offset + ppf].view(ppf, 1, 1, -1).expand(
            ppf, pph, ppw, -1
        )
        freqs_complex_h = self.freqs_h_complex[:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_complex_w = self.freqs_w_complex[:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_complex = torch.cat([freqs_complex_f, freqs_complex_h, freqs_complex_w], dim=-1).reshape(
            1, ppf * pph * ppw, 1, -1
        )
        freqs_cos = freqs_complex.real.repeat_interleave(2, dim=-1)
        freqs_sin = freqs_complex.imag.repeat_interleave(2, dim=-1)

        return freqs_cos, freqs_sin

    def _wan_transformer_block_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        kv_cache: "wan_mod.WanKVCache | None" = None,
        block_idx: int | None = None,
    ) -> torch.Tensor:
        per_frame_modulation = temb.ndim == 4 and temb.shape[1] != hidden_states.shape[1]
        temb = temb.to(hidden_states.dtype)

        if temb.ndim == 4:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0).to(device=hidden_states.device, dtype=hidden_states.dtype) + temb
            ).chunk(6, dim=2)
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.to(device=hidden_states.device, dtype=hidden_states.dtype) + temb
            ).chunk(6, dim=1)

        if per_frame_modulation:
            num_frames = shift_msa.shape[1]
            frame_seq_len = hidden_states.shape[1] // num_frames

            norm_hidden_states = self.norm1(hidden_states).unflatten(1, (num_frames, frame_seq_len))
            norm_hidden_states = (norm_hidden_states * (1 + scale_msa.unsqueeze(2)) + shift_msa.unsqueeze(2)).flatten(
                1, 2
            )
            attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb, kv_cache=kv_cache, block_idx=block_idx)

            hidden_states = hidden_states + (
                attn_output.unflatten(1, (num_frames, frame_seq_len)) * gate_msa.unsqueeze(2)
            ).flatten(1, 2)

            norm_hidden_states = self.norm2(hidden_states)
            attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
            hidden_states = hidden_states + attn_output

            norm_hidden_states = self.norm3(hidden_states).unflatten(1, (num_frames, frame_seq_len))
            norm_hidden_states = (
                norm_hidden_states * (1 + c_scale_msa.unsqueeze(2)) + c_shift_msa.unsqueeze(2)
            ).flatten(1, 2)
            ff_output = self.ffn(norm_hidden_states)

            hidden_states = hidden_states + (
                ff_output.unflatten(1, (num_frames, frame_seq_len)) * c_gate_msa.unsqueeze(2)
            ).flatten(1, 2)
            return hidden_states

        norm_hidden_states = self.norm1(hidden_states) * (1 + scale_msa) + shift_msa
        attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb, kv_cache=kv_cache, block_idx=block_idx)
        hidden_states = hidden_states + attn_output * gate_msa

        norm_hidden_states = self.norm2(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm3(hidden_states) * (1 + c_scale_msa) + c_shift_msa
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = hidden_states + ff_output * c_gate_msa

        return hidden_states

    def _wan_transformer_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        return_dict: bool = True,
        attention_kwargs: dict[str, torch.Tensor] | None = None,
        frame_offset: int = 0,
    ):
        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states, frame_offset=frame_offset)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep,
            encoder_hidden_states,
            encoder_hidden_states_image,
            timestep_seq_len=ts_seq_len,
        )
        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        kv_cache = (attention_kwargs or {}).pop("kv_cache", None)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block_idx, block in enumerate(self.blocks):
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                )
        else:
            for block_idx, block in enumerate(self.blocks):
                hidden_states = block(
                    hidden_states, encoder_hidden_states, timestep_proj, rotary_emb,
                    kv_cache=kv_cache, block_idx=block_idx,
                )

        if temb.ndim == 3:
            if temb.shape[1] == hidden_states.shape[1]:
                shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(
                    2, dim=2
                )
                shift = shift.squeeze(2)
                scale = scale.squeeze(2)
            else:
                num_frames = temb.shape[1]
                frame_seq_len = hidden_states.shape[1] // num_frames
                shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(
                    2, dim=2
                )
                shift = shift.squeeze(2)
                scale = scale.squeeze(2)
        else:
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        if temb.ndim == 3 and temb.shape[1] != hidden_states.shape[1]:
            norm_hidden_states = self.norm_out(hidden_states).unflatten(1, (num_frames, frame_seq_len))
            hidden_states = (
                norm_hidden_states * (1 + scale.unsqueeze(2).to(hidden_states.device))
                + shift.unsqueeze(2).to(hidden_states.device)
            ).flatten(1, 2).type_as(hidden_states)
        else:
            hidden_states = (self.norm_out(hidden_states) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)

        return wan_mod.Transformer2DModelOutput(sample=output)

    # Match upstream WanRMSNorm semantics for Q/K norms: compute the RMS in float32,
    # cast the normalized result back, then apply the affine weight in the input's dtype.
    # diffusers' attn1.norm_q/norm_k use torch.nn.RMSNorm which computes the entire RMS
    # in the input dtype (bfloat16). Tiny per-step rounding accumulates across denoising
    # steps and chunks into latent diffs of ~3.9 against upstream — restoring float32 RMS
    # gives bit-exact parity.
    def _wan_rms_norm_forward(self, inputs):
        normed = inputs.float() * torch.rsqrt(
            inputs.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return normed.type_as(inputs) * self.weight

    wan_mod.WanTimeTextImageEmbedding.__init__ = _wan_time_text_image_embedding_init
    wan_mod.WanTimeTextImageEmbedding.forward = _wan_time_text_image_embedding_forward
    wan_mod.WanRotaryPosEmbed.__init__ = _wan_rotary_pos_embed_init
    wan_mod.WanRotaryPosEmbed.forward = _wan_rotary_pos_embed_forward
    wan_mod.WanTransformerBlock.forward = _wan_transformer_block_forward
    wan_mod.WanTransformer3DModel.forward = wan_mod.apply_lora_scale("attention_kwargs")(_wan_transformer_forward)
    torch.nn.RMSNorm.forward = _wan_rms_norm_forward

    _PATCHES_APPLIED = True
