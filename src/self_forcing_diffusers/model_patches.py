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

    from diffusers.models.normalization import RMSNorm
    from diffusers.models.transformers import transformer_wan as wan_mod

    def _wan_attn_processor_call(
        self,
        attn: "wan_mod.WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = wan_mod._get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                hidden_states_complex = torch.view_as_complex(
                    hidden_states.to(torch.float64).reshape(*hidden_states.shape[:-1], -1, 2)
                )
                freqs_complex = torch.complex(
                    freqs_cos[..., 0::2].to(torch.float64),
                    freqs_sin[..., 0::2].to(torch.float64),
                )
                out = torch.view_as_real(hidden_states_complex * freqs_complex).flatten(-2)
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = wan_mod._get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = wan_mod.dispatch_attention_fn(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
                parallel_config=None,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = wan_mod.dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=(self._parallel_config if encoder_hidden_states is None else None),
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    def _wan_attention_init(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-5,
        dropout: float = 0.0,
        added_kv_proj_dim: int | None = None,
        cross_attention_dim_head: int | None = None,
        processor=None,
        is_cross_attention=None,
    ):
        super(wan_mod.WanAttention, self).__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.inner_dim, dim, bias=True),
                torch.nn.Dropout(dropout),
            ]
        )
        self.norm_q = RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)
        self.norm_k = RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)

        self.add_k_proj = self.add_v_proj = None
        if added_kv_proj_dim is not None:
            self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.norm_added_k = torch.nn.RMSNorm(dim_head * heads, eps=eps)

        if is_cross_attention is not None:
            self.is_cross_attention = is_cross_attention
        else:
            self.is_cross_attention = cross_attention_dim_head is not None

        self.set_processor(processor)

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
            attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)

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
        attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)
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

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

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

    wan_mod.WanAttnProcessor.__call__ = _wan_attn_processor_call
    wan_mod.WanAttention.__init__ = _wan_attention_init
    wan_mod.WanTimeTextImageEmbedding.__init__ = _wan_time_text_image_embedding_init
    wan_mod.WanTimeTextImageEmbedding.forward = _wan_time_text_image_embedding_forward
    wan_mod.WanRotaryPosEmbed.__init__ = _wan_rotary_pos_embed_init
    wan_mod.WanRotaryPosEmbed.forward = _wan_rotary_pos_embed_forward
    wan_mod.WanTransformerBlock.forward = _wan_transformer_block_forward
    wan_mod.WanTransformer3DModel.forward = wan_mod.apply_lora_scale("attention_kwargs")(_wan_transformer_forward)

    _PATCHES_APPLIED = True
