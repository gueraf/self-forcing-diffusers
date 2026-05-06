from __future__ import annotations

import torch


def _chunk_sequence(chunks: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
    if isinstance(chunks, torch.Tensor):
        if chunks.ndim != 5:
            raise ValueError(
                "`latents` must be a single 5D latent chunk `(batch, channels, frames, height, width)` or a list "
                "of such tensors."
            )
        return [chunks]

    if not isinstance(chunks, (list, tuple)) or len(chunks) == 0:
        raise ValueError("`latents` must be a tensor or a non-empty list/tuple of tensors.")

    for chunk in chunks:
        if not isinstance(chunk, torch.Tensor) or chunk.ndim != 5:
            raise ValueError("Every latent chunk must be a 5D torch.Tensor.")

    return list(chunks)


def _normalize_frame_offsets(
    transformer: torch.nn.Module,
    chunks: list[torch.Tensor],
    frame_offset: int | list[int] | tuple[int, ...],
) -> list[int]:
    if isinstance(frame_offset, int):
        offsets = []
        current_offset = frame_offset
        for chunk in chunks:
            patch_frames = chunk.shape[2] // transformer.config.patch_size[0]
            offsets.append(current_offset)
            current_offset += patch_frames
        return offsets

    if len(frame_offset) != len(chunks):
        raise ValueError("`frame_offset` must have the same length as `latents` when passing multiple chunks.")

    return list(frame_offset)


@torch.no_grad()
def write_rolling_kv_cache(
    transformer: torch.nn.Module,
    latents: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    encoder_hidden_states: torch.Tensor,
    rolling_kv_cache,
    *,
    frame_offset: int | list[int] | tuple[int, ...] = 0,
    write_mode: str = "append",
) -> None:
    """Write one or more chunks into the rolling KV cache.

    The first chunk uses ``write_mode``; subsequent chunks always append (they sit at the
    new cache end after the first write).
    """
    chunks = _chunk_sequence(latents)
    frame_offsets = _normalize_frame_offsets(transformer, chunks, frame_offset)

    prev_write_mode = rolling_kv_cache.write_mode

    try:
        for i, (chunk, chunk_frame_offset) in enumerate(zip(chunks, frame_offsets)):
            patch_frames = chunk.shape[2] // transformer.config.patch_size[0]
            timestep = torch.zeros((chunk.shape[0], patch_frames), device=chunk.device, dtype=torch.long)

            rolling_kv_cache.configure_write(write_mode=write_mode if i == 0 else "append")

            transformer(
                hidden_states=chunk,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                frame_offset=chunk_frame_offset,
                return_dict=False,
                attention_kwargs={"rolling_kv_cache": rolling_kv_cache},
            )
    finally:
        rolling_kv_cache.write_mode = prev_write_mode
