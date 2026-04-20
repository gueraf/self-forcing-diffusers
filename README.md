# self-forcing-diffusers

Self-Forcing conversion, validation, and E2E generation utilities on top of a clean `diffusers` hook branch.

This repo keeps the Self-Forcing-specific Wan patches, checkpoint conversion logic, and upstream parity validation outside `gueraf/diffusers`. The only required `diffusers` fork dependency is the rolling KV cache hook branch:

- `https://github.com/gueraf/diffusers/tree/rolling-kv-cache-hook-12600`

## Setup

```bash
uv sync
```

The `pyproject.toml` pins `diffusers` to the hook-only fork branch via `tool.uv.sources`.

## Main Commands

Convert a checkpoint:

```bash
uv run python scripts/convert_self_forcing_to_diffusers.py \
  --output_path ./artifacts/self_forcing_diffusers
```

If `--checkpoint_path` is omitted, the script downloads `checkpoints/self_forcing_dmd.pt` from `gdhe17/Self-Forcing`.

Run autoregressive generation:

```bash
uv run python scripts/autoregressive_video_generation.py \
  --model_id ./artifacts/self_forcing_diffusers \
  --prompt "A cat walks on the grass, realistic style, high quality" \
  --num_chunks 2 \
  --frames_per_chunk 9 \
  --output ./artifacts/autoregressive.mp4
```

Validate directly against the original upstream repo:

```bash
uv run python scripts/validate_self_forcing_against_upstream.py \
  --upstream_repo_path /path/to/Self-Forcing \
  --diffusers_model_path ./artifacts/self_forcing_diffusers \
  --prompt "A cat walks on the grass, realistic style, high quality" \
  --output_dir ./artifacts/upstream_compare
```

If `--checkpoint_path`, `--wan_model_config`, or `--vae_path` are omitted, the validator downloads them automatically from Hugging Face:

- `gdhe17/Self-Forcing` for `checkpoints/self_forcing_dmd.pt`
- `Wan-AI/Wan2.1-T2V-1.3B` for `config.json`, `Wan2.1_VAE.pth`, and the upstream tokenizer/text-encoder weights used by the default `--text_encoder_source upstream` flow

## Tests

Unit tests:

```bash
uv run python -m unittest tests.test_conversion tests.test_validation_helpers -v
```

The heavier E2E coverage lives in the scripts themselves and expects:

- the original `guandeh17/Self-Forcing` repo checked out locally
- the upstream Self-Forcing checkpoint
- the Wan base model assets
