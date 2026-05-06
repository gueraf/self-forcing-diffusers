# self-forcing-diffusers

Self-Forcing conversion, validation, and E2E generation utilities using `WanKVCache` from diffusers.

This repo keeps the Self-Forcing-specific Wan patches, checkpoint conversion logic, and upstream parity validation outside `gueraf/diffusers`. The only required `diffusers` fork dependency is the rolling KV cache branch, which is synced to current `huggingface/diffusers` plus `WanKVCache` for autoregressive inference:

- `https://github.com/gueraf/diffusers/tree/wan-rolling-kv-cache`

## Setup

```bash
uv sync
```

The `pyproject.toml` pins `diffusers` to `gueraf/diffusers@wan-rolling-kv-cache` via `tool.uv.sources`.

## Main Commands

Run the full parity flow in one command:

```bash
uv run sf-e2e-parity \
  --device cuda:1 \
  --text_encoder_device cpu \
  --vae_device cpu
```

That command:

- converts the original Self-Forcing checkpoint into diffusers format
- validates exact parity against `guandeh17/Self-Forcing`
- runs the public diffusers autoregressive export path and writes a roughly `25s` `clean_export.mp4` by default
- bundles the reports and videos
- uploads the artifact bundle, manifest, and the three videos as standalone assets (`<prefix>.clean_export.mp4`, `<prefix>.validation_diffusers.mp4`, `<prefix>.validation_original.mp4`) to the `parity-artifacts` GitHub release in `gueraf/self-forcing-diffusers`

The remote artifact path uses GitHub release assets rather than Git LFS, so the bundle stays under the per-file release limit and does not consume LFS storage/bandwidth quota.

You can override the long export length with either `--clean_export_duration_seconds` or `--clean_export_num_chunks`. The shorter exact upstream parity check still uses `--num_chunks`, which defaults to `3`.

If `--upstream_repo_path` is omitted, the original repo is cloned or refreshed automatically under `~/.cache/self-forcing-diffusers/upstream-repos/Self-Forcing`.

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
  --output ./artifacts/autoregressive.mp4
```

The standalone autoregressive script now defaults to `45` chunks at `16fps`, which is about `25.3s` of video with the default `9` frames per chunk.

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
uv run python -m unittest tests.test_hf_assets tests.test_conversion tests.test_validation_helpers tests.test_parity_runner -v
```

The heavier E2E coverage lives in the scripts themselves and expects:

- the original `guandeh17/Self-Forcing` repo checked out locally
- the upstream Self-Forcing checkpoint
- the Wan base model assets
