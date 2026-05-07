# self-forcing-diffusers

Tools to run the Self-Forcing causal Wan model on top of the upstream `diffusers` `WanKVCache` API,
plus an end-to-end test that confirms the diffusers path stays bit-exact against the original
[`guandeh17/Self-Forcing`](https://github.com/guandeh17/Self-Forcing) repo.

The only required `diffusers` fork dependency is the rolling KV cache branch, which tracks
current `huggingface/diffusers` plus `WanKVCache`:

- `https://github.com/gueraf/diffusers/tree/wan-rolling-kv-cache`

## Setup

```bash
uv sync
```

`pyproject.toml` pins `diffusers` to `gueraf/diffusers@wan-rolling-kv-cache` via `tool.uv.sources`.

## Generate a video

```bash
uv run python scripts/autoregressive_video_generation.py \
  --prompt "A cat walks on the grass, realistic style, high quality" \
  --output ./autoregressive.mp4
```

The default `--model_id` is `gueraf/Self-Forcing-diffusers` (already converted, on the Hub), so no
manual conversion is needed for plain inference. Defaults: `45` chunks, `9` frames/chunk, `16fps`,
~25s of video, unbounded KV cache. Pass `--device`, `--text_encoder_device`, `--vae_device` to
split across GPUs if you don't have a single 80GB+ card.

To start from a real reference clip, pass `--conditioning_video path/to/clip.mp4
--conditioning_start_chunk 0`; the reference is VAE-encoded into the cache and generation
continues from there.

## Verify parity against upstream

```bash
uv run sf-e2e-parity \
  --device cuda:0 \
  --text_encoder_device cuda:1 \
  --vae_device cuda:1
```

`sf-e2e-parity` runs the full check in one command:

1. converts the original Self-Forcing checkpoint into diffusers format,
2. runs both the upstream causal-DMD path and the diffusers `WanKVCache` path from the same noise
   and seed and asserts `max_abs_diff = 0.0` on the latents,
3. exports a longer `clean_export.mp4` (default ~25s) via the diffusers script,
4. bundles the reports and videos and uploads the artifact bundle plus the three videos to the
   `parity-artifacts` GitHub release on `gueraf/self-forcing-diffusers`.

Override the long export length with `--clean_export_duration_seconds` or
`--clean_export_num_chunks`. The shorter parity check uses `--num_chunks` (default `3`).

If `--upstream_repo_path` is omitted, the original repo is cloned or refreshed automatically under
`~/.cache/self-forcing-diffusers/upstream-repos/Self-Forcing`.

The same parity check is also runnable on its own without the conversion / export / upload steps:

```bash
uv run python scripts/validate_self_forcing_against_upstream.py \
  --diffusers_model_path gueraf/Self-Forcing-diffusers \
  --output_dir ./upstream_compare
```

## Convert a checkpoint manually

```bash
uv run python scripts/convert_self_forcing_to_diffusers.py \
  --output_path ./self_forcing_diffusers
```

If `--checkpoint_path` is omitted, the script downloads `checkpoints/self_forcing_dmd.pt` from
`gdhe17/Self-Forcing`.

## Tests

```bash
uv run python -m unittest tests.test_hf_assets tests.test_conversion tests.test_validation_helpers tests.test_parity_runner -v
```

These are unit tests; the parity check above is the heavy E2E coverage and expects the original
`guandeh17/Self-Forcing` repo and the upstream checkpoint/Wan VAE weights.
