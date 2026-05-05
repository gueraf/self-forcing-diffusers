"""Generate a cat video using autoregressive Self-Forcing inference with the rolling KV cache.

Usage:
    uv run python scripts/generate_cat_video.py --device cuda
    uv run python scripts/generate_cat_video.py --device cuda --upload
"""

import argparse
import os
import subprocess
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPTS_DIR, "..", "src"))
sys.path.insert(0, _SCRIPTS_DIR)

from autoregressive_video_generation import generate_autoregressive_video

from diffusers.utils import export_to_video


CAT_PROMPT = "A cat walks on the grass, realistic style, high quality"
CAT_NEGATIVE_PROMPT = "Bright tones, overexposed, static, blurred details, subtitles, worst quality, low quality"

GITHUB_REPO = "gueraf/self-forcing-diffusers"
GITHUB_RELEASE = "parity-artifacts"


def main():
    parser = argparse.ArgumentParser(description="Generate a cat video with rolling KV cache")
    parser.add_argument("--model_id", default="gueraf/Self-Forcing-diffusers")
    parser.add_argument("--wan_base_model_id", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--num_chunks", type=int, default=45)
    parser.add_argument("--frames_per_chunk", type=int, default=9)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--window_size", type=int, default=-1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text_encoder_device", default=None)
    parser.add_argument("--vae_device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--output", default="cat_video.mp4")
    parser.add_argument("--upload", action="store_true", help="Upload to parity-artifacts GitHub release")
    args = parser.parse_args()

    print(f"Generating {args.num_chunks} chunks ({args.num_chunks * args.frames_per_chunk / args.fps:.1f}s) ...")
    frames = generate_autoregressive_video(
        prompt=CAT_PROMPT,
        negative_prompt=CAT_NEGATIVE_PROMPT,
        num_chunks=args.num_chunks,
        frames_per_chunk=args.frames_per_chunk,
        height=args.height,
        width=args.width,
        window_size=args.window_size,
        model_id=args.model_id,
        wan_base_model_id=args.wan_base_model_id,
        device=args.device,
        text_encoder_device=args.text_encoder_device,
        vae_device=args.vae_device,
        seed=args.seed,
    )

    export_to_video(frames, args.output, fps=args.fps)
    print(f"Saved {args.output} ({len(frames)} frames @ {args.fps}fps)")

    if args.upload:
        print(f"Uploading {args.output} to {GITHUB_REPO} release '{GITHUB_RELEASE}' ...")
        result = subprocess.run(
            ["gh", "release", "upload", GITHUB_RELEASE, args.output, "--clobber", "--repo", GITHUB_REPO],
            check=True,
        )
        print(f"Uploaded: https://github.com/{GITHUB_REPO}/releases/tag/{GITHUB_RELEASE}")


if __name__ == "__main__":
    main()
