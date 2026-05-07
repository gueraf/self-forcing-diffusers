from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import pathlib
import shutil
import subprocess
import sys
import tarfile

from .hf_assets import (
    DEFAULT_SELF_FORCING_CHECKPOINT_FILENAME,
    DEFAULT_SELF_FORCING_REPO_ID,
)


DEFAULT_PROMPT = "A cat walks on the grass, realistic style, high quality"
DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, worst quality, low quality"
)
DEFAULT_UPSTREAM_REPO_URL = "https://github.com/guandeh17/Self-Forcing.git"
DEFAULT_UPSTREAM_REPO_REF = "main"
DEFAULT_UPLOAD_REPO = "gueraf/self-forcing-diffusers"
DEFAULT_UPLOAD_TAG = "parity-artifacts"
DEFAULT_UPLOAD_ASSET_PREFIX = "sf-parity-latest"
DEFAULT_CLEAN_EXPORT_DURATION_SECONDS = 25.0


class ParityAssertionError(RuntimeError):
    pass


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _run_command(command, cwd=None, check=True, capture_output=False):
    print(f"$ {' '.join(str(part) for part in command)}")
    return subprocess.run(
        [str(part) for part in command],
        cwd=str(cwd) if cwd is not None else None,
        check=check,
        text=True,
        capture_output=capture_output,
    )


def _run_json_command(command, cwd=None):
    completed = _run_command(command, cwd=cwd, capture_output=True)
    return json.loads(completed.stdout)


def _git_head(repo_path: pathlib.Path) -> str:
    completed = _run_command(["git", "rev-parse", "HEAD"], cwd=repo_path, capture_output=True)
    return completed.stdout.strip()


def _resolve_origin_ref(repo_path: pathlib.Path, ref: str) -> str:
    probe = _run_command(["git", "rev-parse", "--verify", f"refs/remotes/origin/{ref}"], cwd=repo_path, check=False)
    if probe.returncode == 0:
        return f"origin/{ref}"
    return ref


def resolve_upstream_repo(upstream_repo_path: str | None, repo_url: str, ref: str) -> tuple[pathlib.Path, str]:
    if upstream_repo_path is not None:
        repo_path = pathlib.Path(upstream_repo_path).expanduser().resolve()
        return repo_path, _git_head(repo_path)

    cache_root = pathlib.Path.home() / ".cache" / "self-forcing-diffusers" / "upstream-repos"
    repo_path = cache_root / "Self-Forcing"
    cache_root.mkdir(parents=True, exist_ok=True)

    if not repo_path.exists():
        _run_command(["git", "clone", repo_url, repo_path])
    else:
        _run_command(["git", "fetch", "--tags", "--prune", "origin"], cwd=repo_path)

    checkout_target = _resolve_origin_ref(repo_path, ref)
    _run_command(["git", "checkout", "--detach", checkout_target], cwd=repo_path)
    return repo_path, _git_head(repo_path)


def load_json(path: pathlib.Path):
    with path.open() as handle:
        return json.load(handle)


def assert_conversion_report_exact(report: dict) -> dict:
    tensor_equivalence = report.get("tensor_equivalence", {})
    num_compared = int(tensor_equivalence.get("num_compared_tensors", 0))
    num_exact = int(tensor_equivalence.get("num_exact_tensor_matches", 0))
    max_abs_diff = tensor_equivalence.get("max_abs_diff")

    if num_compared == 0:
        raise ParityAssertionError("Checkpoint conversion report compared zero tensors.")
    if num_exact != num_compared or max_abs_diff != 0.0:
        raise ParityAssertionError(
            "Checkpoint conversion is not exact: "
            f"{num_exact}/{num_compared} exact tensors, max_abs_diff={max_abs_diff}."
        )

    return {
        "num_compared_tensors": num_compared,
        "num_exact_tensor_matches": num_exact,
        "max_abs_diff": max_abs_diff,
    }


def assert_validation_report_exact(report: dict) -> dict:
    latent_report = report.get("latent_report", {})
    max_abs_diff = latent_report.get("max_abs_diff")
    psnr_db = report.get("video_psnr_db")

    if max_abs_diff != 0.0 or not math.isinf(psnr_db):
        raise ParityAssertionError(
            "Upstream parity failed: "
            f"latent max_abs_diff={max_abs_diff}, video_psnr_db={psnr_db}."
        )

    return {
        "latent_max_abs_diff": max_abs_diff,
        "video_psnr_db": psnr_db,
        "num_frames": int(report.get("num_frames", 0)),
    }


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_artifact_bundle(run_dir: pathlib.Path, bundle_path: pathlib.Path, include_converted_model: bool) -> list[str]:
    excluded_paths = []
    bundle_path.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(bundle_path, "w:gz") as archive:
        for path in sorted(run_dir.rglob("*")):
            if path.is_dir():
                continue

            rel_path = path.relative_to(run_dir)
            if not include_converted_model and path.name == "diffusion_pytorch_model.safetensors":
                excluded_paths.append(str(rel_path))
                continue

            archive.add(path, arcname=str(pathlib.Path(run_dir.name) / rel_path))

    return excluded_paths


def ensure_release_exists(repo: str, tag: str, title: str):
    result = _run_command(["gh", "release", "view", tag, "--repo", repo], check=False)
    if result.returncode == 0:
        return

    _run_command(
        [
            "gh",
            "release",
            "create",
            tag,
            "--repo",
            repo,
            "--title",
            title,
            "--notes",
            "Automated Self-Forcing parity artifacts.",
        ]
    )


def upload_release_assets(repo: str, tag: str, assets: list[pathlib.Path]) -> dict:
    ensure_release_exists(repo, tag, title="Self-Forcing Parity Artifacts")

    upload_command = ["gh", "release", "upload", tag, "--repo", repo, "--clobber", *assets]
    _run_command(upload_command)

    release_payload = _run_json_command(["gh", "api", f"repos/{repo}/releases/tags/{tag}"])
    asset_urls = {
        asset["name"]: asset["browser_download_url"]
        for asset in release_payload.get("assets", [])
        if asset.get("name") in {path.name for path in assets}
    }

    return {
        "release_html_url": release_payload["html_url"],
        "assets": asset_urls,
    }


def _default_run_dir() -> pathlib.Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return _repo_root() / ".artifacts" / "parity-runs" / stamp


def _append_optional_path(command: list[str], flag: str, value: str | None):
    if value is not None:
        command.extend([flag, value])


def clean_export_num_chunks_for_duration(duration_seconds: float, fps: int, frames_per_chunk: int) -> int:
    if duration_seconds <= 0:
        raise ValueError(f"`duration_seconds` must be positive, but received {duration_seconds}.")
    if fps <= 0:
        raise ValueError(f"`fps` must be positive, but received {fps}.")
    if frames_per_chunk <= 0:
        raise ValueError(f"`frames_per_chunk` must be positive, but received {frames_per_chunk}.")

    required_frames = duration_seconds * fps
    return max(1, math.ceil(required_frames / frames_per_chunk))


def clean_export_duration_seconds(num_chunks: int, fps: int, frames_per_chunk: int) -> float:
    if num_chunks <= 0:
        raise ValueError(f"`num_chunks` must be positive, but received {num_chunks}.")
    if fps <= 0:
        raise ValueError(f"`fps` must be positive, but received {fps}.")
    if frames_per_chunk <= 0:
        raise ValueError(f"`frames_per_chunk` must be positive, but received {frames_per_chunk}.")

    return (num_chunks * frames_per_chunk) / fps


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run checkpoint conversion, upstream parity validation, clean export, and artifact upload in one command."
    )
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--checkpoint_repo_id", type=str, default=DEFAULT_SELF_FORCING_REPO_ID)
    parser.add_argument("--checkpoint_filename", type=str, default=DEFAULT_SELF_FORCING_CHECKPOINT_FILENAME)
    parser.add_argument("--upstream_repo_path", type=str, default=None)
    parser.add_argument("--upstream_repo_url", type=str, default=DEFAULT_UPSTREAM_REPO_URL)
    parser.add_argument("--upstream_repo_ref", type=str, default=DEFAULT_UPSTREAM_REPO_REF)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--text_encoder_device", type=str, default="cpu")
    parser.add_argument("--vae_device", type=str, default="cpu")
    parser.add_argument("--conversion_device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_chunks", type=int, default=3)
    parser.add_argument("--frames_per_chunk", type=int, default=9)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--clean_export_num_chunks", type=int, default=None)
    parser.add_argument("--clean_export_duration_seconds", type=float, default=DEFAULT_CLEAN_EXPORT_DURATION_SECONDS)
    parser.add_argument("--upload", choices=("github-release", "none"), default="github-release")
    parser.add_argument("--upload_repo", type=str, default=DEFAULT_UPLOAD_REPO)
    parser.add_argument("--upload_release_tag", type=str, default=DEFAULT_UPLOAD_TAG)
    parser.add_argument("--upload_asset_prefix", type=str, default=DEFAULT_UPLOAD_ASSET_PREFIX)
    parser.add_argument("--include_converted_model", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    repo_root = _repo_root()
    run_dir = pathlib.Path(args.run_dir).expanduser().resolve() if args.run_dir is not None else _default_run_dir()
    if run_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Run directory already exists: {run_dir}. Pass `--overwrite` to replace it.")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    converted_model_dir = run_dir / "converted_model"
    validation_dir = run_dir / "validation"
    clean_export_path = run_dir / "clean_export.mp4"
    run_manifest_path = run_dir / "run_manifest.json"
    bundle_path = run_dir.parent / f"{args.upload_asset_prefix}.tar.gz"
    upload_manifest_path = run_dir.parent / f"{args.upload_asset_prefix}.manifest.json"
    clean_export_asset_path = run_dir.parent / f"{args.upload_asset_prefix}.clean_export.mp4"
    validation_diffusers_asset_path = run_dir.parent / f"{args.upload_asset_prefix}.validation_diffusers.mp4"
    validation_original_asset_path = run_dir.parent / f"{args.upload_asset_prefix}.validation_original.mp4"
    clean_export_num_chunks = (
        args.clean_export_num_chunks
        if args.clean_export_num_chunks is not None
        else clean_export_num_chunks_for_duration(
            duration_seconds=args.clean_export_duration_seconds,
            fps=args.fps,
            frames_per_chunk=args.frames_per_chunk,
        )
    )
    clean_export_actual_duration_seconds = clean_export_duration_seconds(
        num_chunks=clean_export_num_chunks,
        fps=args.fps,
        frames_per_chunk=args.frames_per_chunk,
    )

    upstream_repo_dir, upstream_commit = resolve_upstream_repo(
        args.upstream_repo_path,
        repo_url=args.upstream_repo_url,
        ref=args.upstream_repo_ref,
    )

    convert_command = [
        sys.executable,
        repo_root / "scripts" / "convert_self_forcing_to_diffusers.py",
        "--checkpoint_repo_id",
        args.checkpoint_repo_id,
        "--checkpoint_filename",
        args.checkpoint_filename,
        "--output_path",
        converted_model_dir,
        "--device",
        args.conversion_device,
    ]
    _append_optional_path(convert_command, "--checkpoint_path", args.checkpoint_path)
    _run_command(convert_command, cwd=repo_root)

    validate_command = [
        sys.executable,
        repo_root / "scripts" / "validate_self_forcing_against_upstream.py",
        "--upstream_repo_path",
        upstream_repo_dir,
        "--checkpoint_repo_id",
        args.checkpoint_repo_id,
        "--checkpoint_filename",
        args.checkpoint_filename,
        "--diffusers_model_path",
        converted_model_dir,
        "--prompt",
        args.prompt,
        "--output_dir",
        validation_dir,
        "--device",
        args.device,
        "--text_encoder_device",
        args.text_encoder_device,
        "--seed",
        str(args.seed),
        "--num_chunks",
        str(args.num_chunks),
        "--frames_per_chunk",
        str(args.frames_per_chunk),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
    ]
    _append_optional_path(validate_command, "--checkpoint_path", args.checkpoint_path)
    _run_command(validate_command, cwd=repo_root)

    export_command = [
        sys.executable,
        repo_root / "scripts" / "autoregressive_video_generation.py",
        "--model_id",
        converted_model_dir,
        "--prompt",
        args.prompt,
        "--negative_prompt",
        args.negative_prompt,
        "--num_chunks",
        str(clean_export_num_chunks),
        "--frames_per_chunk",
        str(args.frames_per_chunk),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--device",
        args.device,
        "--text_encoder_device",
        args.text_encoder_device,
        "--vae_device",
        args.vae_device,
        "--seed",
        str(args.seed),
        "--fps",
        str(args.fps),
        "--output",
        clean_export_path,
    ]
    _run_command(export_command, cwd=repo_root)

    conversion_report = load_json(converted_model_dir / "validation_report.json")
    validation_report = load_json(validation_dir / "report.json")
    conversion_summary = assert_conversion_report_exact(conversion_report)
    validation_summary = assert_validation_report_exact(validation_report)

    run_manifest = {
        "run_dir": str(run_dir),
        "prompt": args.prompt,
        "checkpoint_path": args.checkpoint_path,
        "checkpoint_repo_id": args.checkpoint_repo_id,
        "checkpoint_filename": args.checkpoint_filename,
        "upstream_repo": {
            "path": str(upstream_repo_dir),
            "url": args.upstream_repo_url,
            "ref": args.upstream_repo_ref,
            "commit": upstream_commit,
        },
        "validation": {
            "num_chunks": args.num_chunks,
            "frames_per_chunk": args.frames_per_chunk,
            "fps": args.fps,
        },
        "clean_export": {
            "num_chunks": clean_export_num_chunks,
            "frames_per_chunk": args.frames_per_chunk,
            "fps": args.fps,
            "target_duration_seconds": args.clean_export_duration_seconds,
            "actual_duration_seconds": clean_export_actual_duration_seconds,
        },
        "conversion_summary": conversion_summary,
        "validation_summary": validation_summary,
        "artifacts": {
            "converted_model_dir": str(converted_model_dir),
            "conversion_report": str(converted_model_dir / "validation_report.json"),
            "validation_dir": str(validation_dir),
            "validation_report": str(validation_dir / "report.json"),
            "clean_export_video": str(clean_export_path),
        },
    }
    with run_manifest_path.open("w") as handle:
        json.dump(run_manifest, handle, indent=2)

    excluded_paths = create_artifact_bundle(
        run_dir=run_dir,
        bundle_path=bundle_path,
        include_converted_model=args.include_converted_model,
    )

    standalone_video_pairs = [
        (clean_export_path, clean_export_asset_path),
        (validation_dir / "diffusers.mp4", validation_diffusers_asset_path),
        (validation_dir / "original.mp4", validation_original_asset_path),
    ]
    for src, dst in standalone_video_pairs:
        shutil.copy2(src, dst)

    upload_manifest = dict(run_manifest)
    upload_manifest["excluded_paths"] = excluded_paths
    upload_manifest["bundle"] = {
        "path": str(bundle_path),
        "size_bytes": bundle_path.stat().st_size,
        "sha256": sha256_file(bundle_path),
    }
    upload_manifest["standalone_videos"] = {
        dst.name: {
            "path": str(dst),
            "size_bytes": dst.stat().st_size,
            "sha256": sha256_file(dst),
        }
        for _, dst in standalone_video_pairs
    }

    with upload_manifest_path.open("w") as handle:
        json.dump(upload_manifest, handle, indent=2)

    if args.upload == "github-release":
        upload_manifest["remote_storage"] = upload_release_assets(
            repo=args.upload_repo,
            tag=args.upload_release_tag,
            assets=[bundle_path, upload_manifest_path, *(dst for _, dst in standalone_video_pairs)],
        )
        with upload_manifest_path.open("w") as handle:
            json.dump(upload_manifest, handle, indent=2)
        _run_command(
            [
                "gh",
                "release",
                "upload",
                args.upload_release_tag,
                "--repo",
                args.upload_repo,
                "--clobber",
                upload_manifest_path,
            ]
        )
        print(json.dumps(upload_manifest["remote_storage"], indent=2))
    else:
        upload_manifest["remote_storage"] = None
        with upload_manifest_path.open("w") as handle:
            json.dump(upload_manifest, handle, indent=2)

    print(json.dumps(upload_manifest, indent=2))


if __name__ == "__main__":
    main()
