from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


DEFAULT_SELF_FORCING_REPO_ID = "gdhe17/Self-Forcing"
DEFAULT_SELF_FORCING_CHECKPOINT_FILENAME = "checkpoints/self_forcing_dmd.pt"

DEFAULT_WAN_REPO_ID = "Wan-AI/Wan2.1-T2V-1.3B"
DEFAULT_WAN_CONFIG_FILENAME = "config.json"
DEFAULT_WAN_VAE_FILENAME = "Wan2.1_VAE.pth"
DEFAULT_WAN_TEXT_ENCODER_FILENAME = "models_t5_umt5-xxl-enc-bf16.pth"
DEFAULT_WAN_TOKENIZER_SUBDIR = "google/umt5-xxl"


def _resolve_local_or_downloaded_file(
    path: str | None,
    *,
    repo_id: str,
    filename: str,
) -> str:
    if path is not None:
        return str(Path(path).expanduser().resolve())
    return hf_hub_download(repo_id=repo_id, filename=filename)


def resolve_self_forcing_checkpoint_path(
    checkpoint_path: str | None,
    *,
    repo_id: str = DEFAULT_SELF_FORCING_REPO_ID,
    filename: str = DEFAULT_SELF_FORCING_CHECKPOINT_FILENAME,
) -> str:
    return _resolve_local_or_downloaded_file(checkpoint_path, repo_id=repo_id, filename=filename)


def resolve_wan_model_config_path(
    model_config_path: str | None,
    *,
    repo_id: str = DEFAULT_WAN_REPO_ID,
    filename: str = DEFAULT_WAN_CONFIG_FILENAME,
) -> str:
    return _resolve_local_or_downloaded_file(model_config_path, repo_id=repo_id, filename=filename)


def resolve_wan_vae_path(
    vae_path: str | None,
    *,
    repo_id: str = DEFAULT_WAN_REPO_ID,
    filename: str = DEFAULT_WAN_VAE_FILENAME,
) -> str:
    return _resolve_local_or_downloaded_file(vae_path, repo_id=repo_id, filename=filename)


def resolve_wan_text_encoder_weights_path(
    text_encoder_path: str | None,
    *,
    repo_id: str = DEFAULT_WAN_REPO_ID,
    filename: str = DEFAULT_WAN_TEXT_ENCODER_FILENAME,
) -> str:
    return _resolve_local_or_downloaded_file(text_encoder_path, repo_id=repo_id, filename=filename)


def resolve_wan_tokenizer_path(
    tokenizer_path: str | None,
    *,
    repo_id: str = DEFAULT_WAN_REPO_ID,
    subdir: str = DEFAULT_WAN_TOKENIZER_SUBDIR,
) -> str:
    if tokenizer_path is not None:
        return str(Path(tokenizer_path).expanduser().resolve())

    snapshot_dir = snapshot_download(repo_id=repo_id, allow_patterns=[f"{subdir}/*"])
    return str(Path(snapshot_dir) / subdir)
