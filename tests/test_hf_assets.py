import pathlib
import tempfile
import unittest
from unittest import mock

from self_forcing_diffusers.hf_assets import (
    DEFAULT_SELF_FORCING_CHECKPOINT_FILENAME,
    DEFAULT_SELF_FORCING_REPO_ID,
    DEFAULT_WAN_REPO_ID,
    DEFAULT_WAN_TOKENIZER_SUBDIR,
    DEFAULT_WAN_VAE_FILENAME,
    resolve_self_forcing_checkpoint_path,
    resolve_wan_tokenizer_path,
    resolve_wan_vae_path,
)


class TestHfAssets(unittest.TestCase):
    def test_resolve_self_forcing_checkpoint_path_keeps_explicit_local_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = pathlib.Path(tmpdir) / "checkpoint.pt"
            checkpoint_path.write_bytes(b"checkpoint")

            resolved = resolve_self_forcing_checkpoint_path(str(checkpoint_path))

        self.assertEqual(resolved, str(checkpoint_path.resolve()))

    @mock.patch("self_forcing_diffusers.hf_assets.hf_hub_download")
    def test_resolve_self_forcing_checkpoint_path_downloads_default_when_missing(self, mock_download):
        mock_download.return_value = "/tmp/self_forcing_dmd.pt"

        resolved = resolve_self_forcing_checkpoint_path(None)

        self.assertEqual(resolved, "/tmp/self_forcing_dmd.pt")
        mock_download.assert_called_once_with(
            repo_id=DEFAULT_SELF_FORCING_REPO_ID,
            filename=DEFAULT_SELF_FORCING_CHECKPOINT_FILENAME,
        )

    @mock.patch("self_forcing_diffusers.hf_assets.hf_hub_download")
    def test_resolve_wan_vae_path_downloads_default_when_missing(self, mock_download):
        mock_download.return_value = "/tmp/Wan2.1_VAE.pth"

        resolved = resolve_wan_vae_path(None)

        self.assertEqual(resolved, "/tmp/Wan2.1_VAE.pth")
        mock_download.assert_called_once_with(
            repo_id=DEFAULT_WAN_REPO_ID,
            filename=DEFAULT_WAN_VAE_FILENAME,
        )

    @mock.patch("self_forcing_diffusers.hf_assets.snapshot_download")
    def test_resolve_wan_tokenizer_path_downloads_tokenizer_subdir_when_missing(self, mock_download):
        mock_download.return_value = "/tmp/wan_snapshot"

        resolved = resolve_wan_tokenizer_path(None)

        self.assertEqual(resolved, "/tmp/wan_snapshot/google/umt5-xxl")
        mock_download.assert_called_once_with(
            repo_id=DEFAULT_WAN_REPO_ID,
            allow_patterns=[f"{DEFAULT_WAN_TOKENIZER_SUBDIR}/*"],
        )


if __name__ == "__main__":
    unittest.main()
