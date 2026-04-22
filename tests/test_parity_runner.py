import json
import pathlib
import subprocess
import tarfile
import tempfile
import unittest
from unittest import mock

from self_forcing_diffusers.parity_runner import (
    ParityAssertionError,
    assert_conversion_report_exact,
    assert_validation_report_exact,
    clean_export_duration_seconds,
    clean_export_num_chunks_for_duration,
    create_artifact_bundle,
    upload_release_assets,
)


class TestParityRunner(unittest.TestCase):
    def test_assert_conversion_report_exact_accepts_exact_converter_report(self):
        report = {
            "tensor_equivalence": {
                "num_compared_tensors": 825,
                "num_exact_tensor_matches": 825,
                "max_abs_diff": 0.0,
            }
        }

        summary = assert_conversion_report_exact(report)

        self.assertEqual(summary["num_compared_tensors"], 825)
        self.assertEqual(summary["num_exact_tensor_matches"], 825)
        self.assertEqual(summary["max_abs_diff"], 0.0)

    def test_assert_conversion_report_exact_rejects_non_exact_converter_report(self):
        report = {
            "tensor_equivalence": {
                "num_compared_tensors": 825,
                "num_exact_tensor_matches": 824,
                "max_abs_diff": 1e-5,
            }
        }

        with self.assertRaisesRegex(ParityAssertionError, "Checkpoint conversion is not exact"):
            assert_conversion_report_exact(report)

    def test_assert_validation_report_exact_rejects_non_exact_parity(self):
        report = {
            "latent_report": {
                "max_abs_diff": 0.01,
            },
            "video_psnr_db": 32.0,
        }

        with self.assertRaisesRegex(ParityAssertionError, "Upstream parity failed"):
            assert_validation_report_exact(report)

    def test_assert_validation_report_exact_returns_reported_frame_count(self):
        report = {
            "num_frames": 9,
            "latent_report": {
                "max_abs_diff": 0.0,
            },
            "video_psnr_db": float("inf"),
        }

        summary = assert_validation_report_exact(report)

        self.assertEqual(summary["num_frames"], 9)

    def test_clean_export_num_chunks_for_duration_rounds_up_to_cover_requested_runtime(self):
        self.assertEqual(clean_export_num_chunks_for_duration(25.0, fps=16, frames_per_chunk=9), 45)

    def test_clean_export_duration_seconds_matches_chunk_count(self):
        self.assertEqual(clean_export_duration_seconds(num_chunks=45, fps=16, frames_per_chunk=9), 25.3125)

    def test_create_artifact_bundle_excludes_converted_weights_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = pathlib.Path(tmpdir) / "run"
            converted_dir = run_dir / "converted_model"
            validation_dir = run_dir / "validation"
            converted_dir.mkdir(parents=True)
            validation_dir.mkdir(parents=True)

            (converted_dir / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
            (converted_dir / "config.json").write_text("{}")
            (validation_dir / "report.json").write_text("{}")

            bundle_path = pathlib.Path(tmpdir) / "bundle.tar.gz"
            excluded = create_artifact_bundle(run_dir, bundle_path, include_converted_model=False)

            self.assertEqual(excluded, ["converted_model/diffusion_pytorch_model.safetensors"])
            with tarfile.open(bundle_path, "r:gz") as archive:
                names = archive.getnames()

            self.assertIn("run/converted_model/config.json", names)
            self.assertIn("run/validation/report.json", names)
            self.assertNotIn("run/converted_model/diffusion_pytorch_model.safetensors", names)

    @mock.patch("self_forcing_diffusers.parity_runner.subprocess.run")
    def test_upload_release_assets_creates_release_if_missing(self, mock_run):
        mock_run.side_effect = [
            subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps(
                    {
                        "html_url": "https://github.com/gueraf/self-forcing-diffusers/releases/tag/parity-artifacts",
                        "assets": [
                            {
                                "name": "bundle.tar.gz",
                                "browser_download_url": "https://example.com/bundle.tar.gz",
                            }
                        ],
                    }
                ),
                stderr="",
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            asset_path = pathlib.Path(tmpdir) / "bundle.tar.gz"
            asset_path.write_bytes(b"bundle")

            upload_info = upload_release_assets(
                repo="gueraf/self-forcing-diffusers",
                tag="parity-artifacts",
                assets=[asset_path],
            )

        self.assertEqual(
            upload_info["assets"]["bundle.tar.gz"],
            "https://example.com/bundle.tar.gz",
        )

        release_create_call = mock_run.call_args_list[1]
        self.assertIn("release", release_create_call.args[0])
        self.assertIn("create", release_create_call.args[0])

        release_upload_call = mock_run.call_args_list[2]
        self.assertIn("--clobber", release_upload_call.args[0])
        self.assertIn(str(asset_path), release_upload_call.args[0])


if __name__ == "__main__":
    unittest.main()
