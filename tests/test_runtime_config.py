from pathlib import Path

import pytest

from oblisk.config import (
    Settings,
    default_unet_checkpoint_path,
    default_yolo_detector_path,
    repo_root,
    runtime_model_paths,
)


def test_settings_remains_available_via_package_config() -> None:
    settings = Settings()
    assert settings.denoise is True
    assert settings.inner_margin_crop_px == 50


def test_default_model_paths_use_repo_layout_when_no_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OBLISK_YOLO_MODEL_PATH", raising=False)
    monkeypatch.delenv("OBLISK_UNET_CHECKPOINT_PATH", raising=False)

    expected_yolo = repo_root() / "yolo-tune" / "thomson-cutter.onnx"
    expected_unet = repo_root() / "unet-denoiser" / "unet_denoise_best.pth"

    assert default_yolo_detector_path() == expected_yolo
    assert default_unet_checkpoint_path() == expected_unet
    assert runtime_model_paths().yolo_detector == expected_yolo
    assert runtime_model_paths().unet_checkpoint == expected_unet


def test_default_model_paths_honor_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OBLISK_YOLO_MODEL_PATH", "/tmp/custom-yolo.onnx")
    monkeypatch.setenv("OBLISK_UNET_CHECKPOINT_PATH", "/tmp/custom-unet.pth")

    assert default_yolo_detector_path() == Path("/tmp/custom-yolo.onnx")
    assert default_unet_checkpoint_path() == Path("/tmp/custom-unet.pth")
