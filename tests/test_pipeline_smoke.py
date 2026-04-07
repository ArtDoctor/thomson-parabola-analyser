import json
import sys
from pathlib import Path

import pytest

import oblisk.processing.preprocessing as preprocessing_module
from oblisk.config import Settings
from oblisk.cli import main as cli_main
from oblisk.processing.preprocessing import preprocess_image

from tests.synthetic_pipeline_cases import (
    SYNTHETIC_DETECTOR_IMAGE,
    full_frame_coords,
)


def test_preprocess_image_smoke_on_synthetic_raw_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = SYNTHETIC_DETECTOR_IMAGE
    assert image_path.exists(), (
        f"Missing {image_path}. "
        "Run: bash synthetic_data/generate_image.sh"
    )
    monkeypatch.setattr(
        preprocessing_module,
        "cut_detector_image",
        full_frame_coords,
    )

    result = preprocess_image(
        image_path,
        Settings(),
    )

    expected_shape = (1100, 1100)
    assert result.cropped.shape == expected_shape
    assert result.opened.shape == expected_shape
    assert result.denoise_title == "UNet Denoised"
    assert len(result.log_entries) == 4
    assert {"load_crop", "denoise"} <= set(result.timings)
    y_found, x_found = result.brightest_spot
    assert 0 <= y_found < expected_shape[0]
    assert 0 <= x_found < expected_shape[1]


def test_cli_smoke_writes_res_json_for_synthetic_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_path = SYNTHETIC_DETECTOR_IMAGE
    assert image_path.exists(), (
        f"Missing {image_path}. "
        "Run: bash synthetic_data/generate_image.sh"
    )
    output_dir = tmp_path / "out"
    monkeypatch.setattr(
        preprocessing_module,
        "cut_detector_image",
        full_frame_coords,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            str(image_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    cli_main()

    result = json.loads((output_dir / "res.json").read_text())
    assert result["eval"]["species_summary"]["n_classified_parabolae"] > 0
    assert len(result["classified"]) > 0
    assert result["timings"]["total"] > 0.0
