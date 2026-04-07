from pathlib import Path
from typing import Any

import pytest

import oblisk.processing.preprocessing as preprocessing_module
from oblisk.processing.pipeline import run

from tests.synthetic_pipeline_cases import (
    SYNTHETIC_DETECTOR_IMAGE,
    full_frame_coords,
    synthetic_settings,
)

REQUIRED_CARBON_LABELS = frozenset({"C^1+", "C^2+", "C^3+"})


def _non_unknown_labels(result: dict[str, Any]) -> set[str]:
    classified = result["classified"]
    return {
        str(entry["label"])
        for entry in classified
        if str(entry["label"]) != "?"
    }


def test_synthetic_detector_image_finds_c1_c2_c3(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    if not SYNTHETIC_DETECTOR_IMAGE.exists():
        pytest.fail(
            f"Missing {SYNTHETIC_DETECTOR_IMAGE}. "
            "Generate it with: bash synthetic_data/generate_image.sh"
        )

    monkeypatch.setattr(
        preprocessing_module,
        "cut_detector_image",
        full_frame_coords,
    )

    result = run(
        image_path=SYNTHETIC_DETECTOR_IMAGE,
        output_dir=tmp_path / "detector_image_run",
        add_plots=False,
        settings=synthetic_settings(),
    )

    labels = _non_unknown_labels(result)
    missing = REQUIRED_CARBON_LABELS - labels
    need = sorted(REQUIRED_CARBON_LABELS)
    got = sorted(labels)
    assert not missing, (
        f"Expected classified labels to include {need}; "
        f"missing {sorted(missing)}; got {got}"
    )
