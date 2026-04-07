from pathlib import Path

import pytest
import torch

import oblisk.web_service as web_service
from oblisk.analysis.species import DEFAULT_CLASSIFICATION_ELEMENTS


def test_build_web_settings_matches_web_defaults() -> None:
    settings = web_service.build_web_settings(inner_margin_crop=True)

    assert settings.denoise is True
    assert settings.window == 15
    assert settings.prominence == 5
    assert settings.max_peak_distance == 30
    assert settings.inner_margin_crop_px == (
        web_service.WEB_INNER_MARGIN_CROP_PX
    )
    assert settings.classification_element_symbols == list(
        DEFAULT_CLASSIFICATION_ELEMENTS,
    )

    no_margin_settings = web_service.build_web_settings(
        inner_margin_crop=False,
    )
    assert no_margin_settings.inner_margin_crop_px == 0


def test_build_web_settings_custom_classification_elements() -> None:
    settings = web_service.build_web_settings(
        inner_margin_crop=True,
        classification_element_symbols=["N", "Ne"],
    )
    assert settings.classification_element_symbols == ["N", "Ne"]


def test_run_web_analysis_delegates_to_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}

    def fake_run(**kwargs: object) -> dict[str, bool]:
        calls.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(web_service, "run", fake_run)

    result = web_service.run_web_analysis(
        image_path=tmp_path / "input.tif",
        output_dir=tmp_path / "out",
        use_denoise_unet=False,
        inner_margin_crop=False,
    )

    assert result == {"ok": True}
    assert calls["add_plots"] is True
    assert calls["use_denoise_unet"] is False
    assert calls["image_path"] == tmp_path / "input.tif"
    assert calls["output_dir"] == tmp_path / "out"
    settings = calls["settings"]
    assert isinstance(settings, web_service.Settings)
    assert settings.inner_margin_crop_px == 0
    assert settings.classification_element_symbols == list(
        DEFAULT_CLASSIFICATION_ELEMENTS,
    )


def test_preload_web_models_passes_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_preload_unet_denoiser(
        *,
        device: torch.device | None = None,
        checkpoint_path: Path | None = None,
    ) -> None:
        calls["device"] = device
        calls["checkpoint_path"] = checkpoint_path

    monkeypatch.setattr(
        web_service,
        "preload_unet_denoiser",
        fake_preload_unet_denoiser,
    )

    device = torch.device("cpu")
    web_service.preload_web_models(device=device)

    assert calls["device"] == device
    assert calls["checkpoint_path"] is None
