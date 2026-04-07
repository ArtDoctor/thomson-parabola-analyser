from pathlib import Path
from typing import Any

import torch

from oblisk.analysis.species import DEFAULT_CLASSIFICATION_ELEMENTS
from oblisk.config import Settings
from oblisk.processing.pipeline import run
from oblisk.runtime import preload_unet_denoiser

WEB_INNER_MARGIN_CROP_PX = 50


def build_web_settings(
    *,
    inner_margin_crop: bool,
    classification_element_symbols: list[str] | None = None,
    spectrometer_params: dict[str, float] | None = None,
) -> Settings:
    elems = (
        classification_element_symbols
        if classification_element_symbols is not None
        else list(DEFAULT_CLASSIFICATION_ELEMENTS)
    )
    spec_kwargs: dict[str, float | None] = {}
    if spectrometer_params:
        _map = {
            "E_kVm": "spec_E_kVm",
            "LiE_cm": "spec_LiE_cm",
            "LfE_cm": "spec_LfE_cm",
            "B_mT": "spec_B_mT",
            "LiB_cm": "spec_LiB_cm",
            "LfB_cm": "spec_LfB_cm",
            "detector_size_cm": "spec_detector_size_cm",
        }
        for src_key, dst_key in _map.items():
            val = spectrometer_params.get(src_key)
            if val is not None:
                spec_kwargs[dst_key] = float(val)
    return Settings(
        denoise=True,
        denoise_kernel_size=5,
        window=15,
        prominence=5,
        distance=10,
        max_peak_distance=30,
        min_line_length_1=10,
        min_line_length_2=90,
        max_x_gap=10,
        direction_tol_px=3,
        inner_margin_crop_px=(
            WEB_INNER_MARGIN_CROP_PX if inner_margin_crop else 0
        ),
        classification_element_symbols=elems,
        **spec_kwargs,
    )


def preload_web_models(device: torch.device | None = None) -> None:
    dev = device
    if dev is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preload_unet_denoiser(device=dev)


def run_web_analysis(
    *,
    image_path: Path,
    output_dir: Path,
    use_denoise_unet: bool,
    inner_margin_crop: bool,
    classification_element_symbols: list[str] | None = None,
    spectrometer_params: dict[str, float] | None = None,
) -> dict[str, Any]:
    return run(
        image_path=image_path,
        output_dir=output_dir,
        add_plots=True,
        settings=build_web_settings(
            inner_margin_crop=inner_margin_crop,
            classification_element_symbols=classification_element_symbols,
            spectrometer_params=spectrometer_params,
        ),
        use_denoise_unet=use_denoise_unet,
    )
