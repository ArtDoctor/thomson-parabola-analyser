from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from oblisk.analysis.geometry import PerspectiveReference
from oblisk.analysis.spectra import (
    SpectraResult,
    SpectrumGeometry,
    plot_sampling_overlay,
    plot_single_numbered_log_spectra,
    plot_spectra_linear_energy_logy,
)
from oblisk.plotting import plot_classified_rot


def save_classification_and_spectra_plots(
    save_plots: bool,
    plot_path_for: Callable[[str], Path | None],
    classified: list[dict[str, Any]],
    hydrogen_line: dict[str, Any],
    xp_plot_min: float,
    xp_plot_max: float,
    opened: np.ndarray,
    x0_fit: float,
    y0_fit: float,
    theta_fit: float,
    gamma_fit: float,
    delta_fit: float,
    k1_fit: float,
    k2_fit: float,
    img_center: tuple[float, float],
    img_diag: float,
    perspective_reference: PerspectiveReference,
    test_img: np.ndarray,
    spectra_result: SpectraResult,
    spectra_geometry: SpectrumGeometry,
) -> None:
    if not save_plots:
        return
    plot_classified_rot(
        classified=classified,
        hydrogen_line=hydrogen_line,
        title="Merged + Classified (dashed fits only, rotation-aware)",
        Xp_span=(xp_plot_min, xp_plot_max),
        image=opened,
        x0_fit=x0_fit,
        y0_fit=y0_fit,
        theta_fit=theta_fit,
        gamma_fit=gamma_fit,
        delta_fit=delta_fit,
        save_path=plot_path_for("11_classified"),
        k1_fit=k1_fit,
        k2_fit=k2_fit,
        img_center=img_center,
        img_diag=img_diag,
        perspective_reference=perspective_reference,
    )
    plot_sampling_overlay(
        image=test_img,
        result=spectra_result,
        geometry=spectra_geometry,
        title=(
            "Parabola samples (used for spectra) over ORIGINAL image "
            "+ background region"
        ),
        save_path=plot_path_for("12_sampling_overlay"),
    )
    plot_single_numbered_log_spectra(
        spectra_result,
        print_caption_map=True,
        save_path=plot_path_for("15_numbered_log_spectra"),
    )
    plot_spectra_linear_energy_logy(
        spectra_result,
        smoothing_sigma_bins=5.0,
        edge_sigma_keep=0.5,
        clip_min=1.0,
        save_path=plot_path_for("16_linear_energy_logy"),
    )
