from typing import Any

import numpy as np

from oblisk.analysis.background import BackgroundRoi
from oblisk.analysis.geometry import PerspectiveReference
from oblisk.analysis.overlay import (
    ProjectionGeometry,
    build_classified_projected_curves,
    serialize_classified_overlay_curves,
    serialize_sampling_overlay_curves,
)
from oblisk.analysis.spectra import SpectraResult, SpectrumGeometry
from oblisk.reporting.eval_report import RunEvalPayload
from oblisk.reporting.results import build_result_json


def build_pipeline_overlays(
    classified: list[dict[str, Any]],
    spectra_result: SpectraResult,
    filtered_lines: list[list[list[int]]],
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
    xp_plot_min: float,
    xp_plot_max: float,
    test_img: np.ndarray,
    background_roi: BackgroundRoi,
) -> dict[str, Any]:
    overlay_geometry = ProjectionGeometry(
        x0_fit=float(x0_fit),
        y0_fit=float(y0_fit),
        theta_fit=float(theta_fit),
        gamma_fit=float(gamma_fit),
        delta_fit=float(delta_fit),
        k1_fit=float(k1_fit),
        k2_fit=float(k2_fit),
        img_center_x=float(img_center[0]),
        img_center_y=float(img_center[1]),
        img_diag=float(img_diag),
        perspective_reference=perspective_reference,
    )
    classified_overlay_curves = build_classified_projected_curves(
        classified=classified,
        geometry=overlay_geometry,
        xp_span=(float(xp_plot_min), float(xp_plot_max)),
        image_shape=(int(test_img.shape[0]), int(test_img.shape[1])),
        n_samples=600,
    )
    sampling_overlay_curves = serialize_sampling_overlay_curves(
        labels=[str(spectrum.label) for spectrum in spectra_result.spectra],
        polylines=[
            None
            if spectrum.polyline is None
            else (
                np.asarray(spectrum.polyline.x, dtype=float),
                np.asarray(spectrum.polyline.y, dtype=float),
            )
            for spectrum in spectra_result.spectra
        ],
    )
    return {
        "classified": {
            "curves": [
                curve.model_dump(mode="json")
                for curve in serialize_classified_overlay_curves(
                    classified_overlay_curves
                )
            ],
        },
        "sampling": {
            "background_roi": {
                "x0": int(background_roi.x0),
                "x1": int(background_roi.x1),
                "y0": int(background_roi.y0),
                "y1": int(background_roi.y1),
            },
            "curves": [
                curve.model_dump(mode="json")
                for curve in sampling_overlay_curves
            ],
        },
    }


def finalize_pipeline_result(
    classified: list[dict[str, Any]],
    spectra_result: SpectraResult,
    spectra_geometry: SpectrumGeometry,
    overlays: dict[str, Any],
    timings: dict[str, float],
    log_entries_dump: list[dict[str, Any]],
    eval_payload: RunEvalPayload,
    xp_plot_min: float,
    xp_plot_max: float,
) -> dict[str, Any]:
    return build_result_json(
        classified,
        spectra_result,
        spectra_geometry,
        overlays,
        timings,
        log_entries_dump,
        eval_payload,
        xp_plot_range=(xp_plot_min, xp_plot_max),
    )
