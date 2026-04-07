import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from oblisk.analysis.curvature_peaks import (
    compute_peak_half_width_bounds,
    deduplicate_close_curvature_peaks,
    detect_good_a_values,
    plot_a_score_and_peaks,
)
from oblisk.analysis.geometry import PerspectiveReference
from oblisk.analysis.geometry_fit_sampling import score_parabolas_over_a
from oblisk.reporting.pipeline_log import CurvaturePeaksLog


def run_curvature_score_and_peak_detection(
    opened: np.ndarray,
    filtered_lines: list[list[list[int]]],
    x0_fit: float,
    y0_fit: float,
    theta_fit: float,
    gamma_fit: float,
    delta_fit: float,
    a_list: np.ndarray,
    k1_fit: float,
    k2_fit: float,
    img_center: tuple[float, float],
    img_diag: float,
    perspective_reference: PerspectiveReference,
    save_plots: bool,
    plot_path_for: Callable[[str], Path | None],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    CurvaturePeaksLog,
]:
    t0 = time.perf_counter()
    a_grid, scores = score_parabolas_over_a(
        image=opened,
        filtered_lines=filtered_lines,
        x0_fit=x0_fit,
        y0_fit=y0_fit,
        theta_fit=theta_fit,
        gamma_fit=gamma_fit,
        delta_fit=delta_fit,
        a_list_initial=a_list,
        n_a=600,
        a_padding_factor=0.5,
        n_samples_per_parabola=1000,
        Xp_margin=50.0,
        k1_fit=k1_fit,
        k2_fit=k2_fit,
        img_center=img_center,
        img_diag=img_diag,
        perspective_reference=perspective_reference,
    )
    elapsed_score = time.perf_counter() - t0

    peak_prominence_rel = 0.025
    peak_height_rel = 0.008
    peak_min_distance = 10
    peak_floor_rel = 0.10
    peak_floor_abs = 2.5
    peak_dedup_rel_sep = 0.032

    t1 = time.perf_counter()
    good_a, peak_indices = detect_good_a_values(
        a_grid,
        scores,
        prominence_rel=peak_prominence_rel,
        height_rel=peak_height_rel,
        min_distance=peak_min_distance,
        floor_margin_rel=peak_floor_rel,
        floor_margin_abs=peak_floor_abs,
    )
    good_a, peak_indices = deduplicate_close_curvature_peaks(
        a_grid,
        scores,
        peak_indices,
        rel_sep=peak_dedup_rel_sep,
    )
    peak_left_a, peak_right_a, half_height_scores = compute_peak_half_width_bounds(
        a_grid,
        scores,
        peak_indices,
    )
    integration_windows_a = (
        np.column_stack((peak_left_a, peak_right_a))
        if len(peak_left_a) > 0
        else np.empty((0, 2), dtype=float)
    )
    elapsed_detect = time.perf_counter() - t1

    if save_plots:
        plot_a_score_and_peaks(
            a_grid,
            scores,
            good_a,
            peak_indices,
            peak_left_a=peak_left_a,
            peak_right_a=peak_right_a,
            half_height_scores=half_height_scores,
            a_list_fitted=a_list,
            title="Thomson parabolas intensity score vs curvature a",
            save_path=plot_path_for("09_a_score_peaks"),
            peak_floor_rel=peak_floor_rel,
            peak_floor_abs=peak_floor_abs,
        )

    peaks_log = CurvaturePeaksLog(
        good_a_values=[float(value) for value in np.asarray(good_a).ravel()],
        score_grid_size=int(len(a_grid)),
        prominence_rel=peak_prominence_rel,
        height_rel=peak_height_rel,
        min_distance=peak_min_distance,
    )

    return (
        a_grid,
        scores,
        good_a,
        peak_indices,
        peak_left_a,
        peak_right_a,
        half_height_scores,
        integration_windows_a,
        elapsed_score,
        elapsed_detect,
        peaks_log,
    )
