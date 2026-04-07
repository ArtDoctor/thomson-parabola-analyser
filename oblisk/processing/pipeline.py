import math
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel

from oblisk.analysis.geometry import dominant_xp_sign_from_points, xp_span_px_from_image
from oblisk.analysis.geometry_fit import perspective_reference_from_lines
from oblisk.analysis.geometry_fit_global_origin import fit_global_origin_with_rotation
from oblisk.analysis.geometry_fit_line_metrics import per_line_parabola_fit_errors
from oblisk.analysis.geometry_fit_sampling import get_global_Xp_range_from_lines
from oblisk.analysis.line_merging import merge_lines
from oblisk.analysis.parabola_plotting import (
    plot_detected_parabolas,
    plot_lines_rotated,
)
from oblisk.analysis.spectra import (
    BackgroundRoi,
    SpectrumGeometry,
    build_spectra_result,
)
from oblisk.config import Settings
from oblisk.plot_display import show_or_save
from oblisk.plotting import (
    plot_grayscale_panel,
    plot_lines_with_labels,
    plot_peaks_overlay,
)
from oblisk.processing.line_processing import smooth_lines
from oblisk.processing.preprocessing import PreprocessedImage, preprocess_image
from oblisk.reporting.eval_report import (
    ParabolaLineFitEvalRow,
    build_run_eval,
)
from oblisk.reporting.pipeline_log import (
    DetectionSettingsLog,
    GlobalFitLog,
    LinePipelineStatsLog,
    PeakScanLog,
    RunFlagsLog,
    SpectraIntegrationLog,
    XpSpanLog,
    dump_entries,
)
from oblisk.processing.pipeline_classification_stage import (
    run_classification_and_xp_span,
)
from oblisk.processing.pipeline_curvature_peaks import (
    run_curvature_score_and_peak_detection,
)
from oblisk.processing.pipeline_finalize import (
    build_pipeline_overlays,
    finalize_pipeline_result,
)
from oblisk.processing.pipeline_stage_plots import save_classification_and_spectra_plots
from oblisk.analysis.trace_detection import build_lines, extract_peaks


def run(
    image_path: Path,
    output_dir: Path,
    add_plots: bool,
    settings: Settings,
    use_experimental_a_for_hydrogen: bool = False,
    use_denoise_unet: bool = True,
    add_plots_full: bool = False,
    preprocessed: PreprocessedImage | None = None,
) -> dict[str, Any]:
    save_plots = add_plots or add_plots_full
    plot_dir: Path | None = output_dir / "plots" if save_plots else None
    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)

    def _p(name: str) -> Path | None:
        return (plot_dir / f"{name}.png") if plot_dir else None

    timings: dict[str, float] = {}
    log_entries: list[BaseModel] = []

    log_entries.append(
        RunFlagsLog(
            use_experimental_a_for_hydrogen=use_experimental_a_for_hydrogen,
            use_denoise_unet=use_denoise_unet,
        )
    )
    log_entries.append(
        DetectionSettingsLog(
            denoise=settings.denoise,
            denoise_kernel_size=settings.denoise_kernel_size,
            window=settings.window,
            prominence=settings.prominence,
            distance=settings.distance,
            max_peak_distance=settings.max_peak_distance,
            min_line_length_1=settings.min_line_length_1,
            min_line_length_2=settings.min_line_length_2,
            max_x_gap=settings.max_x_gap,
            direction_tol_px=settings.direction_tol_px,
            inner_margin_crop_px=settings.inner_margin_crop_px,
        )
    )

    if preprocessed is None:
        preprocessed = preprocess_image(image_path, settings, use_denoise_unet)

    test_img = preprocessed.cropped
    opened = preprocessed.opened
    brightest_spot = preprocessed.brightest_spot
    detector_config = preprocessed.detector_config
    m_per_px_img = preprocessed.m_per_px_img
    orig_w = preprocessed.orig_w
    orig_h = preprocessed.orig_h
    denoise_title = preprocessed.denoise_title
    log_entries.extend(preprocessed.log_entries)
    timings.update(preprocessed.timings)

    image = test_img
    starting_pixel = brightest_spot[0] - 210

    if save_plots:
        plot_grayscale_panel(
            test_img,
            title="Cropped detector + standardized orientation/parity",
            save_path=_p("01_cropped_standardized"),
        )
        raw_path = _p("00_raw_cropped")
        if raw_path is not None:
            tmp_raw = raw_path.with_suffix(".tmp.png")
            cv2.imwrite(str(tmp_raw), test_img)
            tmp_raw.replace(raw_path)

    if save_plots:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original")
        plt.imshow(image, cmap="gray", origin="lower")
        plt.xlabel("x (pixels)")
        plt.ylabel("y (pixels)")
        plt.subplot(1, 2, 2)
        plt.title(denoise_title)
        plt.imshow(opened, cmap="gray", origin="lower")
        plt.xlabel("x (pixels)")
        plt.ylabel("y (pixels)")
        show_or_save(_p("02_morphological"))

    t0 = time.perf_counter()
    row_max = opened.shape[0]
    all_peaks = extract_peaks(
        image=opened,
        starting_pixel=starting_pixel,
        ending_pixel=row_max,
        settings=settings,
    )
    all_peaks = [peak for peak in all_peaks if len(peak) > 0]
    brightest_x = brightest_spot[1]
    all_peaks_cleaned = [
        [peak for peak in row_peaks if peak[0] > brightest_x + opened.shape[1] / 10]
        for row_peaks in all_peaks
    ]
    all_peaks_cleaned = [row for row in all_peaks_cleaned if len(row) > 0]
    timings["peak_extraction"] = time.perf_counter() - t0
    log_entries.append(
        PeakScanLog(
            brightest_spot_yx=(int(brightest_spot[0]), int(brightest_spot[1])),
            peak_extraction_start_row=starting_pixel,
            peak_extraction_end_row=row_max,
        )
    )

    if save_plots:
        plot_peaks_overlay(
            opened,
            all_peaks_cleaned,
            title="Row-wise intensity peaks (after brightest-column mask)",
            save_path=_p("03_peaks_overlay"),
        )

    t0 = time.perf_counter()
    filtered_lines = build_lines(
        all_peaks=all_peaks_cleaned,
        settings=settings,
        image_width=opened.shape[1],
    )
    timings["build_lines"] = time.perf_counter() - t0
    n_after_build = len(filtered_lines)

    t0 = time.perf_counter()
    merge_out = merge_lines(
        filtered_lines=filtered_lines,
        all_peaks=all_peaks_cleaned,
        settings=settings,
        opened=opened,
        brightest_spot=brightest_spot,
    )
    timings["merge_lines"] = time.perf_counter() - t0
    filtered_lines = merge_out.filtered_lines
    n_after_merge = len(filtered_lines)

    t0 = time.perf_counter()
    filtered_lines = smooth_lines(filtered_lines)
    timings["smooth_lines"] = time.perf_counter() - t0
    log_entries.append(
        LinePipelineStatsLog(
            num_lines_after_build=n_after_build,
            num_lines_after_merge=n_after_merge,
            num_lines_after_smooth=len(filtered_lines),
        )
    )
    if save_plots:
        plot_lines_with_labels(
            filtered_lines,
            opened,
            save_path=_p("05_smoothed_lines"),
        )

    x0 = brightest_spot[1]
    y0 = brightest_spot[0]

    img_h, img_w = opened.shape[:2]
    img_diag = float(np.sqrt(img_w**2 + img_h**2))
    img_center = (img_w / 2.0, img_h / 2.0)

    t0 = time.perf_counter()
    (
        x0_fit,
        y0_fit,
        theta_fit,
        gamma_fit,
        delta_fit,
        k1_fit,
        k2_fit,
        a_list,
        path,
    ) = fit_global_origin_with_rotation(
        filtered_lines,
        x0_init=x0,
        y0_init=y0,
        theta_init=0.0,
        gamma_init=0.0,
        delta_init=0.0,
        max_nfev=20000,
        fit_rotation_only=False,
        k1_init=0.0,
        k2_init=0.0,
        img_center=img_center,
        img_diag=img_diag,
    )
    timings["fit_origin"] = time.perf_counter() - t0
    log_entries.append(
        GlobalFitLog(
            x0_fit=float(x0_fit),
            y0_fit=float(y0_fit),
            theta_fit=float(theta_fit),
            gamma_fit=float(gamma_fit),
            delta_fit=float(delta_fit),
            k1_fit=float(k1_fit),
            k2_fit=float(k2_fit),
            per_line_curvatures_from_shared_vertex=[
                float(value) for value in np.asarray(a_list).ravel()
            ],
        )
    )
    perspective_reference = perspective_reference_from_lines(
        filtered_lines=filtered_lines,
        x0_fit=float(x0_fit),
        y0_fit=float(y0_fit),
        theta_fit=float(theta_fit),
        k1_fit=float(k1_fit),
        k2_fit=float(k2_fit),
        img_center=img_center,
        img_diag=img_diag,
    )

    line_fit_dicts, global_line_rmse = per_line_parabola_fit_errors(
        filtered_lines,
        x0_fit,
        y0_fit,
        theta_fit,
        gamma_fit,
        delta_fit,
        k1_fit=k1_fit,
        k2_fit=k2_fit,
        img_center=img_center,
        img_diag=img_diag,
        perspective_reference=perspective_reference,
    )
    line_fit_rows = [
        ParabolaLineFitEvalRow(
            line_index=int(row["line_index"]),
            n_points=int(row["n_points"]),
            rmse_y_eff_px=float(row["rmse_y_eff_px"]),
            mean_abs_residual_y_eff_px=float(row["mean_abs_residual_y_eff_px"]),
            rmse_y_eff_weighted_px=float(row["rmse_y_eff_weighted_px"]),
        )
        for row in line_fit_dicts
    ]
    global_line_rmse_json: float | None = (
        float(global_line_rmse) if math.isfinite(global_line_rmse) else None
    )

    if save_plots:
        if add_plots_full:
            plot_lines_rotated(
                filtered_lines=filtered_lines,
                x0=x0,
                y0=y0,
                title=(
                    "Filtered & Merged Lines with Y' = a_i * X_eff^2 "
                    "(shared vertex, rotation, tilt)"
                ),
                image=opened,
                x0_fit=x0_fit,
                y0_fit=y0_fit,
                theta_fit=theta_fit,
                gamma_fit=gamma_fit,
                delta_fit=delta_fit,
                a_list=a_list,
                path=path,
                save_paths=[
                    _p("06_rotated_with_bg"),
                    _p("07_rotated_nobg"),
                    _p("08_fit_rmse_vs_iter"),
                ],
                output_indices=[0, 1, 2],
                k1_fit=k1_fit,
                k2_fit=k2_fit,
                img_center=img_center,
                img_diag=img_diag,
                perspective_reference=perspective_reference,
            )
        else:
            plot_lines_rotated(
                filtered_lines=filtered_lines,
                x0=x0,
                y0=y0,
                title=(
                    "Filtered & Merged Lines with Y' = a_i * X_eff^2 "
                    "(shared vertex, rotation, tilt)"
                ),
                image=opened,
                x0_fit=x0_fit,
                y0_fit=y0_fit,
                theta_fit=theta_fit,
                gamma_fit=gamma_fit,
                delta_fit=delta_fit,
                a_list=a_list,
                path=path,
                save_paths=[None, _p("07_rotated_nobg"), None],
                output_indices=[1],
                k1_fit=k1_fit,
                k2_fit=k2_fit,
                img_center=img_center,
                img_diag=img_diag,
                perspective_reference=perspective_reference,
            )

    (
        _a_grid,
        _scores,
        good_a,
        _peak_indices,
        _peak_left_a,
        _peak_right_a,
        _half_height_scores,
        integration_windows_a,
        elapsed_score,
        elapsed_detect,
        peaks_log,
    ) = run_curvature_score_and_peak_detection(
        opened=opened,
        filtered_lines=filtered_lines,
        x0_fit=x0_fit,
        y0_fit=y0_fit,
        theta_fit=theta_fit,
        gamma_fit=gamma_fit,
        delta_fit=delta_fit,
        a_list=a_list,
        k1_fit=k1_fit,
        k2_fit=k2_fit,
        img_center=img_center,
        img_diag=img_diag,
        perspective_reference=perspective_reference,
        save_plots=save_plots,
        plot_path_for=_p,
    )
    timings["score_parabolas"] = elapsed_score
    timings["detect_good_a"] = elapsed_detect
    log_entries.append(peaks_log)

    t0 = time.perf_counter()
    Xp_min, Xp_max = get_global_Xp_range_from_lines(
        filtered_lines,
        x0_fit,
        y0_fit,
        theta_fit,
        margin=50.0,
        a_values=good_a,
        image=opened,
        gamma_fit=gamma_fit,
        delta_fit=delta_fit,
        perspective_reference=perspective_reference,
    )
    timings["get_xp_range"] = time.perf_counter() - t0
    log_entries.append(XpSpanLog(xp_min=float(Xp_min), xp_max=float(Xp_max)))
    Xp_plot_min, Xp_plot_max = xp_span_px_from_image(
        image=opened,
        x0_fit=x0_fit,
        y0_fit=y0_fit,
        theta_fit=theta_fit,
    )
    xp_plot_sign = dominant_xp_sign_from_points(
        points_list=[np.asarray(line, dtype=float) for line in filtered_lines],
        x0_fit=x0_fit,
        y0_fit=y0_fit,
        theta_fit=theta_fit,
    )
    if xp_plot_sign >= 0:
        Xp_plot_min = max(0.0, Xp_plot_min)
    else:
        Xp_plot_max = min(0.0, Xp_plot_max)

    if save_plots:
        plot_detected_parabolas(
            image=opened,
            good_a=good_a,
            x0_fit=x0_fit,
            y0_fit=y0_fit,
            theta_fit=theta_fit,
            gamma_fit=gamma_fit,
            delta_fit=delta_fit,
            Xp_min=Xp_plot_min,
            Xp_max=Xp_plot_max,
            n_samples_per_parabola=1000,
            title="Detected Thomson parabolas (intensity-based search over a)",
            save_path=_p("10_detected_parabolas"),
            k1_fit=k1_fit,
            k2_fit=k2_fit,
            img_center=img_center,
            img_diag=img_diag,
            perspective_reference=perspective_reference,
        )

    a_list = good_a
    (
        classified,
        hydrogen_line,
        xp_span,
        match_tol_pct,
        ref_log,
        classify_elapsed,
        b_field_chosen_t,
    ) = run_classification_and_xp_span(
        good_a,
        m_per_px_img,
        orig_w,
        orig_h,
        x0_fit,
        y0_fit,
        theta_fit,
        gamma_fit,
        delta_fit,
        filtered_lines,
        opened,
        detector_config,
        settings,
        use_experimental_a_for_hydrogen,
    )
    timings["classify"] = classify_elapsed
    log_entries.append(ref_log)

    eval_payload = build_run_eval(
        line_fit_rows,
        global_line_rmse_json,
        classified,
        bright_spot_yx=(int(brightest_spot[0]), int(brightest_spot[1])),
        peak_extraction_start_row=int(starting_pixel),
    )

    spectra_geometry = SpectrumGeometry(
        x0_fit=x0_fit,
        y0_fit=y0_fit,
        theta_fit=theta_fit,
        gamma_fit=gamma_fit,
        delta_fit=delta_fit,
        k1_fit=k1_fit,
        k2_fit=k2_fit,
        img_center_x=img_center[0],
        img_center_y=img_center[1],
        img_diag=img_diag,
        meters_per_pixel=m_per_px_img,
        b_field_t=float(b_field_chosen_t),
        perspective_reference=perspective_reference,
    )

    integration_a_samples = 33
    background_roi = BackgroundRoi(
        x0=test_img.shape[1] // 10,
        x1=test_img.shape[1] * 4 // 10,
        y0=test_img.shape[0] * 6 // 10,
        y1=test_img.shape[0] * 9 // 10,
    )
    t0 = time.perf_counter()
    spectra_result = build_spectra_result(
        image=test_img,
        classified=classified,
        geometry=spectra_geometry,
        match_tol=match_tol_pct,
        background_roi=background_roi,
        xp_bounds_px=xp_span,
        integration_windows_a=integration_windows_a,
        integration_a_samples=integration_a_samples,
    )
    timings["build_spectra"] = time.perf_counter() - t0
    log_entries.append(
        SpectraIntegrationLog(
            integration_a_samples=integration_a_samples,
            integration_windows_a=(
                integration_windows_a.tolist()
                if integration_windows_a.size
                else []
            ),
            background_roi_x0=background_roi.x0,
            background_roi_x1=background_roi.x1,
            background_roi_y0=background_roi.y0,
            background_roi_y1=background_roi.y1,
        )
    )

    save_classification_and_spectra_plots(
        save_plots=save_plots,
        plot_path_for=_p,
        classified=classified,
        hydrogen_line=hydrogen_line,
        xp_plot_min=Xp_plot_min,
        xp_plot_max=Xp_plot_max,
        opened=opened,
        x0_fit=x0_fit,
        y0_fit=y0_fit,
        theta_fit=theta_fit,
        gamma_fit=gamma_fit,
        delta_fit=delta_fit,
        k1_fit=k1_fit,
        k2_fit=k2_fit,
        img_center=img_center,
        img_diag=img_diag,
        perspective_reference=perspective_reference,
        test_img=test_img,
        spectra_result=spectra_result,
        spectra_geometry=spectra_geometry,
    )

    overlays = build_pipeline_overlays(
        classified=classified,
        spectra_result=spectra_result,
        filtered_lines=filtered_lines,
        x0_fit=x0_fit,
        y0_fit=y0_fit,
        theta_fit=theta_fit,
        gamma_fit=gamma_fit,
        delta_fit=delta_fit,
        k1_fit=k1_fit,
        k2_fit=k2_fit,
        img_center=img_center,
        img_diag=img_diag,
        perspective_reference=perspective_reference,
        xp_plot_min=Xp_plot_min,
        xp_plot_max=Xp_plot_max,
        test_img=test_img,
        background_roi=background_roi,
    )

    return finalize_pipeline_result(
        classified,
        spectra_result,
        spectra_geometry,
        overlays,
        timings,
        dump_entries(log_entries),
        eval_payload,
        Xp_plot_min,
        Xp_plot_max,
    )
