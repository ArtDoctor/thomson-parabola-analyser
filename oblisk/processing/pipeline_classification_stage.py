import logging
import time
from typing import Any

import numpy as np

from oblisk.analysis.classification import (
    detect_isolated_hydrogen,
    select_magnet_calibration_and_classify,
)
from oblisk.analysis.physics import make_Xp_span_rot
from oblisk.config import Settings
from oblisk.reporting.pipeline_log import HydrogenReferenceLog

logger = logging.getLogger(__name__)


def run_classification_and_xp_span(
    good_a: np.ndarray,
    m_per_px_img: float,
    orig_w: int,
    orig_h: int,
    x0_fit: float,
    y0_fit: float,
    theta_fit: float,
    gamma_fit: float,
    delta_fit: float,
    filtered_lines: list[list[list[int]]],
    opened: np.ndarray,
    detector_config: str,
    settings: Settings,
    use_experimental_a_for_hydrogen: bool,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    np.ndarray,
    float,
    HydrogenReferenceLog,
    float,
    float,
]:
    match_tol_pct = 0.18
    t0 = time.perf_counter()
    (
        magnet_i_amps,
        b_field_chosen_t,
        hydrogen_a,
        hydrogen_mode,
        magnet_cal_label,
        classified,
    ) = select_magnet_calibration_and_classify(
        good_a,
        good_a,
        m_per_px_img,
        orig_w,
        orig_h,
        x0_fit,
        y0_fit,
        theta_fit,
        use_experimental_a_for_hydrogen,
        match_tol_pct,
        settings,
        detector_config,
        settings.classification_element_symbols,
    )

    cfg_label = "15×15 cm" if detector_config == "A" else "6×6 cm"
    if hydrogen_mode == "detected_isolated_min_a":
        logger.debug(
            "Hydrogen a: auto-detected isolated low-a peak "
            "(far from others), using min(good_a)"
        )
    elif hydrogen_mode == "analytical_forced":
        logger.debug(
            "Hydrogen a: analytical config %s (%s) (--use-experimental)",
            detector_config,
            cfg_label,
        )
    else:
        logger.debug(
            "Hydrogen a: analytical config %s (%s) "
            "(peaks not isolated; default smart mode)",
            detector_config,
            cfg_label,
        )
    logger.debug(
        "Magnet calibration: %s (I=%.3g A, B=%.5f T)",
        magnet_cal_label,
        magnet_i_amps,
        b_field_chosen_t,
    )

    ref_log = HydrogenReferenceLog(
        hydrogen_a=float(hydrogen_a),
        mode=hydrogen_mode,
        classification_match_tolerance=match_tol_pct,
        isolated_lowest_a_peak=detect_isolated_hydrogen(good_a),
        magnet_current_amps=float(magnet_i_amps),
        magnet_calibration=str(magnet_cal_label),
    )

    xp_span = make_Xp_span_rot(
        [{"points": np.asarray(line)} for line in filtered_lines],
        opened,
        x0_fit,
        y0_fit,
        theta_fit,
        gamma_fit,
        delta_fit,
        pad=0.0,
    )
    hydrogen_line: dict[str, Any] = {
        "idxs": ["H_ref"],
        "points": None,
        "a": float(hydrogen_a),
    }
    elapsed = time.perf_counter() - t0

    return (
        classified,
        hydrogen_line,
        xp_span,
        match_tol_pct,
        ref_log,
        elapsed,
        float(b_field_chosen_t),
    )
