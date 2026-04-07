import collections
from collections.abc import Sequence
from typing import Any

import numpy as np

from oblisk.analysis.physics import (
    b_tesla_from_magnet_current_amps,
    get_hydrogen_a,
    magnet_calibration_sorted_matrix,
)
from oblisk.analysis.species import build_species_set
from oblisk.config import Settings

# Prefer carbon over silicon when both match tolerance but Si barely wins
# (typical for C-dominated simulator output and many carbon plasmas).
_CARBON_OVER_SILICON_REL_ERR_MARGIN = 0.12


def detect_isolated_hydrogen(good_a: np.ndarray) -> bool:
    """True if the smallest good_a peak is well separated from the next."""

    if len(good_a) < 2:
        return False
    sorted_a = np.sort(good_a)
    return sorted_a[1] >= 3.0 * sorted_a[0]


def resolve_hydrogen_a(
    good_a: np.ndarray,
    m_per_px_img: float,
    w: int,
    h: int,
    x0_fit: float,
    y0_fit: float,
    theta_fit: float,
    force_analytical: bool,
    b_tesla: float,
    diagnostic_prints: bool = True,
    detector_config: str = "B",
) -> tuple[float, str]:
    """Choose the hydrogen reference curvature in the rotated frame."""

    def _pick(a1: float, a2: float) -> float:
        return a1 if detector_config == "A" else a2

    if force_analytical:
        a1, a2 = get_hydrogen_a(
            m_per_px_img,
            w,
            h,
            x0_fit,
            y0_fit,
            theta_fit,
            b_tesla=b_tesla,
            diagnostic_prints=diagnostic_prints,
        )
        return float(_pick(a1, a2)), "analytical_forced"
    if detect_isolated_hydrogen(good_a):
        return float(np.min(good_a)), "detected_isolated_min_a"
    a1, a2 = get_hydrogen_a(
        m_per_px_img,
        w,
        h,
        x0_fit,
        y0_fit,
        theta_fit,
        b_tesla=b_tesla,
        diagnostic_prints=diagnostic_prints,
    )
    return float(_pick(a1, a2)), "analytical_smart"


def _classification_score_tuple(
    classified: list[dict[str, Any]],
) -> tuple[int, float]:
    n_unidentified = sum(
        1 for match in classified if match.get("label") == "?"
    )
    sum_rel = 0.0
    for match in classified:
        if match.get("label") == "?":
            continue
        candidates = match.get("candidates") or []
        if candidates:
            sum_rel += float(candidates[0]["rel_err"])
    return n_unidentified, sum_rel


def _maybe_prefer_carbon_over_silicon(
    candidates: list[dict[str, float | str]],
    match_tol: float,
) -> str | None:
    if not candidates:
        return None
    best = candidates[0]
    best_name = str(best["name"])
    best_rel = float(best["rel_err"])
    if not best_name.startswith("Si^"):
        return None
    carbon_candidates = [
        entry for entry in candidates if str(entry["name"]).startswith("C^")
    ]
    if not carbon_candidates:
        return None
    carbon_best = min(
        carbon_candidates,
        key=lambda entry: float(entry["rel_err"]),
    )
    carbon_rel = float(carbon_best["rel_err"])
    if carbon_rel > match_tol:
        return None
    if carbon_rel - best_rel > _CARBON_OVER_SILICON_REL_ERR_MARGIN:
        return None
    return str(carbon_best["name"])


def classify_lines(
    a_list: Sequence[float] | np.ndarray,
    a_h: float,
    match_tol: float,
    element_symbols: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    species = build_species_set(element_symbols)
    out: list[dict[str, Any]] = []
    for a in a_list:
        r = float(a) / float(a_h)
        candidates: list[dict[str, float | str]] = []
        for candidate_species in species:
            mq = float(candidate_species["m_over_q"])
            rel = abs(r - mq) / max(mq, 1e-12)
            if rel <= match_tol:
                candidates.append(
                    {
                        "name": str(candidate_species["name"]),
                        "mq_target": mq,
                        "rel_err": rel,
                    }
                )
        candidates.sort(key=lambda entry: entry["rel_err"])
        label_text = candidates[0]["name"] if candidates else "?"
        carbon_override = _maybe_prefer_carbon_over_silicon(
            candidates,
            match_tol,
        )
        if carbon_override is not None:
            label_text = carbon_override
        out.append(
            {
                "a": a,
                "mq_meas": r,
                "candidates": candidates,
                "label": label_text,
            }
        )

    def rel_err_to_species(row: dict[str, Any], name: str) -> float:
        for candidate in row["candidates"]:
            if str(candidate["name"]) == name:
                return float(candidate["rel_err"])
        return float("inf")

    by_label: dict[str, list[int]] = collections.defaultdict(list)
    for index, row in enumerate(out):
        label = row["label"]
        if label != "?":
            by_label[str(label)].append(index)
    for species_name, indices in by_label.items():
        if len(indices) <= 1:
            continue
        winner = min(
            indices,
            key=lambda index: (
                rel_err_to_species(out[index], species_name),
                index,
            ),
        )
        for index in indices:
            if index != winner:
                out[index]["label"] = "?"

    return out


def select_magnet_calibration_and_classify(
    a_list: Sequence[float] | np.ndarray,
    good_a: np.ndarray,
    m_per_px_img: float,
    w: int,
    h: int,
    x0_fit: float,
    y0_fit: float,
    theta_fit: float,
    force_analytical: bool,
    match_tol: float,
    settings: Settings,
    detector_config: str = "B",
    element_symbols: Sequence[str] | None = None,
) -> tuple[float, float, float, str, str, list[dict[str, Any]]]:
    """Choose the better magnet calibration and classify species under it."""

    calib = magnet_calibration_sorted_matrix(settings)
    trials: list[tuple[float, str]] = [
        (float(settings.magnet_current_standard_amps), "standard"),
        (float(settings.magnet_current_phosphor_amps), "phosphor"),
    ]
    best_score: tuple[int, float] | None = None
    best_magnet_i = 0.0
    best_b = 0.0
    best_ha = 0.0
    best_mode = ""
    best_label = ""
    best_classified: list[dict[str, Any]] = []

    for i_amps, cal_label in trials:
        b = b_tesla_from_magnet_current_amps(i_amps, calib)
        ha, mode = resolve_hydrogen_a(
            good_a,
            m_per_px_img,
            w,
            h,
            x0_fit,
            y0_fit,
            theta_fit,
            force_analytical=force_analytical,
            b_tesla=b,
            diagnostic_prints=False,
            detector_config=detector_config,
        )
        classified = classify_lines(
            a_list,
            ha,
            match_tol,
            element_symbols=element_symbols,
        )
        score = _classification_score_tuple(classified)
        if best_score is None or score < best_score:
            best_score = score
            best_magnet_i = i_amps
            best_b = b
            best_ha = ha
            best_mode = mode
            best_label = cal_label
            best_classified = classified

    if force_analytical or not detect_isolated_hydrogen(
        good_a,
    ):
        get_hydrogen_a(
            m_per_px_img,
            w,
            h,
            x0_fit,
            y0_fit,
            theta_fit,
            b_tesla=best_b,
            diagnostic_prints=True,
        )

    return (
        best_magnet_i,
        best_b,
        best_ha,
        best_mode,
        best_label,
        best_classified,
    )
