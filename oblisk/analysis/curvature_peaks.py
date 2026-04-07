from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

from oblisk.plot_display import show_or_save


def _split_merged_peaks(
    a_grid: np.ndarray,
    scores: np.ndarray,
    peak_indices: np.ndarray,
    valley_frac: float = 0.6,
) -> np.ndarray:
    """
    Split peaks that were merged due to min_distance but have a significant valley.
    For each peak, look for a local minimum in its neighborhood; if it dips below
    valley_frac * peak_score, find sub-peaks and add them instead of the single peak.
    """
    if len(peak_indices) == 0:
        return peak_indices

    sorted_peaks = np.sort(peak_indices)
    n = len(scores)
    refined: list[int] = []

    for i, peak_idx in enumerate(sorted_peaks):
        peak_score = float(scores[peak_idx])
        left_bound = int(sorted_peaks[i - 1]) + 1 if i > 0 else 0
        right_bound = (
            int(sorted_peaks[i + 1]) - 1
            if i < len(sorted_peaks) - 1
            else n - 1
        )
        right_bound = max(left_bound, min(right_bound, n - 1))
        segment = scores[left_bound:right_bound + 1]
        if segment.size < 3:
            refined.append(peak_idx)
            continue
        valley_local = int(np.argmin(segment))
        valley_idx = left_bound + valley_local
        valley_thresh = valley_frac * peak_score
        if scores[valley_idx] >= valley_thresh:
            refined.append(peak_idx)
            continue
        sub_peaks, _ = find_peaks(
            segment,
            prominence=max(1e-9, 0.1 * (peak_score - float(scores[valley_idx]))),
            distance=1,
        )
        sub_peaks_global = (sub_peaks + left_bound).astype(int)
        if len(sub_peaks_global) >= 2:
            for sub_peak in sub_peaks_global:
                if scores[sub_peak] >= valley_thresh:
                    refined.append(sub_peak)
        else:
            refined.append(peak_idx)

    return np.unique(np.array(refined, dtype=int))


def compute_peak_score_floor(
    scores: np.ndarray,
    floor_margin_rel: float = 0.12,
    floor_margin_abs: float = 3.0,
) -> float:
    score_min = float(np.min(scores))
    score_max = float(np.max(scores))
    score_span = max(score_max - score_min, 1e-12)
    required_delta = max(
        0.0,
        float(floor_margin_rel) * score_span,
        float(floor_margin_abs),
    )
    return score_min + required_delta


def detect_good_a_values(
    a_grid: np.ndarray,
    scores: np.ndarray,
    prominence_rel: float = 0.05,
    height_rel: float = 0.02,
    min_distance: int | None = None,
    width_range: tuple[float, float] | None = None,
    split_merged_peaks: bool = True,
    floor_margin_rel: float = 0.12,
    floor_margin_abs: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find peaks in the score(a) curve.

    prominence_rel and height_rel are relative to score range.
    Peaks must also rise above the global score floor by at least
    max(floor_margin_rel * score_span, floor_margin_abs).
    """
    if len(a_grid) != len(scores):
        raise ValueError("a_grid and scores must have same length")

    score_min = float(np.min(scores))
    score_max = float(np.max(scores))
    score_span = max(score_max - score_min, 1e-12)
    prominence = prominence_rel * score_span
    height = score_min + height_rel * score_span

    kwargs: dict = {"prominence": prominence, "height": height}
    if min_distance is not None:
        kwargs["distance"] = min_distance
    if width_range is not None:
        kwargs["width"] = width_range

    peaks, _ = find_peaks(scores, **kwargs)

    if split_merged_peaks and len(peaks) > 0:
        peaks = _split_merged_peaks(a_grid, scores, peaks, valley_frac=0.6)

    if len(peaks) > 0:
        score_floor = compute_peak_score_floor(
            scores,
            floor_margin_rel=floor_margin_rel,
            floor_margin_abs=floor_margin_abs,
        )
        peaks = peaks[scores[peaks] >= score_floor]

    good_a = a_grid[peaks]
    return good_a, peaks


def deduplicate_close_curvature_peaks(
    a_grid: np.ndarray,
    scores: np.ndarray,
    peak_indices: np.ndarray,
    rel_sep: float = 0.038,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Drop peaks whose curvature a is nearly identical to a higher-scoring peak
    (relative difference <= rel_sep), which otherwise draws duplicate overlays.
    """
    if peak_indices.size == 0:
        return np.array([], dtype=float), np.array([], dtype=int)
    pk = np.asarray(peak_indices, dtype=int)
    peak_scores = scores[pk]
    order = np.argsort(-peak_scores)
    kept_list: list[int] = []
    kept_a_list: list[float] = []
    for order_index in order:
        idx = int(pk[int(order_index)])
        a_val = float(a_grid[idx])
        scale = max(abs(a_val), 1e-15)
        if all(abs(a_val - kept_a) / max(scale, abs(kept_a), 1e-15) > rel_sep for kept_a in kept_a_list):
            kept_list.append(idx)
            kept_a_list.append(a_val)
    kept_arr = np.array(sorted(kept_list), dtype=int)
    return a_grid[kept_arr], kept_arr


def compute_peak_half_width_bounds(
    a_grid: np.ndarray,
    scores: np.ndarray,
    peak_indices: np.ndarray,
    rel_height: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate left/right peak bounds in `a` at the requested relative height.

    The height is measured from the peak top down to an absolute zero baseline,
    not relative to the local prominence. For `rel_height=0.5`, the target is
    therefore the classic half-maximum level `0.5 * peak_score`.

    When neighboring peaks are close enough that the half-maximum crossing would
    extend into the adjacent peak, the bound is clamped to the minimum score
    between the two peaks so the integration windows stay disjoint.
    """
    if len(a_grid) != len(scores):
        raise ValueError("a_grid and scores must have same length")
    if len(peak_indices) == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty

    if rel_height < 0.0 or rel_height > 1.0:
        raise ValueError("rel_height must be in the interval [0, 1].")

    peak_indices = np.asarray(peak_indices, dtype=int)
    order = np.argsort(peak_indices)
    sorted_peaks = peak_indices[order]

    def _interp_crossing(left_idx: int, right_idx: int, target: float) -> float:
        y_left = float(scores[left_idx])
        y_right = float(scores[right_idx])
        a_left = float(a_grid[left_idx])
        a_right = float(a_grid[right_idx])
        if np.isclose(y_left, y_right):
            return 0.5 * (a_left + a_right)
        frac = (target - y_left) / (y_right - y_left)
        frac = float(np.clip(frac, 0.0, 1.0))
        return a_left + frac * (a_right - a_left)

    def _search_left_crossing(peak_idx: int, limit_idx: int, target: float) -> float:
        for left_idx in range(peak_idx - 1, limit_idx - 1, -1):
            right_idx = left_idx + 1
            if scores[left_idx] <= target <= scores[right_idx]:
                return _interp_crossing(left_idx, right_idx, target)
        return float(a_grid[limit_idx])

    def _search_right_crossing(peak_idx: int, limit_idx: int, target: float) -> float:
        for right_idx in range(peak_idx + 1, limit_idx + 1):
            left_idx = right_idx - 1
            if scores[right_idx] <= target <= scores[left_idx]:
                return _interp_crossing(left_idx, right_idx, target)
        return float(a_grid[limit_idx])

    left_limits = np.zeros(len(sorted_peaks), dtype=int)
    right_limits = np.full(len(sorted_peaks), len(a_grid) - 1, dtype=int)

    for i in range(len(sorted_peaks) - 1):
        left_peak = int(sorted_peaks[i])
        right_peak = int(sorted_peaks[i + 1])
        valley_offset = int(np.argmin(scores[left_peak:right_peak + 1]))
        valley_idx = left_peak + valley_offset
        right_limits[i] = valley_idx
        left_limits[i + 1] = valley_idx

    left_sorted = np.zeros(len(sorted_peaks), dtype=float)
    right_sorted = np.zeros(len(sorted_peaks), dtype=float)
    height_sorted = np.zeros(len(sorted_peaks), dtype=float)

    for i, peak_idx in enumerate(sorted_peaks):
        peak_score = float(scores[peak_idx])
        target = (1.0 - rel_height) * peak_score
        height_sorted[i] = target
        left_sorted[i] = _search_left_crossing(
            peak_idx=peak_idx,
            limit_idx=int(left_limits[i]),
            target=target,
        )
        right_sorted[i] = _search_right_crossing(
            peak_idx=peak_idx,
            limit_idx=int(right_limits[i]),
            target=target,
        )

    left_a = np.zeros(len(peak_indices), dtype=float)
    right_a = np.zeros(len(peak_indices), dtype=float)
    height_eval = np.zeros(len(peak_indices), dtype=float)
    left_a[order] = left_sorted
    right_a[order] = right_sorted
    height_eval[order] = height_sorted
    return left_a, right_a, height_eval


def plot_a_score_and_peaks(
    a_grid: np.ndarray,
    scores: np.ndarray,
    good_a: np.ndarray,
    peak_indices: np.ndarray,
    peak_left_a: np.ndarray | None = None,
    peak_right_a: np.ndarray | None = None,
    half_height_scores: np.ndarray | None = None,
    a_list_fitted: np.ndarray | None = None,
    title: str = "Score vs a",
    save_path: Path | None = None,
    peak_floor_rel: float = 0.12,
    peak_floor_abs: float = 3.0,
) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(a_grid, scores, label="score(a)", linewidth=1.5)
    peak_score_floor = compute_peak_score_floor(
        scores,
        floor_margin_rel=peak_floor_rel,
        floor_margin_abs=peak_floor_abs,
    )
    if peak_score_floor > float(np.min(scores)) + 1e-12:
        plt.axhline(
            peak_score_floor,
            color="tab:red",
            linestyle=":",
            linewidth=1.5,
            alpha=0.88,
            label=f"peak floor ({peak_score_floor:.2f})",
        )
    if len(good_a) > 0:
        plt.plot(
            a_grid[peak_indices],
            scores[peak_indices],
            "rx",
            markersize=10,
            markeredgewidth=2,
            label=f"detected peaks ({len(good_a)})",
        )
    has_half_width_data = peak_left_a is not None and peak_right_a is not None and half_height_scores is not None
    if has_half_width_data:
        left_a_vals = peak_left_a
        right_a_vals = peak_right_a
        half_height_vals = half_height_scores
        assert left_a_vals is not None and right_a_vals is not None and half_height_vals is not None
        n_peaks = len(peak_indices)
        if all(len(v) == n_peaks for v in (left_a_vals, right_a_vals, half_height_vals)):
            for i, (left_a, right_a, half_height) in enumerate(
                zip(left_a_vals, right_a_vals, half_height_vals)
            ):
                hlines_kw = {"label": "half-width window"} if i == 0 else {}
                plt.hlines(
                    half_height,
                    left_a,
                    right_a,
                    colors="tab:orange",
                    linestyles="-",
                    linewidth=2.0,
                    alpha=1.0,
                    **hlines_kw,
                )
                vlines_kw = {"label": "half-width bounds"} if i == 0 else {}
                plt.vlines(
                    [left_a, right_a],
                    ymin=0.0,
                    ymax=half_height,
                    colors="tab:green",
                    linestyles="--",
                    linewidth=1.6,
                    alpha=0.82,
                    **vlines_kw,
                )
    if a_list_fitted is not None and len(a_list_fitted) > 0:
        for i, a_fit in enumerate(a_list_fitted):
            plt.axvline(
                a_fit,
                color="green",
                linestyle="--",
                alpha=0.78,
                label="fitted a" if i == 0 else None,
            )
    plt.xlabel("a (curvature parameter)")
    plt.ylabel("brightness score")
    plt.title(f"{title}\n{len(good_a)} peaks detected")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    show_or_save(save_path)
