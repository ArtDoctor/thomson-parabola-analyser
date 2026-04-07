import collections
import logging

import numpy as np
from pydantic import BaseModel

from oblisk.config import Settings

logger = logging.getLogger(__name__)


class MergeLinesResult(BaseModel):
    """Final merge output plus tracks after merge/stitch, before height and QA filters."""

    filtered_lines: list[list[list[int]]]
    merged_before_quality_filters: list[list[list[int]]]


def _renumber_polylines(lines_in: list[list[list[int]]]) -> list[list[list[int]]]:
    return [[[lid, pt[1], pt[2]] for pt in line] for lid, line in enumerate(lines_in)]


def _split_polyline_on_hops(
    line: list[list[int]],
    settings: Settings,
) -> list[list[list[int]]]:
    """
    Break a polyline where it likely jumps between parallel traces: either a large
    horizontal step over a small row step, or a point far off a short linear trend.
    """
    if len(line) < 5:
        return [line]
    pts = sorted(line, key=lambda p: (p[2], p[1]))
    n = len(pts)
    mpd = float(settings.max_peak_distance)
    hard_dx = 1.02 * mpd
    max_dy_hard = 5
    max_dy_fit = 16
    resid_tol = min(22.0, 0.72 * mpd)
    max_row_gap_split = max(settings.max_x_gap * 2, 22)

    cuts: set[int] = {0, n}
    for i in range(1, n):
        dy = int(pts[i][2]) - int(pts[i - 1][2])
        dx = abs(int(pts[i][1]) - int(pts[i - 1][1]))
        if dy > max_row_gap_split:
            cuts.add(i)
            continue
        if 1 <= dy <= max_dy_hard and dx > hard_dx:
            cuts.add(i)
            continue
        if i >= 4 and dy <= max_dy_fit:
            yw = np.array([float(pts[j][2]) for j in range(i - 4, i)], dtype=float)
            xw = np.array([float(pts[j][1]) for j in range(i - 4, i)], dtype=float)
            yi = float(pts[i][2])
            xi = float(pts[i][1])
            a_mat = np.vstack([yw, np.ones(4)]).T
            coef, _, _, _ = np.linalg.lstsq(a_mat, xw, rcond=None)
            pred = float(coef[0] * yi + coef[1])
            if abs(xi - pred) > resid_tol:
                cuts.add(i)

    ordered = sorted(cuts)
    out: list[list[list[int]]] = []
    for u, v in zip(ordered, ordered[1:]):
        if v > u:
            out.append([list(pts[j]) for j in range(u, v)])
    return out if out else [line]


def _split_pool_on_hops(
    pool: list[list[list[int]]],
    settings: Settings,
) -> list[list[list[int]]]:
    pieces: list[list[list[int]]] = []
    for ln in pool:
        for seg in _split_polyline_on_hops(ln, settings):
            if len(seg) >= settings.min_line_length_1:
                pieces.append(seg)
    return _renumber_polylines(pieces)


def _polyline_redundant_vs_pool(
    bline: list[list[int]],
    pool: list[list[list[int]]],
    px_tol: float,
    frac_threshold: float,
) -> bool:
    if not pool:
        return False
    bx = np.array([float(p[1]) for p in bline])
    by = np.array([float(p[2]) for p in bline])
    parts: list[np.ndarray] = []
    for pl in pool:
        parts.append(np.array([[float(p[1]), float(p[2])] for p in pl], dtype=float))
    all_ex = np.vstack(parts)
    d_min = np.min(
        (bx[:, None] - all_ex[None, :, 0]) ** 2 + (by[:, None] - all_ex[None, :, 1]) ** 2,
        axis=1,
    )
    n_close = int(np.count_nonzero(np.sqrt(d_min) <= px_tol))
    return (n_close / max(len(bx), 1)) >= frac_threshold


def _predictive_trace_from_peaks(
    all_peaks: list[list[tuple[int, int]]],
    settings: Settings,
    image_width: int,
) -> list[list[list[int]]]:
    """Bottom-to-top greedy trace with multi-row lookahead."""
    tol_x = settings.direction_tol_px
    tol_y = settings.direction_tol_px
    max_row_gap = settings.max_x_gap
    max_peak_dist = settings.max_peak_distance
    edge_margin_px = 50

    predict_tol_px_base = 2.0
    predict_tol_per_row = 0.6
    k_sigma = 2.5
    slope_ema_alpha = 0.5
    resid_var_alpha = 0.2
    predict_penalty_weight = 1.0
    excess_penalty_weight = 6.0
    short_len_loosen = 3
    direction_w = 1.0

    line_state: dict[int, dict[str, float | int]] = {}

    def init_line_state(line_id: int, x: int, y: int) -> None:
        line_state[line_id] = dict(m=0.0, var=4.0, last_x=float(x), last_y=float(y), length=1)

    def update_line_state(line_id: int, x: int, y: int) -> None:
        st = line_state[line_id]
        ly = float(st["last_y"])
        lx = float(st["last_x"])
        dy = float(y) - ly
        dx = float(x) - lx
        if dy != 0:
            m_obs = dx / dy
            st["m"] = (1 - slope_ema_alpha) * float(st["m"]) + slope_ema_alpha * m_obs
        pred_x = lx + float(st["m"]) * (float(y) - ly)
        innov = float(x) - pred_x
        st["var"] = (1 - resid_var_alpha) * float(st["var"]) + resid_var_alpha * (innov * innov)
        st["last_x"], st["last_y"] = float(x), float(y)
        st["length"] = int(st["length"]) + 1

    def predict(line_id: int, cy: int) -> float:
        st = line_state[line_id]
        return float(st["last_x"]) + float(st["m"]) * (float(cy) - float(st["last_y"]))

    def pred_tolerance(line_id: int, drow: int) -> float:
        st = line_state[line_id]
        base = predict_tol_px_base + predict_tol_per_row * drow + k_sigma * (float(st["var"]) ** 0.5)
        if int(st["length"]) <= short_len_loosen:
            base *= 2.5
        return base

    lines: list[list[list[int]]] = []
    current_line_id = 0
    active_lines_history: collections.deque = collections.deque(maxlen=max_row_gap)
    num_rows = len(all_peaks)

    for i in range(num_rows - 1, -1, -1):
        curr_peaks = all_peaks[i]
        new_active_lines: dict[tuple[int, int, int], int] = {}

        if i == num_rows - 1:
            for px, py in curr_peaks:
                lines.append([[current_line_id, px, py]])
                init_line_state(current_line_id, px, py)
                new_active_lines[(i, px, py)] = current_line_id
                current_line_id += 1
            active_lines_history.append(new_active_lines)
            continue

        candidates: list[tuple[float, int, int, int]] = []
        for cx, cy in curr_peaks:
            for _j, prev_active_lines in enumerate(reversed(active_lines_history), start=1):
                for (prev_i, px, py), line_id in prev_active_lines.items():
                    drow = prev_i - i
                    if not (1 <= drow <= max_row_gap):
                        continue
                    dcol = abs(px - cx)
                    if dcol > max_peak_dist:
                        continue
                    if not (px >= cx - tol_x and py >= cy - tol_y):
                        continue
                    if dcol < 5 and cx >= image_width - edge_margin_px and len(lines[line_id]) >= 5:
                        continue
                    if len(lines[line_id]) >= 2:
                        prev_dx = lines[line_id][-1][1] - lines[line_id][-2][1]
                        delta_x = cx - lines[line_id][-1][1]
                        if abs(prev_dx) > 3 and abs(delta_x) < 2:
                            continue

                    base_score = float(dcol + drow * 4)

                    if len(lines[line_id]) >= 2:
                        prev_dx = lines[line_id][-1][1] - lines[line_id][-2][1]
                        delta_x = cx - lines[line_id][-1][1]
                        dir_pen = direction_w * abs(np.sign(prev_dx) - np.sign(delta_x))
                    else:
                        dir_pen = 0.0

                    pred_x = predict(line_id, cy)
                    pred_err = abs(float(cx) - pred_x)
                    tol = pred_tolerance(line_id, drow)
                    if pred_err > 1.25 * tol:
                        continue
                    inside = min(pred_err, tol)
                    excess = max(0.0, pred_err - tol)
                    pred_pen = predict_penalty_weight * inside + excess_penalty_weight * excess
                    total_score = base_score + dir_pen + pred_pen
                    candidates.append((total_score, cx, cy, line_id))

        used_points: set[tuple[int, int]] = set()
        used_lines: set[int] = set()
        assignments: list[tuple[int, int, int]] = []
        for score, cx, cy, line_id in sorted(candidates, key=lambda t: t[0]):
            _ = score
            key = (cx, cy)
            if key in used_points or line_id in used_lines:
                continue
            used_points.add(key)
            used_lines.add(line_id)
            assignments.append((cx, cy, line_id))

        assigned_points = {(cx, cy) for (cx, cy, _) in assignments}
        for cx, cy, line_id in assignments:
            lines[line_id].append([line_id, cx, cy])
            update_line_state(line_id, cx, cy)
            new_active_lines[(i, cx, cy)] = line_id

        for cx, cy in curr_peaks:
            if (cx, cy) in assigned_points:
                continue
            lines.append([[current_line_id, cx, cy]])
            init_line_state(current_line_id, cx, cy)
            new_active_lines[(i, cx, cy)] = current_line_id
            current_line_id += 1

        active_lines_history.append(new_active_lines)

    return lines


def _stitch_polylines_one_round(
    lines_in: list[list[list[int]]],
    image_width: int,
    max_bridge_rows: int,
    max_bridge_dx: float,
    max_m_diff: float,
    max_extensions_per_line: int = 1,
) -> tuple[list[list[list[int]]], int]:
    edge_margin_px = 50

    def end_state(line: list[list[int]]) -> tuple[float, float, float]:
        """Tail position and slope dx/dy from up to the last 8 points (y increasing)."""
        arr = line
        x_hi, y_hi = float(arr[-1][1]), float(arr[-1][2])
        k = min(8, len(arr))
        x_lo, y_lo = float(arr[-k][1]), float(arr[-k][2])
        dy = max(y_hi - y_lo, 3.0)
        m = (x_hi - x_lo) / dy
        return x_hi, y_hi, m

    def start_state(line: list[list[int]]) -> tuple[float, float, float]:
        """Head position and slope dx/dy from the first up to 8 points."""
        arr = line
        k = min(8, len(arr))
        x_lo, y_lo = float(arr[0][1]), float(arr[0][2])
        x_hi, y_hi = float(arr[k - 1][1]), float(arr[k - 1][2])
        dy = max(y_hi - y_lo, 3.0)
        m = (x_hi - x_lo) / dy
        return x_lo, y_lo, m

    used_r = [False] * len(lines_in)
    out_lists: list[list[list[int]]] = []
    merges = 0
    for idx_a, a_line in enumerate(lines_in):
        if used_r[idx_a]:
            continue
        for _ext in range(max_extensions_per_line):
            xa, ya, ma = end_state(a_line)
            best: tuple[float, float, int] | None = None
            for idx_b, b_line in enumerate(lines_in):
                if idx_a == idx_b or used_r[idx_b]:
                    continue
                xb, yb, mb = start_state(b_line)
                drow = yb - ya
                if not (1 <= drow <= max_bridge_rows):
                    continue
                x_pred = xa + ma * (yb - ya)
                dx = abs(x_pred - xb)
                both_near_right = xa >= image_width - edge_margin_px and xb >= image_width - edge_margin_px
                if both_near_right and abs(xa - xb) < 3:
                    continue
                m_diff = abs(ma - mb)
                if dx > max_bridge_dx or m_diff > max_m_diff:
                    continue
                key = (dx, m_diff, float(idx_b))
                if best is None or key < (best[0], best[1], float(best[2])):
                    best = (dx, m_diff, idx_b)
            if best is None:
                break
            idx_b = int(best[2])
            b_line = lines_in[idx_b]
            a_line.extend(b_line)
            used_r[idx_b] = True
            merges += 1
        out_lists.append(a_line)
        used_r[idx_a] = True
    return _renumber_polylines(out_lists), merges


def _multi_stitch_polylines(
    lines_in: list[list[list[int]]],
    image_width: int,
    max_bridge_rows: int,
    max_bridge_dx: float,
    max_m_diff: float,
    max_extensions_per_line: int = 1,
) -> tuple[list[list[list[int]]], int]:
    total = 0
    cur = lines_in
    for _ in range(96):
        cur, m = _stitch_polylines_one_round(
            cur,
            image_width,
            max_bridge_rows,
            max_bridge_dx,
            max_m_diff,
            max_extensions_per_line=max_extensions_per_line,
        )
        total += m
        if m == 0:
            break
    return cur, total


def merge_lines(
    filtered_lines: list[list[list[int]]],
    all_peaks: list[list[tuple[int, int]]],
    settings: Settings,
    opened: np.ndarray,
    brightest_spot: tuple[int, ...],
) -> MergeLinesResult:
    image_width = opened.shape[1]

    max_bridge_rows = (
        settings.stitch_max_rows
        if settings.stitch_max_rows is not None
        else max(settings.max_x_gap * 5, min(96, opened.shape[0] // 9))
    )
    max_bridge_dx = (
        settings.stitch_max_dx
        if settings.stitch_max_dx is not None
        else min(34.0, settings.max_peak_distance * 1.12)
    )
    max_m_diff = settings.stitch_max_slope_diff if settings.stitch_max_slope_diff is not None else 0.38

    pred_wrapped = _predictive_trace_from_peaks(all_peaks, settings, image_width)
    # Predictive pass accumulates from top row downward; reverse each polyline so order
    # matches build_lines (increasing row / y) and stitch (end above start) works.
    pred_raw = [[list(pt) for pt in reversed(line)] for line in pred_wrapped]
    logger.debug(
        "merge_lines: predictive trace %d raw polylines; build_lines input %d traces",
        len(pred_raw),
        len(filtered_lines),
    )

    pool = _renumber_polylines(
        [line for line in pred_raw if len(line) >= settings.min_line_length_1]
    )
    pool.sort(key=lambda line: (-len(line), -max(p[2] for p in line)))
    pool, m1 = _multi_stitch_polylines(
        pool, image_width, max_bridge_rows, max_bridge_dx, max_m_diff
    )
    logger.debug(
        "After predictive + stitch: %d polylines (%d merges)", len(pool), m1
    )
    pool = _split_pool_on_hops(pool, settings)
    logger.debug("After hop-split: %d polylines", len(pool))

    built = [
        [[lid, pt[1], pt[2]] for pt in line]
        for lid, line in enumerate(filtered_lines)
    ]
    added = 0
    for line in built:
        if len(line) < settings.min_line_length_1:
            continue
        if _polyline_redundant_vs_pool(line, pool, px_tol=6.0, frac_threshold=0.62):
            continue
        pool.append([[len(pool), pt[1], pt[2]] for pt in line])
        added += 1
    if added:
        pool = _renumber_polylines(pool)
        pool.sort(key=lambda line: (-len(line), -max(p[2] for p in line)))
        pool, m2 = _multi_stitch_polylines(
            pool, image_width, max_bridge_rows, max_bridge_dx, max_m_diff
        )
        logger.debug(
            "Added %d build_lines tracks; after 2nd stitch: %d (%d merges)",
            added,
            len(pool),
            m2,
        )
        pool = _split_pool_on_hops(pool, settings)
        logger.debug("After hop-split (post-build): %d polylines", len(pool))

    merge_keep_min = max(
        settings.min_line_length_1,
        min(settings.min_line_length_2, settings.merge_output_length_cap),
    )
    filtered_lines_out = [line for line in pool if len(line) >= merge_keep_min]
    logger.debug(
        "After len filter (>=%d = max(min_1, min(min_2=%d, cap=%d))): %d polylines",
        merge_keep_min,
        settings.min_line_length_2,
        settings.merge_output_length_cap,
        len(filtered_lines_out),
    )

    merged_before_quality_filters = [
        [list(pt) for pt in line] for line in filtered_lines_out
    ]

    image_height = opened.shape[0]
    max_avg_height = 0.8 * image_height
    filtered_lines_final = [
        line for line in filtered_lines_out
        if np.mean([pt[2] for pt in line]) <= max_avg_height
    ]
    logger.debug(
        "After height filtering: %d lines remain", len(filtered_lines_final)
    )

    filtered_lines_final = [
        line
        for line in filtered_lines_final
        if not any(pt[1] < brightest_spot[1] for pt in line)
    ]

    def compute_jitter(line: list[list[int]]) -> float:
        """Compute jitter as mean absolute second derivative of x."""
        if len(line) < 4:
            return 0.0
        x_vals = np.array([pt[1] for pt in line])
        dx = np.diff(x_vals)
        d2x = np.diff(dx)
        return float(np.mean(np.abs(d2x)))

    max_jitter_threshold = 3.0
    min_length_for_leniency = settings.min_line_length_2 * 3

    logger.debug("Line jitter values:")
    for idx, line in enumerate(filtered_lines_final):
        jitter = compute_jitter(line)
        length = len(line)
        logger.debug("  Line %d: len=%d, jitter=%.3f", idx, length, jitter)

    def is_too_oscillating(line: list[list[int]]) -> bool:
        length = len(line)
        if length >= min_length_for_leniency:
            return False
        jitter = compute_jitter(line)
        threshold = max_jitter_threshold * (1 + length / settings.min_line_length_2)
        return jitter > threshold

    before_osc = len(filtered_lines_final)
    filtered_lines_final = [line for line in filtered_lines_final if not is_too_oscillating(line)]
    removed = before_osc - len(filtered_lines_final)
    logger.debug(
        "After oscillation filtering: %d lines remain (removed %d)",
        len(filtered_lines_final),
        removed,
    )

    return MergeLinesResult(
        filtered_lines=filtered_lines_final,
        merged_before_quality_filters=merged_before_quality_filters,
    )
