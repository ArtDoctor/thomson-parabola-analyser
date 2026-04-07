import logging

import numpy as np
from scipy.signal import find_peaks

from oblisk.config import Settings

logger = logging.getLogger(__name__)


def moving_average(signal: np.ndarray, window_size: int) -> np.ndarray:
    if window_size % 2 == 0:
        window_size += 1
    pad = window_size // 2
    kernel = np.ones(window_size) / window_size
    return np.convolve(np.pad(signal, pad, mode="edge"), kernel, mode="valid")


def extract_peaks(
    image: np.ndarray,
    starting_pixel: int,
    ending_pixel: int,
    settings: Settings,
) -> list[list[tuple[int, int]]]:
    all_peaks: list[list[tuple[int, int]]] = []
    for row in range(starting_pixel, ending_pixel):
        y = image[row, :]
        x = np.arange(len(y))

        y_ma = moving_average(y, settings.window)
        y_ma = moving_average(y_ma, settings.window)
        y_ma = moving_average(y_ma, settings.window)

        peaks, _ = find_peaks(
            y_ma,
            prominence=settings.prominence,
            distance=settings.distance,
        )
        row_peaks = [(int(x[peak]), int(row)) for peak in peaks]
        all_peaks.append(row_peaks)
    return all_peaks


def build_lines(
    all_peaks: list[list[tuple[int, int]]],
    settings: Settings,
    image_width: int | None = None,
) -> list[list[list[int]]]:
    """Connect row-wise peaks into trace candidates."""

    lines = []
    current_line_id = 0
    active_lines = {}

    direction_change_penalty = 5.0
    continuation_bonus = 8.0

    for i in range(len(all_peaks)):
        curr_peaks = all_peaks[i]

        if i == 0:
            for px, py in curr_peaks:
                lines.append([[current_line_id, px, py]])
                active_lines[(px, py)] = current_line_id
                current_line_id += 1
            continue

        prev_peaks = all_peaks[i - 1]
        new_active_lines = {}

        if len(prev_peaks) == 0:
            for cx, cy in curr_peaks:
                lines.append([[current_line_id, cx, cy]])
                new_active_lines[(cx, cy)] = current_line_id
                current_line_id += 1
            active_lines = new_active_lines
            continue

        candidates: list[tuple[float, int, int, int, int, int | None]] = []
        for cx, cy in curr_peaks:
            for px, py in prev_peaks:
                dist = abs(px - cx)
                if dist >= settings.max_peak_distance:
                    continue

                base_score = float(dist)

                if (px, py) in active_lines:
                    line_id = active_lines[(px, py)]
                    line_pts = lines[line_id]
                    if (
                        image_width is not None
                        and dist < 3
                        and cx >= image_width - 50
                        and len(line_pts) >= 10
                    ):
                        continue
                    if len(line_pts) >= 2:
                        prev_dx = line_pts[-1][1] - line_pts[-2][1]
                        curr_dx = cx - px
                        if abs(prev_dx) > 3 and abs(curr_dx) < 2:
                            continue

                    base_score -= continuation_bonus

                    if len(line_pts) >= 2:
                        prev_dx = line_pts[-1][1] - line_pts[-2][1]
                        curr_dx = cx - px
                        if prev_dx * curr_dx < 0:
                            base_score += direction_change_penalty

                    candidates.append((base_score, cx, cy, px, py, line_id))
                else:
                    candidates.append((base_score + 10.0, cx, cy, px, py, None))

        candidates.sort(key=lambda item: item[0])
        used_curr = set()
        used_prev = set()

        for score, cx, cy, px, py, chosen_line_id in candidates:
            _ = score
            if (cx, cy) in used_curr or (px, py) in used_prev:
                continue

            used_curr.add((cx, cy))
            used_prev.add((px, py))

            if chosen_line_id is not None:
                lines[chosen_line_id].append([chosen_line_id, cx, cy])
                new_active_lines[(cx, cy)] = chosen_line_id
            else:
                lines.append([[current_line_id, cx, cy]])
                new_active_lines[(cx, cy)] = current_line_id
                current_line_id += 1

        for cx, cy in curr_peaks:
            if (cx, cy) not in used_curr:
                lines.append([[current_line_id, cx, cy]])
                new_active_lines[(cx, cy)] = current_line_id
                current_line_id += 1

        active_lines = new_active_lines

    filtered_lines = [line for line in lines if len(line) >= settings.min_line_length_1]
    logger.debug(
        "Total lines: %d, after filtering: %d", len(lines), len(filtered_lines)
    )
    return filtered_lines
