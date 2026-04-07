import numpy as np


def smooth_lines(
    filtered_lines: list[list[list[int]]],
    window_size: int = 5,
    threshold_multiplier: float = 2.0,
) -> list[list[list[int]]]:
    """
    Smooth lines by detecting and correcting outlier points.
    Points that jump away from the local trend are interpolated from neighbors.
    """
    smoothed_lines = []

    for line in filtered_lines:
        if len(line) < 3:
            smoothed_lines.append(line)
            continue

        arr = np.array(line, dtype=float)
        n_points = len(arr)
        smoothed_arr = arr.copy()

        for coord_idx in [1, 2]:
            coords = arr[:, coord_idx].copy()
            diffs = np.abs(np.diff(coords))
            if len(diffs) == 0:
                continue

            median_diff = np.median(diffs) if len(diffs) > 0 else 0
            threshold = max(median_diff * threshold_multiplier, 1.0)
            outlier_mask = np.zeros(n_points, dtype=bool)

            half_window = window_size // 2
            for i in range(n_points):
                start = max(0, i - half_window)
                end = min(n_points, i + half_window + 1)

                window_indices = [j for j in range(start, end) if j != i]
                if len(window_indices) < 2:
                    continue

                window_values = coords[window_indices]
                local_median = np.median(window_values)
                deviation = abs(coords[i] - local_median)
                local_spread = np.std(window_values)
                if local_spread < 0.1:
                    local_spread = 0.1

                if deviation > threshold and deviation > 3 * local_spread:
                    outlier_mask[i] = True

            if np.any(outlier_mask):
                valid_indices = np.where(~outlier_mask)[0]
                outlier_indices = np.where(outlier_mask)[0]
                if len(valid_indices) >= 2:
                    smoothed_arr[outlier_indices, coord_idx] = np.interp(
                        outlier_indices,
                        valid_indices,
                        coords[valid_indices],
                    )

        for coord_idx in [1, 2]:
            coords = smoothed_arr[:, coord_idx].copy()
            averaged = np.zeros_like(coords)

            half_window = window_size // 2
            for i in range(n_points):
                start = max(0, i - half_window)
                end = min(n_points, i + half_window + 1)
                averaged[i] = np.mean(coords[start:end])

            smoothed_arr[:, coord_idx] = averaged

        smoothed_lines.append(smoothed_arr.astype(int).tolist())

    return smoothed_lines
