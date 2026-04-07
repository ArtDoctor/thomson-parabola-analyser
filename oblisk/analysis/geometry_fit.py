import numpy as np

from oblisk.analysis.geometry import (
    PerspectiveReference,
    perspective_reference_from_xp,
    to_rotated_frame,
    undistort_points,
)


def perspective_reference_from_lines(
    filtered_lines: list[list[list[int]]],
    x0_fit: float,
    y0_fit: float,
    theta_fit: float,
    k1_fit: float = 0.0,
    k2_fit: float = 0.0,
    img_center: tuple[float, float] | None = None,
    img_diag: float | None = None,
) -> PerspectiveReference:
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    for line in filtered_lines:
        arr = np.asarray(line, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 3:
            continue
        x_list.append(arr[:, 1])
        y_list.append(arr[:, 2])

    if not x_list:
        return PerspectiveReference()

    x = np.concatenate(x_list)
    y = np.concatenate(y_list)
    use_distortion = abs(k1_fit) > 1e-15 or abs(k2_fit) > 1e-15
    if use_distortion and img_center is not None and img_diag is not None:
        radius_norm = max(img_diag * 0.5, 1.0)
        x, y = undistort_points(
            x,
            y,
            img_center[0],
            img_center[1],
            k1_fit,
            radius_norm,
            k2=k2_fit,
        )
    xp, _ = to_rotated_frame(x, y, x0_fit, y0_fit, theta_fit)
    return perspective_reference_from_xp(xp)
