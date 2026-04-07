import numpy as np

from oblisk.analysis.geometry import (
    PerspectiveReference,
    perspective_reference_from_xp,
    tilt_basis_and_shear,
    to_rotated_frame,
    undistort_points,
)


def per_line_parabola_fit_errors(
    filtered_lines: list[list[list[int]]],
    x0_fit: float,
    y0_fit: float,
    theta_fit: float,
    gamma_fit: float,
    delta_fit: float,
    k1_fit: float = 0.0,
    k2_fit: float = 0.0,
    img_center: tuple[float, float] | None = None,
    img_diag: float | None = None,
    perspective_reference: PerspectiveReference | None = None,
) -> tuple[list[dict[str, float | int]], float]:
    """
    For the final shared-vertex geometry, closed-form a_i per line and residuals
    in the warped frame (Y_eff = a_i * X_eff^2). Same model as
    ``fit_global_origin_with_rotation``.

    Returns one row per input line (including lines with zero points after
    validation) and the global unweighted RMS residual over all points.
    """
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    l_list: list[np.ndarray] = []
    num_lines = len(filtered_lines)
    for i, line in enumerate(filtered_lines):
        arr = np.asarray(line, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError(
                f"Each line must be an array of shape (N, >=3). "
                f"Got shape {arr.shape} for line index {i}"
            )
        x_list.append(arr[:, 1])
        y_list.append(arr[:, 2])
        l_list.append(np.full(arr.shape[0], i, dtype=int))

    if num_lines == 0:
        return [], float("nan")

    x = np.concatenate(x_list)
    y = np.concatenate(y_list)
    l_arr = np.concatenate(l_list)
    n_points = int(x.size)
    if n_points == 0:
        rows = [
            {
                "line_index": j,
                "n_points": 0,
                "rmse_y_eff_px": 0.0,
                "mean_abs_residual_y_eff_px": 0.0,
                "rmse_y_eff_weighted_px": 0.0,
            }
            for j in range(num_lines)
        ]
        return rows, float("nan")

    line_counts = np.bincount(l_arr, minlength=num_lines).astype(float)
    line_counts[line_counts == 0.0] = 1.0
    weights_per_point = 1.0 / np.sqrt(line_counts[l_arr])

    if (
        (abs(k1_fit) > 1e-15 or abs(k2_fit) > 1e-15)
        and img_center is not None
        and img_diag is not None
    ):
        cx, cy = img_center
        r_norm = max(img_diag * 0.5, 1.0)
        x_u, y_u = undistort_points(x, y, cx, cy, k1_fit, r_norm, k2=k2_fit)
    else:
        x_u, y_u = x, y

    xp, yp = to_rotated_frame(x_u, y_u, x0_fit, y0_fit, theta_fit)
    shared_reference = (
        perspective_reference
        if perspective_reference is not None
        else perspective_reference_from_xp(xp)
    )
    basis, shear = tilt_basis_and_shear(
        xp,
        gamma_fit,
        delta_fit,
        perspective_reference=shared_reference,
    )
    yp_shifted = yp - shear
    eps = 1e-12
    numer = np.bincount(l_arr, weights=basis * yp_shifted, minlength=num_lines)
    denom = np.bincount(l_arr, weights=basis * basis, minlength=num_lines) + eps
    a_arr = numer / denom
    y_pred = a_arr[l_arr] * basis + shear
    data_res = y_pred - yp

    global_rmse = float(np.sqrt(np.mean(data_res * data_res)))

    rows_out: list[dict[str, float | int]] = []
    for line_idx in range(num_lines):
        mask = l_arr == line_idx
        n_i = int(np.sum(mask))
        if n_i == 0:
            rows_out.append(
                {
                    "line_index": line_idx,
                    "n_points": 0,
                    "rmse_y_eff_px": 0.0,
                    "mean_abs_residual_y_eff_px": 0.0,
                    "rmse_y_eff_weighted_px": 0.0,
                }
            )
            continue
        r = data_res[mask]
        w = weights_per_point[mask]
        rw = r * w
        rows_out.append(
            {
                "line_index": line_idx,
                "n_points": n_i,
                "rmse_y_eff_px": float(np.sqrt(np.mean(r * r))),
                "mean_abs_residual_y_eff_px": float(np.mean(np.abs(r))),
                "rmse_y_eff_weighted_px": float(np.sqrt(np.mean(rw * rw))),
            }
        )
    return rows_out, global_rmse
