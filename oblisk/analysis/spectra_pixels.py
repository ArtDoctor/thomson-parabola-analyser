import numpy as np

from oblisk.analysis.geometry import (
    tilt_basis_and_shear,
    to_rotated_frame,
    undistort_points,
)
from oblisk.analysis.spectra_models import ClassifiedLine, SpectrumGeometry


def _build_pixel_ownership(
    image: np.ndarray,
    normalized: list[ClassifiedLine],
    geometry: SpectrumGeometry,
    integration_windows_a: np.ndarray | None,
    eps_px: float = 3.0,
) -> np.ndarray | None:
    """
    Assign each pixel to at most one parabola (spectrum index).
    Pixels in overlapping bands go to the parabola whose center a is closest.
    Returns (height, width) int array: ownership[y,x] = line_index or -1.
    """
    del eps_px
    if integration_windows_a is None or len(integration_windows_a) == 0:
        return None

    height, width = image.shape[:2]
    ownership = np.full((height, width), -1, dtype=np.int32)
    eps_a = 1e-12

    yy, xx = np.meshgrid(np.arange(height, dtype=float), np.arange(width, dtype=float), indexing="ij")
    xx_flat, yy_flat = xx.ravel(), yy.ravel()
    if abs(geometry.k1_fit) > 1e-15 or abs(geometry.k2_fit) > 1e-15:
        r_norm = max(geometry.img_diag * 0.5, 1.0)
        xx_flat, yy_flat = undistort_points(
            xx_flat,
            yy_flat,
            geometry.img_center_x,
            geometry.img_center_y,
            geometry.k1_fit,
            r_norm,
            k2=geometry.k2_fit,
        )
    xp, yp = to_rotated_frame(
        xx_flat,
        yy_flat,
        geometry.x0_fit,
        geometry.y0_fit,
        geometry.theta_fit,
    )
    basis, shear = tilt_basis_and_shear(
        xp,
        geometry.gamma_fit,
        geometry.delta_fit,
        perspective_reference=geometry.perspective_reference,
    )
    safe_basis = np.where(np.abs(basis) > eps_a, basis, 1.0)
    a_pixel = np.where(
        np.abs(basis) > eps_a,
        (yp - shear) / safe_basis,
        np.nan,
    )

    line_windows: list[tuple[int, float, float, float]] = []
    for line_index in range(min(len(normalized), len(integration_windows_a))):
        if len(integration_windows_a[line_index]) != 2:
            continue
        left_a = float(integration_windows_a[line_index][0])
        right_a = float(integration_windows_a[line_index][1])
        a_center = float(normalized[line_index].a)
        line_windows.append((line_index, min(left_a, right_a), max(left_a, right_a), a_center))

    valid = np.isfinite(a_pixel)
    for idx in np.where(valid)[0]:
        a_val = float(a_pixel[idx])
        best_line = -1
        best_dist = np.inf
        for line_index, a_min, a_max, a_center in line_windows:
            if a_min <= a_val <= a_max:
                dist = abs(a_center - a_val)
                if dist < best_dist:
                    best_dist = dist
                    best_line = line_index
        if best_line >= 0:
            y_idx, x_idx = divmod(idx, width)
            ownership[y_idx, x_idx] = best_line

    return ownership
