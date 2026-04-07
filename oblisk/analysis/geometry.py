from typing import Iterable

import numpy as np
from pydantic import BaseModel


class PerspectiveReference(BaseModel):
    center_xp: float = 0.0
    span_xp: float = 1.0


def perspective_reference_from_xp(Xp: np.ndarray) -> PerspectiveReference:
    Xp_arr = np.asarray(Xp, dtype=float)
    finite = Xp_arr[np.isfinite(Xp_arr)]
    if finite.size == 0:
        return PerspectiveReference()
    span_xp = max(float(np.max(finite) - np.min(finite)), 1e-3)
    center_xp = float(np.mean(finite))
    return PerspectiveReference(center_xp=center_xp, span_xp=span_xp)


def to_rotated_frame(
    X: np.ndarray,
    Y: np.ndarray,
    x0: float,
    y0: float,
    theta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Translate by (x0, y0) and rotate by -theta to align all parabolas
    with the rotated y'-axis.
    """
    c, s = np.cos(theta), np.sin(theta)
    Xt = X - x0
    Yt = Y - y0
    Xp = c * Xt + s * Yt
    Yp = -s * Xt + c * Yt
    return Xp, Yp


def from_rotated_frame(
    Xp: np.ndarray,
    Yp: np.ndarray,
    x0: float,
    y0: float,
    theta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of `to_rotated_frame`."""
    c, s = np.cos(theta), np.sin(theta)
    Xt = c * Xp - s * Yp
    Yt = s * Xp + c * Yp
    return Xt + x0, Yt + y0


def apply_perspective_scale(
    Xp: np.ndarray,
    gamma: float,
    perspective_reference: PerspectiveReference | None = None,
) -> np.ndarray:
    """
    Simple 1-parameter perspective-like model (global):
        X_eff = Xp * (1 + gamma * (Xp - mean(Xp)) / span(Xp))
    """
    if gamma is None or abs(gamma) < 1e-12:
        return np.asarray(Xp, dtype=float)

    Xp_arr = np.asarray(Xp, dtype=float)
    reference = (
        perspective_reference
        if perspective_reference is not None
        else perspective_reference_from_xp(Xp_arr)
    )
    R = max(float(reference.span_xp), 1e-3)
    scale = 1.0 + float(gamma) * (Xp_arr - float(reference.center_xp)) / R
    return Xp_arr * scale


def tilt_basis_and_shear(
    Xp: np.ndarray,
    gamma: float,
    delta: float,
    perspective_reference: PerspectiveReference | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Combined perspective + 3D tilt model.

    gamma  – perspective parameter (position-dependent X-scaling, as before)
    delta  – detector pitch angle (tilt around X-axis, radians)

    The parabola model in the observed rotated frame becomes:

        Yp = a_i * basis + shear

    where
        X_eff  = apply_perspective_scale(Xp, gamma)   [perspective]
        basis  = X_eff²  /  cos(delta)                [pitch foreshortening]
        shear  = sin(delta) * Xp                       [pitch cross-term]

    The yaw component (tilt around Y-axis) is already captured by gamma's
    position-dependent X-scaling.  Only pitch (delta) adds genuinely new
    effects: a Y-axis foreshortening (1/cos) and a linear shear (sin*Xp).
    """
    X_eff = apply_perspective_scale(
        Xp,
        gamma,
        perspective_reference=perspective_reference,
    )
    cos_d = np.cos(delta)
    basis = (X_eff ** 2) / cos_d
    shear = np.sin(delta) * np.asarray(Xp, dtype=float)
    return basis, shear


def tilt_inverse_Yp(
    Xp: np.ndarray,
    a: float,
    gamma: float,
    delta: float,
    perspective_reference: PerspectiveReference | None = None,
) -> np.ndarray:
    """
    Compute Yp for a given curvature *a* under the combined
    perspective + pitch-tilt model.

        Yp = a * X_eff² / cos(delta) + sin(delta) * Xp
    """
    basis, shear = tilt_basis_and_shear(
        Xp,
        gamma,
        delta,
        perspective_reference=perspective_reference,
    )
    return a * basis + shear


def undistort_points(
    X: np.ndarray,
    Y: np.ndarray,
    cx: float,
    cy: float,
    k1: float,
    R_norm: float,
    k2: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply radial undistortion to observed pixel coordinates.

    Maps distorted (observed) coordinates to undistorted coordinates:

        r_norm = sqrt((x - cx)^2 + (y - cy)^2) / R_norm
        x_u = x + (x - cx) * (k1 * r_norm^2 + k2 * r_norm^4)
        y_u = y + (y - cy) * (k1 * r_norm^2 + k2 * r_norm^4)

    Parameters
    ----------
    cx, cy : distortion center (e.g. image center)
    k1 : radial distortion coefficient (r^2 term)
    k2 : radial distortion coefficient (r^4 term)
    R_norm : normalization radius (e.g. half-diagonal) to keep k1,k2 ~ O(1)
    """
    if abs(k1) < 1e-15 and abs(k2) < 1e-15:
        return np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    X_arr = np.asarray(X, dtype=float)
    Y_arr = np.asarray(Y, dtype=float)
    dx = X_arr - cx
    dy = Y_arr - cy
    r2_norm = (dx * dx + dy * dy) / (R_norm * R_norm)
    factor = k1 * r2_norm + k2 * r2_norm * r2_norm
    return X_arr + dx * factor, Y_arr + dy * factor


def distort_points(
    X_u: np.ndarray,
    Y_u: np.ndarray,
    cx: float,
    cy: float,
    k1: float,
    R_norm: float,
    n_iter: int = 5,
    k2: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Invert ``undistort_points`` for a radial model by solving for the distorted
    radius with Newton iterations.

    Given undistorted coordinates, find the distorted (observed) pixel
    coordinates that would map to them under the forward undistortion model.
    """
    if abs(k1) < 1e-15 and abs(k2) < 1e-15:
        return np.asarray(X_u, dtype=float), np.asarray(Y_u, dtype=float)

    X_u_arr = np.asarray(X_u, dtype=float)
    Y_u_arr = np.asarray(Y_u, dtype=float)
    flat_xu = X_u_arr.ravel()
    flat_yu = Y_u_arr.ravel()
    flat_dx = flat_xu - cx
    flat_dy = flat_yu - cy
    flat_ru = np.hypot(flat_dx, flat_dy)

    flat_xd = np.full_like(flat_xu, np.nan)
    flat_yd = np.full_like(flat_yu, np.nan)

    finite = np.isfinite(flat_xu) & np.isfinite(flat_yu) & np.isfinite(flat_ru)
    if not np.any(finite):
        return flat_xd.reshape(X_u_arr.shape), flat_yd.reshape(Y_u_arr.shape)

    centered = finite & (flat_ru <= 1e-12)
    flat_xd[centered] = flat_xu[centered]
    flat_yd[centered] = flat_yu[centered]

    active = finite & ~centered
    if not np.any(active):
        return flat_xd.reshape(X_u_arr.shape), flat_yd.reshape(Y_u_arr.shape)

    flat_rd = flat_ru.copy()
    failed = np.zeros_like(flat_rd, dtype=bool)
    converged = np.zeros_like(flat_rd, dtype=bool)
    tol = 1e-9 * max(float(R_norm), 1.0)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(max(1, int(n_iter))):
            work = active & ~failed & ~converged
            if not np.any(work):
                break

            r_work = flat_rd[work]
            s = np.square(r_work / R_norm)
            residual = r_work * (1.0 + k1 * s + k2 * s * s) - flat_ru[work]
            deriv = 1.0 + 3.0 * k1 * s + 5.0 * k2 * s * s

            valid_step = np.isfinite(residual)
            valid_step = valid_step & np.isfinite(deriv)
            valid_step = valid_step & (np.abs(deriv) > 1e-12)
            next_r = np.full_like(r_work, np.nan)
            next_r[valid_step] = r_work[valid_step] - residual[valid_step] / deriv[valid_step]
            valid_next = valid_step & np.isfinite(next_r) & (next_r >= 0.0)

            work_idx = np.flatnonzero(work)
            flat_rd[work_idx[valid_next]] = next_r[valid_next]
            converged[work_idx[valid_next]] = np.abs(next_r[valid_next] - r_work[valid_next]) <= tol
            failed[work_idx[~valid_next]] = True

        remaining = active & ~failed
        if np.any(remaining):
            r_rem = flat_rd[remaining]
            s = np.square(r_rem / R_norm)
            residual = r_rem * (1.0 + k1 * s + k2 * s * s) - flat_ru[remaining]
            converged_remaining = np.isfinite(residual) & (np.abs(residual) <= tol)
            remaining_idx = np.flatnonzero(remaining)
            converged[remaining_idx[converged_remaining]] = True
            failed[remaining_idx[~converged_remaining]] = True

    ok = active & converged & ~failed
    scale = flat_rd[ok] / flat_ru[ok]
    flat_xd[ok] = cx + flat_dx[ok] * scale
    flat_yd[ok] = cy + flat_dy[ok] * scale
    return flat_xd.reshape(X_u_arr.shape), flat_yd.reshape(Y_u_arr.shape)


def refit_a_rot(
    points: np.ndarray,
    x0_fit: float,
    y0_fit: float,
    theta_fit: float,
) -> float:
    """Refit a in the rotated frame (Y' ~= a * X'^2)."""
    pts = np.asarray(points, dtype=float)
    Xp, Yp = to_rotated_frame(pts[:, 1], pts[:, 2], x0_fit, y0_fit, theta_fit)
    t = Xp**2
    num = float(np.sum(t * Yp))
    den = float(np.sum(t * t) + 1e-12)
    return num / den


def xp_span_px_from_points(
    points_list: Iterable[np.ndarray],
    x0_fit: float,
    y0_fit: float,
    theta_fit: float,
    pad_px: float = 0.0,
) -> tuple[float, float]:
    """
    Compute a global X' span in the rotated frame from a list of point arrays.

    Supported point formats:
    - (N, >=3) with columns [..., x, y] at indices 1 and 2
    - (N, 2) with columns [x, y] at indices 0 and 1
    """
    xmins: list[float] = []
    xmaxs: list[float] = []

    for points in points_list:
        if points is None:
            continue
        arr = np.asarray(points, dtype=float)
        if arr.size == 0:
            continue
        if arr.ndim != 2 or arr.shape[1] < 2:
            continue

        if arr.shape[1] >= 3:
            xs = arr[:, 1]
            ys = arr[:, 2]
        else:
            xs = arr[:, 0]
            ys = arr[:, 1]

        Xp, _ = to_rotated_frame(xs, ys, x0_fit, y0_fit, theta_fit)
        xmins.append(float(np.min(Xp)))
        xmaxs.append(float(np.max(Xp)))

    if not xmins or not xmaxs:
        raise ValueError("No valid points provided to compute X' span.")

    return min(xmins) - pad_px, max(xmaxs) + pad_px


def dominant_xp_sign_from_points(
    points_list: Iterable[np.ndarray],
    x0_fit: float,
    y0_fit: float,
    theta_fit: float,
    eps_px: float = 3.0,
) -> int:
    """
    Infer which X' branch is physically supported by detected line points.

    Returns +1 when positive-X' support dominates, -1 when negative-X' support
    dominates. Support is measured by total absolute X' distance beyond eps_px.
    """
    pos_support = 0.0
    neg_support = 0.0

    for points in points_list:
        if points is None:
            continue
        arr = np.asarray(points, dtype=float)
        if arr.size == 0 or arr.ndim != 2 or arr.shape[1] < 2:
            continue

        if arr.shape[1] >= 3:
            xs = arr[:, 1]
            ys = arr[:, 2]
        else:
            xs = arr[:, 0]
            ys = arr[:, 1]

        Xp, _ = to_rotated_frame(xs, ys, x0_fit, y0_fit, theta_fit)
        finite = np.isfinite(Xp)
        if not np.any(finite):
            continue
        Xp = Xp[finite]
        pos = Xp[Xp > eps_px]
        neg = Xp[Xp < -eps_px]
        pos_support += float(np.sum(pos))
        neg_support += float(np.sum(-neg))

    return 1 if pos_support >= neg_support else -1


def visible_polyline_with_nan_breaks(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Keep only in-image samples from a polyline and insert NaN separators between
    disjoint visible runs so matplotlib does not draw connectors across the frame.
    """
    x = np.asarray(x_arr, dtype=float)
    y = np.asarray(y_arr, dtype=float)
    visible = np.isfinite(x)
    visible = visible & np.isfinite(y)
    visible = visible & (x >= 0.0)
    visible = visible & (x < float(width))
    visible = visible & (y >= 0.0)
    visible = visible & (y < float(height))
    if not np.any(visible):
        return None

    out_x: list[float] = []
    out_y: list[float] = []
    last_was_visible = False
    for index, is_visible in enumerate(visible):
        if not bool(is_visible):
            if last_was_visible:
                out_x.append(float("nan"))
                out_y.append(float("nan"))
            last_was_visible = False
            continue
        out_x.append(float(x[index]))
        out_y.append(float(y[index]))
        last_was_visible = True

    if out_x and np.isnan(out_x[-1]) and np.isnan(out_y[-1]):
        out_x.pop()
        out_y.pop()

    return np.asarray(out_x, dtype=float), np.asarray(out_y, dtype=float)


def longest_finite_polyline_segment(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Return the longest contiguous finite run from a polyline that may contain NaNs.
    """
    x = np.asarray(x_arr, dtype=float)
    y = np.asarray(y_arr, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return None

    best_start = 0
    best_end = -1
    run_start: int | None = None
    for index, ok in enumerate(finite):
        if ok and run_start is None:
            run_start = index
        elif not ok and run_start is not None:
            if index - run_start > best_end - best_start + 1:
                best_start = run_start
                best_end = index - 1
            run_start = None
    if run_start is not None and len(finite) - run_start > best_end - best_start + 1:
        best_start = run_start
        best_end = len(finite) - 1

    return x[best_start:best_end + 1], y[best_start:best_end + 1]


def finite_polyline_segments(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Split a polyline into all contiguous finite runs.

    Inputs may already contain NaN separators. Empty/fully non-finite inputs
    yield an empty list.
    """
    x = np.asarray(x_arr, dtype=float)
    y = np.asarray(y_arr, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return []

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    run_start: int | None = None
    for index, ok in enumerate(finite):
        if ok and run_start is None:
            run_start = index
        elif not ok and run_start is not None:
            segments.append((x[run_start:index], y[run_start:index]))
            run_start = None
    if run_start is not None:
        segments.append((x[run_start:], y[run_start:]))
    return segments


def xp_span_px_from_image(
    image: np.ndarray,
    x0_fit: float,
    y0_fit: float,
    theta_fit: float,
    pad_px: float = 0.0,
) -> tuple[float, float]:
    """Fallback X' span computed from the rotated image corners."""
    height, width = image.shape[:2]
    xs = np.array([0, width - 1, 0, width - 1], dtype=float)
    ys = np.array([0, 0, height - 1, height - 1], dtype=float)
    Xp, _ = to_rotated_frame(xs, ys, x0_fit, y0_fit, theta_fit)
    return float(np.min(Xp)) - pad_px, float(np.max(Xp)) + pad_px
