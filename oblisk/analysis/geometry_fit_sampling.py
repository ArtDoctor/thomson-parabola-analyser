import numpy as np

from oblisk.analysis.geometry import (
    PerspectiveReference,
    distort_points,
    from_rotated_frame,
    tilt_inverse_Yp,
    xp_span_px_from_points,
)


def bilinear_sample(
    image: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Bilinear interpolation of `image` at (x, y) in pixel coordinates."""
    h, w = image.shape[:2]

    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    mask = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)

    samples = np.full_like(x, fill_value, dtype=float)
    if not np.any(mask):
        return samples

    x0m = x0[mask]
    x1m = x1[mask]
    y0m = y0[mask]
    y1m = y1[mask]

    dx = x[mask] - x0m
    dy = y[mask] - y0m

    ia = image[y0m, x0m].astype(float)
    ib = image[y0m, x1m].astype(float)
    ic = image[y1m, x0m].astype(float)
    id_val = image[y1m, x1m].astype(float)

    samples_mask = (
        ia * (1 - dx) * (1 - dy)
        + ib * dx * (1 - dy)
        + ic * (1 - dx) * dy
        + id_val * dx * dy
    )

    samples[mask] = samples_mask
    return samples


def compute_safe_Xp_range_for_a_values(
    x0_fit: float,
    y0_fit: float,
    theta_fit: float,
    gamma_fit: float,
    delta_fit: float,
    a_values: np.ndarray,
    image: np.ndarray,
    Xp_min_raw: float,
    Xp_max_raw: float,
    n_scan: int = 2000,
    perspective_reference: PerspectiveReference | None = None,
) -> tuple[float, float]:
    h, w = image.shape[:2]

    a_array = np.asarray(a_values, dtype=float)
    if a_array.size == 0:
        raise ValueError("a_values must be non-empty")

    a_min = float(np.min(a_array))
    a_max = float(np.max(a_array))

    xp_grid = np.linspace(Xp_min_raw, Xp_max_raw, n_scan)
    safe_mask = np.ones_like(xp_grid, dtype=bool)
    for a in (a_min, a_max):
        yp = tilt_inverse_Yp(
            xp_grid,
            a,
            gamma_fit,
            delta_fit,
            perspective_reference=perspective_reference,
        )

        x_img, y_img = from_rotated_frame(xp_grid, yp, x0_fit, y0_fit, theta_fit)

        in_img = (x_img >= 0) & (x_img < w) & (y_img >= 0) & (y_img < h)
        safe_mask = safe_mask & in_img

    if not np.any(safe_mask):
        raise ValueError("No Xp range stays inside the image for given a_values.")

    safe_xp_min = float(xp_grid[safe_mask].min())
    safe_xp_max = float(xp_grid[safe_mask].max())
    return safe_xp_min, safe_xp_max


def compute_safe_Xp_range(
    x0_fit: float,
    y0_fit: float,
    theta_fit: float,
    gamma_fit: float,
    delta_fit: float,
    a_ref: float,
    image: np.ndarray,
    Xp_min_raw: float,
    Xp_max_raw: float,
    n_scan: int = 2000,
    perspective_reference: PerspectiveReference | None = None,
) -> tuple[float, float]:
    return compute_safe_Xp_range_for_a_values(
        x0_fit,
        y0_fit,
        theta_fit,
        gamma_fit,
        delta_fit,
        np.array([a_ref], dtype=float),
        image,
        Xp_min_raw,
        Xp_max_raw,
        n_scan=n_scan,
        perspective_reference=perspective_reference,
    )


def get_global_Xp_range_from_lines(
    filtered_lines: list[list[list[int]]],
    x0_fit: float,
    y0_fit: float,
    theta_fit: float,
    margin: float = 0.0,
    a_values: np.ndarray | None = None,
    image: np.ndarray | None = None,
    gamma_fit: float | None = None,
    delta_fit: float | None = None,
    n_scan: int = 2000,
    perspective_reference: PerspectiveReference | None = None,
) -> tuple[float, float]:
    """
    Use existing filtered_lines to estimate a reasonable Xp range
    in the rotated frame.

    When `a_values` are provided, the returned range is additionally clamped
    so those parabolas stay inside `image`.
    """
    points_list: list[np.ndarray] = []
    for line in filtered_lines:
        arr = np.asarray(line, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 3:
            continue
        points_list.append(arr)

    if not points_list:
        raise ValueError("No valid lines for estimating Xp range")

    Xp_min_raw, Xp_max_raw = xp_span_px_from_points(
        points_list=points_list,
        x0_fit=float(x0_fit),
        y0_fit=float(y0_fit),
        theta_fit=float(theta_fit),
        pad_px=float(margin),
    )

    a_array = None if a_values is None else np.asarray(a_values, dtype=float)
    if a_array is None or a_array.size == 0:
        return Xp_min_raw, Xp_max_raw

    if image is None or gamma_fit is None or delta_fit is None:
        raise ValueError(
            "image, gamma_fit, and delta_fit are required when a_values are provided"
        )

    return compute_safe_Xp_range_for_a_values(
        x0_fit=float(x0_fit),
        y0_fit=float(y0_fit),
        theta_fit=float(theta_fit),
        gamma_fit=float(gamma_fit),
        delta_fit=float(delta_fit),
        a_values=a_array,
        image=image,
        Xp_min_raw=Xp_min_raw,
        Xp_max_raw=Xp_max_raw,
        n_scan=n_scan,
        perspective_reference=perspective_reference,
    )


def score_parabolas_over_a(
    image: np.ndarray,
    filtered_lines: list[list[list[int]]],
    x0_fit: float,
    y0_fit: float,
    theta_fit: float,
    gamma_fit: float,
    delta_fit: float,
    a_list_initial: np.ndarray | None = None,
    n_a: int = 400,
    a_padding_factor: float = 0.5,
    n_samples_per_parabola: int = 800,
    Xp_margin: float = 50.0,
    k1_fit: float = 0.0,
    k2_fit: float = 0.0,
    img_center: tuple[float, float] | None = None,
    img_diag: float | None = None,
    perspective_reference: PerspectiveReference | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scan over a grid of 'a' values and compute a score for each,
    based on the mean image intensity along the corresponding parabola.

    For each 'a' value, computes a per-parabola safe Xp range so that
    the parabola stays within image bounds.
    """
    if a_list_initial is not None and len(a_list_initial) > 0:
        a_min0 = float(np.min(a_list_initial))
        a_max0 = float(np.max(a_list_initial))
        span = max(a_max0 - a_min0, 1e-6)
        a_min = 0.0
        a_max = a_max0 + a_padding_factor * span
    else:
        a_min, a_max = -1e-3, 1e-3

    a_grid = np.linspace(a_min, a_max, n_a)

    Xp_min_raw, Xp_max_raw = get_global_Xp_range_from_lines(
        filtered_lines,
        x0_fit,
        y0_fit,
        theta_fit,
    )
    Xp_min_raw -= Xp_margin
    Xp_max_raw += Xp_margin

    scores = np.zeros_like(a_grid, dtype=float)
    background_level = float(np.percentile(image, 10))

    h, w = image.shape[:2]
    xp_dense = np.linspace(Xp_min_raw, Xp_max_raw, n_samples_per_parabola)

    do_distort_score = (
        (abs(k1_fit) > 1e-15 or abs(k2_fit) > 1e-15)
        and img_center is not None
        and img_diag is not None
    )
    cx_s = 0.0
    cy_s = 0.0
    r_s = 1.0
    if do_distort_score:
        assert img_center is not None
        assert img_diag is not None
        cx_s, cy_s = img_center
        r_s = max(img_diag * 0.5, 1.0)

    for i, a in enumerate(a_grid):
        xp = xp_dense
        yp = tilt_inverse_Yp(
            xp,
            a,
            gamma_fit,
            delta_fit,
            perspective_reference=perspective_reference,
        )

        x_img, y_img = from_rotated_frame(xp, yp, x0_fit, y0_fit, theta_fit)
        if do_distort_score:
            x_img, y_img = distort_points(
                x_img,
                y_img,
                cx_s,
                cy_s,
                k1_fit,
                r_s,
                k2=k2_fit,
            )

        in_bounds = (x_img >= 0) & (x_img < w - 1) & (y_img >= 0) & (y_img < h - 1)
        n_in_bounds = np.sum(in_bounds)
        if n_in_bounds < 10:
            scores[i] = 0.0
            continue

        vals = bilinear_sample(
            image,
            x_img[in_bounds],
            y_img[in_bounds],
            fill_value=0.0,
        )
        total_brightness = float(np.sum(np.maximum(vals - background_level, 0.0)))
        scores[i] = total_brightness / n_in_bounds

    return a_grid, scores
