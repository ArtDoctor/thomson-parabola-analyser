import numpy as np

from oblisk.analysis.energy import energies_keV_from_xp_px
from oblisk.analysis.geometry import distort_points, from_rotated_frame, tilt_inverse_Yp
from oblisk.analysis.spectra_models import SamplePolyline, SpectrumGeometry, _empty_float_array


def _append_nan_polyline_break(x_points: list[float], y_points: list[float]) -> None:
    if x_points and y_points and not np.isnan(x_points[-1]) and not np.isnan(y_points[-1]):
        x_points.append(float("nan"))
        y_points.append(float("nan"))


def _integrate_unique_band_pixels(
    image: np.ndarray,
    x_samples: np.ndarray,
    y_samples: np.ndarray,
    background_mean: float,
    spectrum_index: int | None = None,
    pixel_ownership: np.ndarray | None = None,
) -> np.ndarray:
    n_xp = x_samples.shape[1]
    if x_samples.size == 0 or y_samples.size == 0:
        return np.zeros(n_xp, dtype=float)

    height, width = image.shape[:2]
    finite = np.isfinite(x_samples) & np.isfinite(y_samples)
    if not np.any(finite):
        return np.zeros(n_xp, dtype=float)

    x_pixel = np.zeros_like(x_samples, dtype=int)
    y_pixel = np.zeros_like(y_samples, dtype=int)
    x_pixel[finite] = np.rint(x_samples[finite]).astype(int)
    y_pixel[finite] = np.rint(y_samples[finite]).astype(int)
    in_bounds = (
        finite
        & ((x_pixel >= 0) & (x_pixel < width))
        & ((y_pixel >= 0) & (y_pixel < height))
    )
    if not np.any(in_bounds):
        return np.zeros(n_xp, dtype=float)

    xp_indices = np.broadcast_to(np.arange(n_xp, dtype=int), x_samples.shape)[in_bounds]
    x_valid = x_samples[in_bounds]
    y_valid = y_samples[in_bounds]
    y_pixel_valid = y_pixel[in_bounds]
    x_pixel_valid = x_pixel[in_bounds]

    if pixel_ownership is not None and spectrum_index is not None:
        owned = pixel_ownership[y_pixel_valid, x_pixel_valid] == spectrum_index
        if not np.any(owned):
            return np.zeros(n_xp, dtype=float)
        xp_indices = xp_indices[owned]
        x_valid = x_valid[owned]
        y_valid = y_valid[owned]
        x_pixel_valid = x_pixel_valid[owned]
        y_pixel_valid = y_pixel_valid[owned]

    pixel_keys = y_pixel_valid.astype(np.int64) * width + x_pixel_valid.astype(np.int64)
    pixel_center_distance_sq = (x_valid - x_pixel_valid) ** 2 + (y_valid - y_pixel_valid) ** 2

    order = np.lexsort((pixel_center_distance_sq, pixel_keys))
    pixel_keys_sorted = pixel_keys[order]
    keep_first = np.ones(len(order), dtype=bool)
    keep_first[1:] = pixel_keys_sorted[1:] != pixel_keys_sorted[:-1]
    chosen = order[keep_first]

    signal = np.maximum(
        image[y_pixel_valid[chosen], x_pixel_valid[chosen]] - background_mean,
        0.0,
    )
    weights = np.zeros(n_xp, dtype=float)
    np.add.at(weights, xp_indices[chosen], signal)
    return weights


def _sample_single_spectrum(
    image: np.ndarray,
    a_rot: float,
    mass_number: int,
    charge_state: int,
    geometry: SpectrumGeometry,
    background_mean: float,
    xp_bounds_px: tuple[float, float],
    eps_px: float,
    sample_count: int,
    local_mean_radius_px: int,
    integration_window_a: tuple[float, float] | None = None,
    integration_a_samples: int = 33,
    spectrum_index: int | None = None,
    pixel_ownership: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, SamplePolyline | None]:
    del local_mean_radius_px
    xp_min, xp_max = xp_bounds_px
    xp_full = np.linspace(xp_min, xp_max, sample_count)
    xp_valid = xp_full[np.abs(xp_full) >= eps_px]
    if xp_valid.size == 0:
        return _empty_float_array(), _empty_float_array(), None

    yp_center = tilt_inverse_Yp(
        xp_valid,
        a_rot,
        geometry.gamma_fit,
        geometry.delta_fit,
        perspective_reference=geometry.perspective_reference,
    )
    x_center, y_center = from_rotated_frame(
        xp_valid,
        yp_center,
        geometry.x0_fit,
        geometry.y0_fit,
        geometry.theta_fit,
    )
    if abs(geometry.k1_fit) > 1e-15 or abs(geometry.k2_fit) > 1e-15:
        r_norm = max(geometry.img_diag * 0.5, 1.0)
        x_center, y_center = distort_points(
            x_center,
            y_center,
            geometry.img_center_x,
            geometry.img_center_y,
            geometry.k1_fit,
            r_norm,
            k2=geometry.k2_fit,
        )
    height, width = image.shape[:2]
    center_in_bounds = (
        (x_center >= 0.0) & (x_center < width) & (y_center >= 0.0) & (y_center < height)
    )

    if integration_window_a is None:
        a_samples = np.array([a_rot], dtype=float)
    else:
        left_a, right_a = integration_window_a
        if not np.isfinite(left_a) or not np.isfinite(right_a):
            a_samples = np.array([a_rot], dtype=float)
        else:
            a_min = min(left_a, right_a)
            a_max = max(left_a, right_a)
            if np.isclose(a_min, a_max):
                a_samples = np.array([a_rot], dtype=float)
            else:
                a_samples = np.linspace(a_min, a_max, max(2, integration_a_samples))

    x_paths = np.zeros((len(a_samples), len(xp_valid)), dtype=float)
    y_paths = np.zeros((len(a_samples), len(xp_valid)), dtype=float)
    do_distort = abs(geometry.k1_fit) > 1e-15 or abs(geometry.k2_fit) > 1e-15
    r_samp = max(geometry.img_diag * 0.5, 1.0) if do_distort else 1.0
    for index, a_sample in enumerate(a_samples):
        yp_sample = tilt_inverse_Yp(
            xp_valid,
            a_sample,
            geometry.gamma_fit,
            geometry.delta_fit,
            perspective_reference=geometry.perspective_reference,
        )
        x_img, y_img = from_rotated_frame(
            xp_valid,
            yp_sample,
            geometry.x0_fit,
            geometry.y0_fit,
            geometry.theta_fit,
        )
        if do_distort:
            x_img, y_img = distort_points(
                x_img,
                y_img,
                geometry.img_center_x,
                geometry.img_center_y,
                geometry.k1_fit,
                r_samp,
                k2=geometry.k2_fit,
            )
        x_paths[index] = x_img
        y_paths[index] = y_img

    weights = _integrate_unique_band_pixels(
        image=image,
        x_samples=x_paths,
        y_samples=y_paths,
        background_mean=background_mean,
        spectrum_index=spectrum_index,
        pixel_ownership=pixel_ownership,
    )

    energies_keV = energies_keV_from_xp_px(
        xp_px=xp_valid,
        mass_number=mass_number,
        charge_state=charge_state,
        meters_per_pixel=geometry.meters_per_pixel,
        b_field_t=geometry.b_field_t,
        magnetic_length_m=geometry.magnetic_length_m,
        drift_length_m=geometry.drift_length_m,
    )

    good = np.isfinite(energies_keV) & (energies_keV > 0.0) & np.isfinite(weights) & (weights > 0.0)
    polyline = None
    if np.any(center_in_bounds):
        poly_x_list: list[float] = []
        poly_y_list: list[float] = []
        last_visible_index: int | None = None
        for index in range(int(xp_valid.size)):
            if not bool(center_in_bounds[index]):
                _append_nan_polyline_break(poly_x_list, poly_y_list)
                last_visible_index = None
                continue
            if last_visible_index is not None and (
                xp_valid[last_visible_index] < 0.0 and xp_valid[index] > 0.0
            ):
                _append_nan_polyline_break(poly_x_list, poly_y_list)
            poly_x_list.append(float(x_center[index]))
            poly_y_list.append(float(y_center[index]))
            last_visible_index = index
        if poly_x_list and np.isnan(poly_x_list[-1]) and np.isnan(poly_y_list[-1]):
            poly_x_list.pop()
            poly_y_list.pop()
        polyline = SamplePolyline(
            x=np.asarray(poly_x_list, dtype=float),
            y=np.asarray(poly_y_list, dtype=float),
        )
    return (
        energies_keV[good],
        weights[good],
        polyline,
    )
