import logging
from collections.abc import Iterable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)
from oblisk.analysis.geometry import (
    from_rotated_frame,
    to_rotated_frame,
    xp_span_px_from_image,
    xp_span_px_from_points,
)
from oblisk.analysis.background import BackgroundRoi, compute_background_mean
from oblisk.analysis.energy import energies_keV_from_xp_m
from oblisk.config import Settings


def magnet_calibration_sorted_matrix(settings: Settings) -> np.ndarray:
    rows = settings.magnet_calibration
    arr = np.array(
        [[float(r.current_amps), float(r.field_millitesla)] for r in rows],
        dtype=float,
    )
    return arr[np.argsort(arr[:, 0])]


def b_millitesla_from_magnet_current_amps(
    i_amps: float,
    calib_sorted: np.ndarray,
) -> float:
    return float(np.interp(i_amps, calib_sorted[:, 0], calib_sorted[:, 1]))


def b_tesla_from_magnet_current_amps(
    i_amps: float,
    calib_sorted: np.ndarray,
) -> float:
    return b_millitesla_from_magnet_current_amps(i_amps, calib_sorted) * 1e-3


def default_primary_b_field_tesla() -> float:
    settings = Settings()
    cal = magnet_calibration_sorted_matrix(settings)
    return b_tesla_from_magnet_current_amps(
        settings.magnet_current_standard_amps,
        cal,
    )


def default_fallback_meters_per_pixel() -> float:
    return Settings().fallback_meters_per_pixel


# Physical detector size for 6×6 cm IPs (Config B), used for analytical H
DETECTOR_SIZE_M = 6.0 * 1e-2          # 60 mm

# ----- 2) Physics constants and K-factor -----
e = 1.602176634e-19  # C
m_p = 1.67262192369e-27  # kg
mq = m_p / e  # C/kg


def K_factor(
    mq: float,
    E: float,
    LiE: float,
    LfE: float,
    B: float,
    LiB: float,
    LfB: float,
) -> float:
    num = mq * E * LiE * (LiE / 2 + LfE)
    den = (B**2) * (LiB**2) * (LiB / 2 + LfB)**2
    return num / den                     # [meters / meters^2] -> [1/m]


# Geometry (meters)
cm = 1e-2
LiB = 8 * cm
LiE = 8 * cm

# Config A: 15×15 cm IPs
LfB_A = 88 * cm
E_A = 0.15e6                        # V/m
LfE_A = LfB_A - 9 * cm

# Config B: 6×6 cm IPs
LfB_B = 29.5 * cm
E_B = 0.13e6                        # V/m
LfE_B = LfB_B - 9 * cm

def get_hydrogen_a(
    m_per_px_img: float,
    w: int,
    h: int,
    x0_fit: float,
    y0_fit: float,
    theta_fit: float,
    b_tesla: float,
    diagnostic_prints: bool = True,
) -> tuple[float, float]:
    """
    Compute analytical hydrogen parabola curvature (a) in the rotated frame.
    """
    b_use = float(b_tesla)
    k_a = K_factor(mq, E_A, LiE, LfE_A, b_use, LiB, LfB_A)
    k_b = K_factor(mq, E_B, LiE, LfE_B, b_use, LiB, LfB_B)
    px_per_m_img = 1.0 / m_per_px_img

    # Take left/right bounds along a row through the vertex's y to get a wide span in the rotated X' frame.
    x_bounds = np.array([0, w - 1], dtype=float)
    y_row = np.array([y0_fit, y0_fit], dtype=float)  # same y as vertex
    Xp_px, _ = to_rotated_frame(x_bounds, y_row, x0_fit, y0_fit, theta_fit)

    Xp_px_min, Xp_px_max = float(np.min(Xp_px)), float(np.max(Xp_px))
    Xp_px_samp = np.linspace(Xp_px_min, Xp_px_max, 1000)

    # Physics in meters (use image-derived scale)
    Xp_m = Xp_px_samp * m_per_px_img
    Yp_m_A = k_a * (Xp_m**2)      # meters
    Yp_m_B = k_b * (Xp_m**2)      # meters

    # Back to pixels
    Yp_px_A = Yp_m_A * px_per_m_img
    Yp_px_B = Yp_m_B * px_per_m_img

    # Map back to the original (image) frame
    xA, yA = from_rotated_frame(Xp_px_samp, Yp_px_A, x0_fit, y0_fit, theta_fit)
    xB, yB = from_rotated_frame(Xp_px_samp, Yp_px_B, x0_fit, y0_fit, theta_fit)

    # Parameter "a" for hydrogen in the rotated frame: Yp = a * Xp^2
    a1 = k_a / px_per_m_img
    a2 = k_b / px_per_m_img
    if diagnostic_prints:
        det_sz_cm = DETECTOR_SIZE_M * 100
        m_per_px_um = m_per_px_img * 1e6
        logger.debug(
            "Hydrogen analytical: m_per_px=%.1f µm (from %d×%d px, %.0f cm detector)",
            m_per_px_um,
            w,
            h,
            det_sz_cm,
        )
        logger.debug("Hydrogen parameter 'a1' for 15x15 cm IPs: %.3e", a1)
        logger.debug("Hydrogen parameter 'a2' for 6x6 cm IPs: %.3e", a2)

    return a1, a2


# Use the same magnetic geometry as for the analytical H^+ (6×6 cm config from the context above)
LfB = LfB_B


# ---------- Background (from ORIGINAL image) ----------
def get_background(test_img: np.ndarray) -> tuple[int, int, int, int, float]:
    img = test_img.astype(float)
    H, W = img.shape[:2]
    bx0, bx1 = 200, 400
    by0, by1 = 400, 800
    roi = BackgroundRoi(x0=bx0, x1=bx1, y0=by0, y1=by1)
    clipped_roi, bg_mean = compute_background_mean(image=img, roi=roi)
    return clipped_roi.x0, clipped_roi.y0, clipped_roi.x1, clipped_roi.y1, bg_mean


def local_mean_radius(
    img: np.ndarray,
    x: float,
    y: float,
    r: int = 2,
) -> float:
    """
    Mean of pixels in a disk of radius r around (x, y) on ORIGINAL image.
    x,y are float in image coordinates (origin lower-left since we use origin='lower' in plotting).
    """
    xi, yi = int(round(x)), int(round(y))
    y_min = max(0, yi - r)
    y_max = min(img.shape[0] - 1, yi + r)
    x_min = max(0, xi - r)
    x_max = min(img.shape[1] - 1, xi + r)
    vals = []
    rr2 = r * r
    for yy in range(y_min, y_max + 1):
        dy2 = (yy - yi)**2
        for xx in range(x_min, x_max + 1):
            if dy2 + (xx - xi)**2 <= rr2:
                vals.append(img[yy, xx])
    return np.mean(vals) if vals else 0.0


def image_corners_rotated(
    W: int,
    H: int,
    x0_fit: float,
    y0_fit: float,
    theta_fit: float,
) -> tuple[np.ndarray, np.ndarray]:
    xs = np.array([0, W - 1, 0, W - 1], dtype=float)
    ys = np.array([0, 0, H - 1, H - 1], dtype=float)
    Xp, Yp = to_rotated_frame(xs, ys, x0_fit, y0_fit, theta_fit)
    return Xp, Yp


def sample_parabola_energy(
    a_rot: float,
    A_massnum: int,
    Z_charge: int,
    W: int,
    H: int,
    x0_fit: float,
    y0_fit: float,
    theta_fit: float,
    B_T: float,
    LiB: float,
    LfB: float,
    m_per_px: float,
    px_per_m: float,
    bg_mean: float,
    img: np.ndarray,
    eps_px: float = 3,
    N: int = 2500,
    r_pix: int = 2,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """
    Sample along Y' = a_rot * X'^2 from the vertex to the image edges (both +X' and -X'),
    measure background-subtracted intensity (mean in r_pix), and compute energy per sample.
    Returns energies_keV (1D), weights (1D), and polyline(s) used for visualization.
    """
    # max X' span available within the image (in pixels, centered at vertex)
    Xp_corners, _ = image_corners_rotated(W, H, x0_fit, y0_fit, theta_fit)
    Xp_max_abs = float(np.max(np.abs(Xp_corners)))
    # avoid divide-by-zero
    Xp_pos = np.linspace(eps_px, Xp_max_abs, N)
    Xp_neg = -Xp_pos

    all_E_keV: list[np.ndarray] = []
    all_w: list[np.ndarray] = []
    polylines: list[tuple[np.ndarray, np.ndarray]] = []

    for Xp_branch in (Xp_pos, Xp_neg):
        Yp = a_rot * (Xp_branch**2)
        x_img, y_img = from_rotated_frame(Xp_branch, Yp, x0_fit, y0_fit, theta_fit)

        # keep points inside the image
        mask = (x_img >= 0) & (x_img < W) & (y_img >= 0) & (y_img < H)
        x_in = x_img[mask]
        y_in = y_img[mask]
        Xp_in = Xp_branch[mask]

        if len(x_in) == 0:
            continue

        # local mean intensity minus background
        # (vectorized sampling with interpolation is possible; here we follow explicit mean in a radius)
        weights = np.array([max(local_mean_radius(img, xi, yi, r=r_pix) - bg_mean, 0.0)
                            for xi, yi in zip(x_in, y_in)], dtype=float)

        # convert |X'| to meters for energy
        Xp_m = np.abs(Xp_in) * m_per_px
        E_keV = energies_keV_from_xp_m(
            xp_m=Xp_m,
            mass_number=int(A_massnum),
            charge_state=int(Z_charge),
            b_field_t=float(B_T),
            magnetic_length_m=float(LiB),
            drift_length_m=float(LfB),
        )

        # filter finite/positive energies
        good = np.isfinite(E_keV) & (E_keV > 0) & np.isfinite(weights)
        all_E_keV.append(E_keV[good])
        all_w.append(weights[good])
        polylines.append((x_in, y_in))

    if all_E_keV:
        E = np.concatenate(all_E_keV)
        Wts = np.concatenate(all_w)
    else:
        E = np.array([], dtype=float)
        Wts = np.array([], dtype=float)
    return E, Wts, polylines


# ---------- Span utilities ----------
def make_Xp_span_rot(
    groups: Iterable[dict[str, Any]],
    img: np.ndarray,
    x0_fit: float,
    y0_fit: float,
    theta_fit: float,
    gamma_fit: float = 0.0,
    delta_fit: float = 0.0,
    pad: float = 0.0,
) -> tuple[float, float]:
    """
    Compute overall X' span across groups, for reference curves.

    Parameters
    ----------
    groups :
        Iterable of objects/dicts each containing a ``'points'`` entry with
        pixel coordinates. Each ``points`` array is expected to have at least
        three columns ``[..., x, y]`` as in ``filtered_lines``.
    img : np.ndarray
        Image used as a fallback when no valid points are available.
    x0_fit, y0_fit : float
        Fitted shared origin in image coordinates.
    theta_fit : float
        Fitted rotation angle (radians) of the canonical parabola frame.
    gamma_fit : float, optional
        Global perspective parameter from the shared geometry model. The span
        calculation is determined by the rotated coordinates and does not
        depend on this value.
    delta_fit : float, optional
        Global linear tilt parameter from the shared geometry model. The span
        calculation is determined by the rotated coordinates and does not
        depend on this value.
    pad : float, optional
        Extra margin added on both sides of the returned X' interval.

    Returns
    -------
    (Xp_min, Xp_max) : tuple[float, float]
        Overall X' span in the rotated frame.
    """
    _ = (gamma_fit, delta_fit)
    points_list: list[np.ndarray] = []
    for g in groups:
        pts = g["points"]
        if pts is None or len(pts) == 0:
            continue
        points_list.append(np.asarray(pts, dtype=float))

    if points_list:
        return xp_span_px_from_points(
            points_list=points_list,
            x0_fit=float(x0_fit),
            y0_fit=float(y0_fit),
            theta_fit=float(theta_fit),
            pad_px=float(pad),
        )
    return xp_span_px_from_image(
        image=img,
        x0_fit=float(x0_fit),
        y0_fit=float(y0_fit),
        theta_fit=float(theta_fit),
        pad_px=float(pad),
    )
