import pandas as pd  # type: ignore[import-untyped]
import numpy as np
from PIL import Image

from oblisk.analysis.physics import DETECTOR_SIZE_M

GRID_SIZE: int = 1200
METERS_PER_PIXEL: float = DETECTOR_SIZE_M / float(GRID_SIZE)

_HALF_WIDTH_M: float = DETECTOR_SIZE_M * 0.5
_DEFAULT_LIM_X: tuple[float, float] = (-_HALF_WIDTH_M, _HALF_WIDTH_M)
_DEFAULT_LIM_Y: tuple[float, float] = (-_HALF_WIDTH_M, _HALF_WIDTH_M)


def detector_limits_from_spot(
    spot_col: int,
    spot_row: int,
    grid_size: int = GRID_SIZE,
    m_per_px: float = METERS_PER_PIXEL,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return (lim_x, lim_y) in meters for histogram2d.

    Maps simulation (x,y)=(0,0) at the beam origin to pixel (spot_col, spot_row)
    after the vertical flip in hits_to_img (matches dataset rendering).
    """
    phys_min_x = -spot_col * m_per_px
    phys_max_x = (grid_size - spot_col) * m_per_px
    col_y = grid_size - spot_row
    phys_min_y = -col_y * m_per_px
    phys_max_y = (grid_size - col_y) * m_per_px
    return (phys_min_x, phys_max_x), (phys_min_y, phys_max_y)


def hits_to_img(
    hits_file: str,
    brightness_scale: float = 1.0,
    lim_x: tuple[float, float] = _DEFAULT_LIM_X,
    lim_y: tuple[float, float] = _DEFAULT_LIM_Y,
) -> Image.Image:
    # 1. Load the data
    try:
        df = pd.read_csv(hits_file, sep=r"\s+", names=["Y", "X", "Energy", "ParticleName"])
    except FileNotFoundError as exc:
        raise FileNotFoundError(hits_file) from exc

    all_X = df["X"].values
    all_Y = df["Y"].values

    # 2. 2D histogram: GRID_SIZE² bins, span matches main analytical scale (m_per_px = DETECTOR_SIZE_M / GRID_SIZE)
    H, _, _ = np.histogram2d(
        all_Y,
        all_X,
        bins=GRID_SIZE,
        range=[lim_y, lim_x],
    )

    # 3. Normalize the data to an 8-bit grayscale range (0-255)
    max_val = H.max()
    if max_val > 0:
        H_normalized = (H / max_val) * 255.0 * brightness_scale
        H_normalized = np.clip(H_normalized, 0, 255)
    else:
        H_normalized = H

    H_uint8 = H_normalized.astype(np.uint8)

    # 4. Flip vertically so physical Y=0 is at the bottom of the image.
    H_img_array = np.flipud(H_uint8)
    img = Image.fromarray(H_img_array, mode="L")
    return img
