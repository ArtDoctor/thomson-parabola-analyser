import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from oblisk.analysis.background import BackgroundRoi
from oblisk.analysis.geometry import PerspectiveReference
from oblisk.analysis.physics import (
    LfB_B,
    LiB,
    default_fallback_meters_per_pixel,
    default_primary_b_field_tesla,
)
from oblisk.analysis.species import CandidateMatch


def _empty_float_array() -> np.ndarray:
    return np.array([], dtype=float)


class SpectrumGeometry(BaseModel):
    x0_fit: float
    y0_fit: float
    theta_fit: float
    gamma_fit: float = 0.0
    delta_fit: float = 0.0
    k1_fit: float = 0.0
    k2_fit: float = 0.0
    img_center_x: float = 0.0
    img_center_y: float = 0.0
    img_diag: float = 1.0
    b_field_t: float = Field(default_factory=default_primary_b_field_tesla)
    magnetic_length_m: float = LiB
    drift_length_m: float = LfB_B
    meters_per_pixel: float = Field(default_factory=default_fallback_meters_per_pixel)
    perspective_reference: PerspectiveReference = Field(
        default_factory=PerspectiveReference
    )


class ClassifiedLine(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    a: float
    label: str
    candidates: list[CandidateMatch] = Field(default_factory=list)
    points: np.ndarray | None = None


class SamplePolyline(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    x: np.ndarray
    y: np.ndarray


def _longest_finite_polyline_segment(
    polyline: SamplePolyline,
) -> tuple[np.ndarray, np.ndarray] | None:
    x = np.asarray(polyline.x, dtype=float)
    y = np.asarray(polyline.y, dtype=float)
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


class IonSpectrum(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    label: str
    symbol: str
    mass_number: int
    charge_state: int
    a_rot: float
    nearby: list[str] = Field(default_factory=list)
    same_mq: list[str] = Field(default_factory=list)
    energies_keV: np.ndarray = Field(default_factory=_empty_float_array)
    weights: np.ndarray = Field(default_factory=_empty_float_array)
    normalized_signal: np.ndarray = Field(default_factory=_empty_float_array)
    polyline: SamplePolyline | None = None


class SpectraResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    background_roi: BackgroundRoi
    background_mean: float
    xp_bounds_px: tuple[float, float]
    energy_centers_keV: np.ndarray = Field(default_factory=_empty_float_array)
    spectra: list[IonSpectrum] = Field(default_factory=list)


class AbsoluteSpectrumCurve(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: int
    label: str
    nearby: list[str] = Field(default_factory=list)
    same_mq: list[str] = Field(default_factory=list)
    energies_keV: np.ndarray = Field(default_factory=_empty_float_array)
    weights: np.ndarray = Field(default_factory=_empty_float_array)
    y_abs: np.ndarray = Field(default_factory=_empty_float_array)
