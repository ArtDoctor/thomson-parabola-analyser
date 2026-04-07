import os
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from oblisk.analysis.species import (
    DEFAULT_CLASSIFICATION_ELEMENTS,
    normalize_classification_elements,
)


_YOLO_MODEL_PATH_ENV = "OBLISK_YOLO_MODEL_PATH"
_UNET_CHECKPOINT_PATH_ENV = "OBLISK_UNET_CHECKPOINT_PATH"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


class MagnetCalibrationRow(BaseModel):
    current_amps: float
    field_millitesla: float


def default_magnet_calibration_rows() -> list[MagnetCalibrationRow]:
    return [
        MagnetCalibrationRow(current_amps=2.10, field_millitesla=154.0),
        MagnetCalibrationRow(current_amps=1.88, field_millitesla=144.0),
        MagnetCalibrationRow(current_amps=1.65, field_millitesla=128.0),
        MagnetCalibrationRow(current_amps=1.45, field_millitesla=112.0),
        MagnetCalibrationRow(current_amps=1.30, field_millitesla=104.0),
        MagnetCalibrationRow(current_amps=1.20, field_millitesla=92.0),
        MagnetCalibrationRow(current_amps=1.00, field_millitesla=82.0),
        MagnetCalibrationRow(current_amps=0.90, field_millitesla=74.5),
        MagnetCalibrationRow(current_amps=2.30, field_millitesla=167.0),
        MagnetCalibrationRow(current_amps=2.20, field_millitesla=162.0),
        MagnetCalibrationRow(current_amps=2.10, field_millitesla=157.0),
    ]


class Settings(BaseModel):
    # Denoising
    denoise: bool = True
    denoise_kernel_size: int = 5

    # Image processing
    init_point_shift: list[int] = [-50, 0]
    # After detector ROI, trim this many pixels from each edge (0 = off).
    inner_margin_crop_px: int = Field(default=50, ge=0)

    # Peak detection
    window: int = 15
    prominence: int = 5
    distance: int = 10
    max_peak_distance: int = 30

    # Merging
    min_line_length_1: int = 10
    min_line_length_2: int = 90
    max_x_gap: int = 10
    direction_penalty_weight: float = 3.0
    direction_tol_px: int = 3
    merge_output_length_cap: int = 55

    # Optional post-pass stitching (set to override defaults)
    stitch_max_rows: int | None = None
    stitch_max_dx: float | None = None
    stitch_max_slope_diff: float | None = None

    # Spectrometer geometry (optional overrides for UI / future pipeline wiring)
    spec_E_kVm: float | None = None
    spec_LiE_cm: float | None = None
    spec_LfE_cm: float | None = None
    spec_B_mT: float | None = None
    spec_LiB_cm: float | None = None
    spec_LfB_cm: float | None = None
    spec_detector_size_cm: float | None = None

    # Magnet I→B table (lab calibration) and trial currents for classification
    magnet_current_standard_amps: float = 1.91
    magnet_current_phosphor_amps: float = 0.8
    magnet_calibration: list[MagnetCalibrationRow] = Field(
        default_factory=default_magnet_calibration_rows,
    )
    # Fallback scale when only pixel grid size is known (e.g. preprocessor default).
    fallback_meters_per_pixel: float = Field(default=50e-6, gt=0)

    classification_element_symbols: list[str] = Field(
        default_factory=lambda: list(DEFAULT_CLASSIFICATION_ELEMENTS),
    )

    @field_validator("magnet_calibration")
    @classmethod
    def _magnet_calibration_min_two(
        cls,
        value: list[MagnetCalibrationRow],
    ) -> list[MagnetCalibrationRow]:
        if len(value) < 2:
            msg = "magnet_calibration must contain at least two points"
            raise ValueError(msg)
        return value

    @field_validator("classification_element_symbols", mode="before")
    @classmethod
    def _coerce_classification_elements(cls, value: object) -> list[str]:
        if value is None:
            return list(DEFAULT_CLASSIFICATION_ELEMENTS)
        if not isinstance(value, list):
            err = "classification_element_symbols must be a list of strings"
            raise TypeError(err)
        return normalize_classification_elements([str(x) for x in value])


class RuntimeModelPaths(BaseModel):
    yolo_detector: Path
    unet_checkpoint: Path


def default_yolo_detector_path() -> Path:
    override = os.environ.get(_YOLO_MODEL_PATH_ENV)
    if override:
        return Path(override).expanduser()
    return repo_root() / "yolo-tune" / "thomson-cutter.onnx"


def default_unet_checkpoint_path() -> Path:
    override = os.environ.get(_UNET_CHECKPOINT_PATH_ENV)
    if override:
        return Path(override).expanduser()
    return repo_root() / "unet-denoiser" / "unet_denoise_best.pth"


def runtime_model_paths() -> RuntimeModelPaths:
    return RuntimeModelPaths(
        yolo_detector=default_yolo_detector_path(),
        unet_checkpoint=default_unet_checkpoint_path(),
    )
