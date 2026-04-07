"""Structured decisions emitted by the Thomson parabola analysis pipeline."""

from typing import Any, Literal

from pydantic import BaseModel


class QuadrantPixelCounts(BaseModel):
    TR: int
    TL: int
    BL: int
    BR: int


class OrientationDecision(BaseModel):
    kind: Literal["orientation"] = "orientation"
    origin_xy: tuple[int, int]
    quadrant_counts: QuadrantPixelCounts
    dominant_quadrant: str
    rotation_applied: Literal[
        "none",
        "rotate_90_ccw",
        "rotate_180",
        "rotate_90_cw",
        "skipped_empty_signal",
    ]


class ParityDecision(BaseModel):
    kind: Literal["parity"] = "parity"
    pixels_high_angle: int
    pixels_low_angle: int
    mirrored_fan_detected: bool
    horizontal_flip_applied: bool
    reorientation_after_horizontal_flip: OrientationDecision | None
    final_vertical_flip_applied: bool


class RunFlagsLog(BaseModel):
    kind: Literal["run_flags"] = "run_flags"
    use_experimental_a_for_hydrogen: bool
    use_denoise_unet: bool


class DetectionSettingsLog(BaseModel):
    kind: Literal["detection_settings"] = "detection_settings"
    denoise: bool
    denoise_kernel_size: int
    window: int
    prominence: int
    distance: int
    max_peak_distance: int
    min_line_length_1: int
    min_line_length_2: int
    max_x_gap: int
    direction_tol_px: int
    inner_margin_crop_px: int


class DetectorCropLog(BaseModel):
    kind: Literal["detector_crop"] = "detector_crop"
    x1: int
    y1: int
    x2: int
    y2: int
    yolo_confidence: float


class DenoiseLog(BaseModel):
    kind: Literal["denoise"] = "denoise"
    method: Literal["unet", "morph_open", "none"]
    unet_resize_scale: float | None


class PeakScanLog(BaseModel):
    kind: Literal["peak_scan"] = "peak_scan"
    brightest_spot_yx: tuple[int, int]
    peak_extraction_start_row: int
    peak_extraction_end_row: int


class LinePipelineStatsLog(BaseModel):
    kind: Literal["line_pipeline"] = "line_pipeline"
    num_lines_after_build: int
    num_lines_after_merge: int
    num_lines_after_smooth: int


class GlobalFitLog(BaseModel):
    kind: Literal["global_fit"] = "global_fit"
    x0_fit: float
    y0_fit: float
    theta_fit: float
    gamma_fit: float
    delta_fit: float
    k1_fit: float = 0.0
    k2_fit: float = 0.0
    per_line_curvatures_from_shared_vertex: list[float]


class CurvaturePeaksLog(BaseModel):
    kind: Literal["curvature_peaks"] = "curvature_peaks"
    good_a_values: list[float]
    score_grid_size: int
    prominence_rel: float
    height_rel: float
    min_distance: int


class XpSpanLog(BaseModel):
    kind: Literal["xp_span"] = "xp_span"
    xp_min: float
    xp_max: float


class HydrogenReferenceLog(BaseModel):
    kind: Literal["hydrogen_reference"] = "hydrogen_reference"
    hydrogen_a: float
    mode: str
    classification_match_tolerance: float
    isolated_lowest_a_peak: bool
    magnet_current_amps: float
    magnet_calibration: str


class SpectraIntegrationLog(BaseModel):
    kind: Literal["spectra_integration"] = "spectra_integration"
    integration_a_samples: int
    integration_windows_a: list[list[float]]
    background_roi_x0: int
    background_roi_x1: int
    background_roi_y0: int
    background_roi_y1: int


def dump_entries(entries: list[BaseModel]) -> list[dict[str, Any]]:
    return [entry.model_dump(mode="json") for entry in entries]


_LOG_ENTRY_BY_KIND: dict[str, type[BaseModel]] = {
    "orientation": OrientationDecision,
    "parity": ParityDecision,
    "run_flags": RunFlagsLog,
    "detection_settings": DetectionSettingsLog,
    "detector_crop": DetectorCropLog,
    "denoise": DenoiseLog,
    "peak_scan": PeakScanLog,
    "line_pipeline": LinePipelineStatsLog,
    "global_fit": GlobalFitLog,
    "curvature_peaks": CurvaturePeaksLog,
    "xp_span": XpSpanLog,
    "hydrogen_reference": HydrogenReferenceLog,
    "spectra_integration": SpectraIntegrationLog,
}


def load_log_entries_from_json(rows: list[dict[str, Any]]) -> list[BaseModel]:
    """Rebuild pipeline log entries from JSON-compatible dicts (see dump_entries)."""

    out: list[BaseModel] = []
    for row in rows:
        kind = row.get("kind")
        if kind is None:
            raise ValueError("log entry missing 'kind'")
        cls = _LOG_ENTRY_BY_KIND.get(str(kind))
        if cls is None:
            raise ValueError(f"unknown log entry kind: {kind!r}")
        out.append(cls.model_validate(row))
    return out
