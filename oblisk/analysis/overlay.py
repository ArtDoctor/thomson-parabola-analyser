from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from oblisk.analysis.geometry import (
    PerspectiveReference,
    distort_points,
    finite_polyline_segments,
    from_rotated_frame,
    tilt_inverse_Yp,
    visible_polyline_with_nan_breaks,
)


class ProjectionGeometry(BaseModel):
    model_config = ConfigDict(frozen=True)

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
    perspective_reference: PerspectiveReference = Field(
        default_factory=PerspectiveReference
    )


class CurvePointPayload(BaseModel):
    x: float
    y: float


class CurveSegment(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    x: np.ndarray
    y: np.ndarray

    def to_payload(self) -> list[CurvePointPayload]:
        return [
            CurvePointPayload(x=float(xx), y=float(yy))
            for xx, yy in zip(self.x.tolist(), self.y.tolist())
        ]


class ProjectedCurvePayload(BaseModel):
    segments: list[list[CurvePointPayload]]
    label_anchor: list[float] | None = None


class ClassifiedOverlayCurvePayload(BaseModel):
    entry_index: int
    segments: list[list[CurvePointPayload]]
    label_anchor: list[float] | None = None


class SamplingOverlayCurvePayload(BaseModel):
    label: str
    segments: list[list[CurvePointPayload]]


class ProjectedCurve(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    segments: list[CurveSegment]
    visible_x: np.ndarray
    visible_y: np.ndarray
    label_anchor: tuple[float, float] | None = None

    def to_payload(self) -> ProjectedCurvePayload:
        anchor = None
        if self.label_anchor is not None:
            anchor = [float(self.label_anchor[0]), float(self.label_anchor[1])]
        return ProjectedCurvePayload(
            segments=[segment.to_payload() for segment in self.segments],
            label_anchor=anchor,
        )


def _distort_if_needed(
    x: np.ndarray,
    y: np.ndarray,
    geometry: ProjectionGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    if abs(geometry.k1_fit) <= 1e-15 and abs(geometry.k2_fit) <= 1e-15:
        return x, y
    radius_norm = max(float(geometry.img_diag) * 0.5, 1.0)
    return distort_points(
        x,
        y,
        float(geometry.img_center_x),
        float(geometry.img_center_y),
        float(geometry.k1_fit),
        radius_norm,
        k2=float(geometry.k2_fit),
    )


def _project_curve_points(
    xp_values: np.ndarray,
    a_value: float,
    geometry: ProjectionGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    yp_values = tilt_inverse_Yp(
        xp_values,
        float(a_value),
        float(geometry.gamma_fit),
        float(geometry.delta_fit),
        perspective_reference=geometry.perspective_reference,
    )
    x_img, y_img = from_rotated_frame(
        xp_values,
        yp_values,
        float(geometry.x0_fit),
        float(geometry.y0_fit),
        float(geometry.theta_fit),
    )
    return _distort_if_needed(x_img, y_img, geometry)


def project_origin_point(
    geometry: ProjectionGeometry,
) -> tuple[float, float]:
    x_img, y_img = _project_curve_points(
        np.asarray([0.0], dtype=float),
        0.0,
        geometry,
    )
    return float(x_img[0]), float(y_img[0])


def _default_label_anchor(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    placed: list[tuple[float, float]],
    fraction: float = 0.75,
    min_dist: float = 30.0,
    step: int = 5,
    max_offset: int = 80,
) -> tuple[float, float]:
    n = len(x_arr)
    if n == 0:
        return 0.0, 0.0
    start_idx = max(0, min(int(round((n - 1) * fraction)), n - 1))
    if not placed:
        return float(x_arr[start_idx]), float(y_arr[start_idx])

    min_dist2 = float(min_dist * min_dist)

    def _too_close(index: int) -> bool:
        xx = float(x_arr[index])
        yy = float(y_arr[index])
        return any((xx - px) ** 2 + (yy - py) ** 2 < min_dist2 for px, py in placed)

    if not _too_close(start_idx):
        return float(x_arr[start_idx]), float(y_arr[start_idx])

    for offset in range(1, max_offset + 1):
        for sign in (+1, -1):
            idx = start_idx + sign * offset * step
            if 0 <= idx < n and not _too_close(idx):
                return float(x_arr[idx]), float(y_arr[idx])
    return float(x_arr[start_idx]), float(y_arr[start_idx])


def project_parabola_curve(
    *,
    a_value: float,
    geometry: ProjectionGeometry,
    xp_min: float,
    xp_max: float,
    image_shape: tuple[int, int],
    n_samples: int = 600,
) -> ProjectedCurve | None:
    xp_values = np.linspace(float(xp_min), float(xp_max), int(n_samples))
    x_img, y_img = _project_curve_points(xp_values, float(a_value), geometry)

    height, width = image_shape
    visible_polyline = visible_polyline_with_nan_breaks(x_img, y_img, width, height)
    if visible_polyline is None:
        return None

    seg_arrays = finite_polyline_segments(*visible_polyline)
    segments = [
        CurveSegment(x=np.asarray(seg_x, dtype=float), y=np.asarray(seg_y, dtype=float))
        for seg_x, seg_y in seg_arrays
        if len(seg_x) >= 2
    ]
    if not segments:
        return None

    visible_x = np.concatenate([segment.x for segment in segments])
    visible_y = np.concatenate([segment.y for segment in segments])
    return ProjectedCurve(
        segments=segments,
        visible_x=visible_x,
        visible_y=visible_y,
    )


def build_classified_projected_curves(
    *,
    classified: list[dict[str, Any]],
    geometry: ProjectionGeometry,
    xp_span: tuple[float, float],
    image_shape: tuple[int, int],
    n_samples: int = 600,
) -> list[tuple[int, ProjectedCurve]]:
    built: list[tuple[int, ProjectedCurve]] = []
    placed: list[tuple[float, float]] = []

    for entry_index, entry in enumerate(classified):
        curve = project_parabola_curve(
            a_value=float(entry["a"]),
            geometry=geometry,
            xp_min=float(xp_span[0]),
            xp_max=float(xp_span[1]),
            image_shape=image_shape,
            n_samples=n_samples,
        )
        if curve is None:
            continue
        curve.label_anchor = _default_label_anchor(
            curve.visible_x,
            curve.visible_y,
            placed=placed,
        )
        placed.append(curve.label_anchor)
        built.append((entry_index, curve))
    return built


def build_detected_projected_curves(
    *,
    a_values: np.ndarray,
    geometry: ProjectionGeometry,
    xp_span: tuple[float, float],
    image_shape: tuple[int, int],
    n_samples: int = 800,
) -> list[tuple[float, ProjectedCurve]]:
    output: list[tuple[float, ProjectedCurve]] = []
    for a_value in np.asarray(a_values, dtype=float):
        curve = project_parabola_curve(
            a_value=float(a_value),
            geometry=geometry,
            xp_min=float(xp_span[0]),
            xp_max=float(xp_span[1]),
            image_shape=image_shape,
            n_samples=n_samples,
        )
        if curve is not None:
            output.append((float(a_value), curve))
    return output


def project_polyline_segments(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
) -> list[CurveSegment]:
    return [
        CurveSegment(x=np.asarray(seg_x, dtype=float), y=np.asarray(seg_y, dtype=float))
        for seg_x, seg_y in finite_polyline_segments(x_arr, y_arr)
        if len(seg_x) >= 2
    ]


def serialize_classified_overlay_curves(
    projected_curves: list[tuple[int, ProjectedCurve]],
) -> list[ClassifiedOverlayCurvePayload]:
    return [
        ClassifiedOverlayCurvePayload(
            entry_index=int(entry_index),
            **curve.to_payload().model_dump(mode="json"),
        )
        for entry_index, curve in projected_curves
    ]


def serialize_sampling_overlay_curves(
    *,
    labels: list[str],
    polylines: list[tuple[np.ndarray, np.ndarray] | None],
) -> list[SamplingOverlayCurvePayload]:
    output: list[SamplingOverlayCurvePayload] = []
    for label, polyline in zip(labels, polylines):
        if polyline is None:
            continue
        segments = project_polyline_segments(polyline[0], polyline[1])
        if not segments:
            continue
        output.append(
            SamplingOverlayCurvePayload(
                label=str(label),
                segments=[segment.to_payload() for segment in segments],
            )
        )
    return output
