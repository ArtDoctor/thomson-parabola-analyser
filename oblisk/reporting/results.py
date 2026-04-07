from typing import Any, cast

import numpy as np
from pydantic import BaseModel

from oblisk.analysis.overlay import ProjectionGeometry, project_origin_point
from oblisk.analysis.spectra import SpectrumGeometry
from oblisk.reporting.eval_report import RunEvalPayload


def serialize_for_json(obj: object) -> object:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize_for_json(v) for v in obj]
    return obj


class ResultCandidatePayload(BaseModel):
    name: str
    mq_target: float
    rel_err: float


class ResultClassifiedPayload(BaseModel):
    a: float
    label: str
    mq_meas: float
    candidates: list[ResultCandidatePayload]


class ResultGeometryPayload(BaseModel):
    x0_fit: float
    y0_fit: float
    common_vertex_x: float
    common_vertex_y: float
    theta_fit: float
    gamma_fit: float
    delta_fit: float
    k1_fit: float
    k2_fit: float
    xp_min: float | None = None
    xp_max: float | None = None


class ResultSpectrumPayload(BaseModel):
    label: str
    energies_keV: object
    weights: object


class ResultPayload(BaseModel):
    classified: list[ResultClassifiedPayload]
    geometry: ResultGeometryPayload
    spectra: list[ResultSpectrumPayload]
    overlays: dict[str, object]
    algorithm_log: list[dict[str, Any]]
    eval: dict[str, Any]
    timings: dict[str, float]


def build_result_json(
    classified: list[dict[str, Any]],
    spectra_result: object,
    geometry: SpectrumGeometry,
    overlays_or_timings: dict[str, object] | dict[str, float],
    timings_or_algorithm_log: dict[str, float] | list[dict[str, Any]],
    algorithm_log_or_eval_payload: list[dict[str, Any]] | RunEvalPayload,
    eval_payload: RunEvalPayload | None = None,
    xp_plot_range: tuple[float, float] | None = None,
) -> dict[str, Any]:
    overlays: dict[str, object]
    timings: dict[str, float]
    algorithm_log: list[dict[str, Any]]
    eval_payload_final: RunEvalPayload
    if eval_payload is None:
        overlays = {}
        timings = cast(dict[str, float], overlays_or_timings)
        algorithm_log = cast(list[dict[str, Any]], timings_or_algorithm_log)
        eval_payload_final = cast(RunEvalPayload, algorithm_log_or_eval_payload)
    else:
        overlays = cast(dict[str, object], overlays_or_timings)
        timings = cast(dict[str, float], timings_or_algorithm_log)
        algorithm_log = cast(list[dict[str, Any]], algorithm_log_or_eval_payload)
        eval_payload_final = eval_payload

    classified_ser: list[ResultClassifiedPayload] = []
    for match in classified:
        classified_ser.append(
            ResultClassifiedPayload(
                a=float(match["a"]),
                label=str(match.get("label", "?")),
                mq_meas=float(match.get("mq_meas", float("nan"))),
                candidates=[
                    ResultCandidatePayload(
                        name=str(candidate["name"]),
                        mq_target=float(candidate["mq_target"]),
                        rel_err=float(candidate["rel_err"]),
                    )
                    for candidate in match.get("candidates", [])
                ],
            )
        )

    proj_for_vertex = ProjectionGeometry(
        x0_fit=geometry.x0_fit,
        y0_fit=geometry.y0_fit,
        theta_fit=geometry.theta_fit,
        gamma_fit=geometry.gamma_fit,
        delta_fit=geometry.delta_fit,
        k1_fit=geometry.k1_fit,
        k2_fit=geometry.k2_fit,
        img_center_x=geometry.img_center_x,
        img_center_y=geometry.img_center_y,
        img_diag=geometry.img_diag,
        perspective_reference=geometry.perspective_reference,
    )
    origin_x, origin_y = project_origin_point(proj_for_vertex)
    geometry_ser = ResultGeometryPayload(
        x0_fit=geometry.x0_fit,
        y0_fit=geometry.y0_fit,
        common_vertex_x=float(origin_x),
        common_vertex_y=float(origin_y),
        theta_fit=geometry.theta_fit,
        gamma_fit=geometry.gamma_fit,
        delta_fit=geometry.delta_fit,
        k1_fit=geometry.k1_fit,
        k2_fit=geometry.k2_fit,
    )
    if xp_plot_range is not None:
        geometry_ser.xp_min = float(xp_plot_range[0])
        geometry_ser.xp_max = float(xp_plot_range[1])

    spectra_ser: list[ResultSpectrumPayload] = []
    if hasattr(spectra_result, "spectra"):
        for spectrum in spectra_result.spectra:
            spectra_ser.append(
                ResultSpectrumPayload(
                    label=str(spectrum.label),
                    energies_keV=serialize_for_json(spectrum.energies_keV),
                    weights=serialize_for_json(spectrum.weights),
                )
            )

    result_payload = ResultPayload(
        classified=classified_ser,
        geometry=geometry_ser,
        spectra=spectra_ser,
        overlays=overlays,
        algorithm_log=algorithm_log,
        eval=eval_payload_final.model_dump(mode="json"),
        timings={
            **{key: round(value, 4) for key, value in timings.items()},
            "total": round(sum(timings.values()), 4),
        },
    )
    return result_payload.model_dump(mode="json")
