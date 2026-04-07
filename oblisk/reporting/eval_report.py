import math
from typing import Any

from pydantic import BaseModel

from oblisk.analysis.species import parse_species


class ParabolaLineFitEvalRow(BaseModel):
    line_index: int
    n_points: int
    rmse_y_eff_px: float
    mean_abs_residual_y_eff_px: float
    rmse_y_eff_weighted_px: float


class SpeciesSummaryEval(BaseModel):
    n_classified_parabolae: int
    n_unidentified: int
    n_carbon_identified: int


class SpeciesIdentifiedEvalRow(BaseModel):
    parabola_index: int
    label: str
    a: float
    mq_meas: float
    mq_target: float
    delta_mq: float
    rel_err: float


class InitialToBrightSpotEval(BaseModel):
    """
    Peak extraction starts at (peak_extraction_start_row, bright_spot column).
    Bright spot is the argmax-intensity pixel (row, col) in standardized coords.
    """

    bright_spot_yx: tuple[int, int]
    initial_spot_yx: tuple[int, int]
    distance_px: float


def make_initial_to_bright_spot_eval(
    bright_spot_yx: tuple[int, int],
    peak_extraction_start_row: int,
) -> InitialToBrightSpotEval:
    y_b, x_b = bright_spot_yx
    y_i = peak_extraction_start_row
    x_i = x_b
    dist = math.hypot(float(x_b - x_i), float(y_b - y_i))
    return InitialToBrightSpotEval(
        bright_spot_yx=bright_spot_yx,
        initial_spot_yx=(y_i, x_i),
        distance_px=float(dist),
    )


class RunEvalPayload(BaseModel):
    parabola_line_fits: list[ParabolaLineFitEvalRow]
    global_rmse_y_eff_px: float | None
    species_summary: SpeciesSummaryEval
    species_identified: list[SpeciesIdentifiedEvalRow]
    initial_to_bright_spot: InitialToBrightSpotEval | None = None


def build_run_eval(
    line_fit_rows: list[ParabolaLineFitEvalRow],
    global_rmse_y_eff_px: float | None,
    classified: list[dict[str, Any]],
    bright_spot_yx: tuple[int, int],
    peak_extraction_start_row: int,
) -> RunEvalPayload:
    n_par = len(classified)
    n_unidentified = sum(1 for row in classified if row.get("label") == "?")
    n_c = 0
    for row in classified:
        label = row.get("label", "?")
        if label == "?":
            continue
        sym, _q = parse_species(str(label))
        if sym == "C":
            n_c += 1

    identified: list[SpeciesIdentifiedEvalRow] = []
    for idx, row in enumerate(classified):
        label = row.get("label", "?")
        if label == "?":
            continue
        candidates = row.get("candidates") or []
        if not candidates:
            continue
        best = candidates[0]
        mq_meas = float(row.get("mq_meas", float("nan")))
        mq_target = float(best["mq_target"])
        identified.append(
            SpeciesIdentifiedEvalRow(
                parabola_index=idx,
                label=str(label),
                a=float(row["a"]),
                mq_meas=mq_meas,
                mq_target=mq_target,
                delta_mq=mq_meas - mq_target,
                rel_err=float(best["rel_err"]),
            )
        )

    return RunEvalPayload(
        parabola_line_fits=line_fit_rows,
        global_rmse_y_eff_px=global_rmse_y_eff_px,
        species_summary=SpeciesSummaryEval(
            n_classified_parabolae=n_par,
            n_unidentified=n_unidentified,
            n_carbon_identified=n_c,
        ),
        species_identified=identified,
        initial_to_bright_spot=make_initial_to_bright_spot_eval(
            bright_spot_yx,
            peak_extraction_start_row,
        ),
    )
