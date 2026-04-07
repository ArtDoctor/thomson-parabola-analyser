import math
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from oblisk.reporting.eval_report import (
    RunEvalPayload,
    SpeciesIdentifiedEvalRow,
    SpeciesSummaryEval,
)


class BatchImageStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    image_path: Path
    output_subdir: Path
    effective_global_rmse_px: float
    global_rmse_y_eff_px: float | None
    n_classified_parabolae: int
    n_unidentified: int
    n_identified: int
    pct_identified: float
    mean_abs_rel_err_identified: float | None
    species_abs_rel_errors: tuple[float, ...]
    n_carbon_identified: int
    total_wall_time_s: float | None


def _effective_global_rmse_px(eval_payload: RunEvalPayload) -> float:
    global_rmse = eval_payload.global_rmse_y_eff_px
    if global_rmse is not None and math.isfinite(global_rmse):
        return float(global_rmse)
    rows = eval_payload.parabola_line_fits
    if not rows:
        return float("nan")
    return sum(row.rmse_y_eff_px for row in rows) / float(len(rows))


def _mean_abs_rel_err(rows: list[SpeciesIdentifiedEvalRow]) -> float | None:
    if not rows:
        return None
    return sum(abs(row.rel_err) for row in rows) / float(len(rows))


def stats_from_run(
    image_path: Path,
    output_subdir: Path,
    result: dict,
) -> BatchImageStats:
    eval_payload = RunEvalPayload.model_validate(result["eval"])
    summary: SpeciesSummaryEval = eval_payload.species_summary
    n_par = summary.n_classified_parabolae
    n_un = summary.n_unidentified
    n_id = n_par - n_un
    pct = (100.0 * float(n_id) / float(n_par)) if n_par > 0 else 0.0
    timings = result.get("timings") or {}
    total_t = timings.get("total")
    wall: float | None = float(total_t) if isinstance(total_t, (int, float)) else None
    rels = tuple(abs(row.rel_err) for row in eval_payload.species_identified)
    return BatchImageStats(
        image_path=image_path,
        output_subdir=output_subdir,
        effective_global_rmse_px=_effective_global_rmse_px(eval_payload),
        global_rmse_y_eff_px=eval_payload.global_rmse_y_eff_px,
        n_classified_parabolae=n_par,
        n_unidentified=n_un,
        n_identified=n_id,
        pct_identified=pct,
        mean_abs_rel_err_identified=_mean_abs_rel_err(eval_payload.species_identified),
        species_abs_rel_errors=rels,
        n_carbon_identified=summary.n_carbon_identified,
        total_wall_time_s=wall,
    )


def _worst_sort_key(row: BatchImageStats) -> tuple[float, float, float]:
    rmse = row.effective_global_rmse_px
    rmse_key = rmse if math.isfinite(rmse) else float("-inf")
    rel = row.mean_abs_rel_err_identified or 0.0
    return (rmse_key, float(row.n_unidentified), rel)


def render_batch_summary_md(rows: list[BatchImageStats]) -> str:
    if not rows:
        return "# Batch processing summary\n\nNo images processed.\n"

    n_img = len(rows)
    tot_par = sum(row.n_classified_parabolae for row in rows)
    tot_un = sum(row.n_unidentified for row in rows)
    tot_id = sum(row.n_identified for row in rows)
    pct_all = (100.0 * float(tot_id) / float(tot_par)) if tot_par > 0 else 0.0
    tot_c = sum(row.n_carbon_identified for row in rows)

    rmses_finite = [
        row.effective_global_rmse_px
        for row in rows
        if math.isfinite(row.effective_global_rmse_px)
    ]
    mean_rmse = (
        sum(rmses_finite) / float(len(rmses_finite)) if rmses_finite else float("nan")
    )

    pooled_abs_errs: list[float] = []
    for row in rows:
        if row.mean_abs_rel_err_identified is not None:
            pooled_abs_errs.append(row.mean_abs_rel_err_identified)

    mean_of_image_mean_rel = (
        sum(pooled_abs_errs) / float(len(pooled_abs_errs))
        if pooled_abs_errs
        else float("nan")
    )

    all_species_abs = [error for row in rows for error in row.species_abs_rel_errors]
    pooled_species_mean_abs = (
        sum(all_species_abs) / float(len(all_species_abs))
        if all_species_abs
        else float("nan")
    )

    lines: list[str] = [
        "# Batch processing summary",
        "",
        f"Analyzed **{n_img}** images.",
        "",
        "## Species classification (aggregate)",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Total parabolae (summed) | {tot_par} |",
        f"| Identified (summed) | {tot_id} |",
        f"| Unidentified (summed) | {tot_un} |",
        f"| Share of parabolae identified | {pct_all:.1f}% |",
        f"| Carbon-tagged parabolae (summed) | {tot_c} |",
        "",
        "## Fit quality (averages over images)",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    if math.isfinite(mean_rmse):
        lines.append(f"| Mean effective global RMSE (px) | {mean_rmse:.4f} |")
    else:
        lines.append("| Mean effective global RMSE (px) | n/a |")
    if math.isfinite(mean_of_image_mean_rel):
        lines.append(
            "| Mean abs. rel. error (mean of per-image means) | "
            f"{mean_of_image_mean_rel:.4f} |"
        )
    else:
        lines.append("| Mean abs. rel. error (mean of per-image means) | n/a |")
    if math.isfinite(pooled_species_mean_abs):
        lines.append(
            "| Mean abs. rel. error (all identified species pooled) | "
            f"{pooled_species_mean_abs:.4f} |"
        )
    else:
        lines.append("| Mean abs. rel. error (all identified species pooled) | n/a |")

    times = [row.total_wall_time_s for row in rows if row.total_wall_time_s is not None]
    if times:
        lines.extend(
            [
                f"| Total wall time (sum) | {sum(times):.2f} s |",
                f"| Mean wall time per image | {sum(times) / float(len(times)):.2f} s |",
            ]
        )

    lines.extend(
        [
            "",
            "## Worst images",
            "",
            "_Ranked by effective global RMSE (px), then unidentified count, then "
            "mean abs. rel. err. on identified species._",
            "",
        ]
    )

    ranked = sorted(rows, key=_worst_sort_key, reverse=True)
    worst_n = ranked[: min(10, len(ranked))]

    lines.extend(
        [
            "| Rank | Image | RMSE (px) | Unidentified / total | % identified | "
            "mean abs. rel. err. (ID'd) |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for index, worst in enumerate(worst_n, start=1):
        rmse_s = (
            f"{worst.effective_global_rmse_px:.4f}"
            if math.isfinite(worst.effective_global_rmse_px)
            else "n/a"
        )
        rel_s = (
            f"{worst.mean_abs_rel_err_identified:.4f}"
            if worst.mean_abs_rel_err_identified is not None
            else "—"
        )
        lines.append(
            f"| {index} | `{worst.image_path.name}` | {rmse_s} | "
            f"{worst.n_unidentified} / {worst.n_classified_parabolae} | "
            f"{worst.pct_identified:.1f}% | {rel_s} |"
        )

    lines.extend(["", "## Output locations", ""])
    for row in sorted(rows, key=lambda current: str(current.image_path)):
        lines.append(f"- `{row.image_path}` → `{row.output_subdir}`")

    lines.append("")
    return "\n".join(lines)


def write_batch_summary_md(summary_parent: Path, rows: list[BatchImageStats]) -> Path:
    summary_parent.mkdir(parents=True, exist_ok=True)
    path = summary_parent / "summary.md"
    path.write_text(render_batch_summary_md(rows), encoding="utf-8")
    return path
