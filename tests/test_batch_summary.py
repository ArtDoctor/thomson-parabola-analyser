from pathlib import Path

from oblisk.reporting.batch_summary import (
    BatchImageStats,
    render_batch_summary_md,
    stats_from_run,
    write_batch_summary_md,
)
from oblisk.reporting.eval_report import ParabolaLineFitEvalRow, build_run_eval


def _result_with_eval(
    line_rows: list[ParabolaLineFitEvalRow],
    global_rmse: float | None,
    classified: list[dict],
    total_time: float = 1.0,
    bright_spot_yx: tuple[int, int] = (400, 120),
    peak_extraction_start_row: int = 190,
) -> dict:
    payload = build_run_eval(
        line_rows,
        global_rmse,
        classified,
        bright_spot_yx=bright_spot_yx,
        peak_extraction_start_row=peak_extraction_start_row,
    )
    return {
        "eval": payload.model_dump(mode="json"),
        "timings": {"total": total_time},
    }


def test_render_batch_summary_empty() -> None:
    text = render_batch_summary_md([])
    assert "No images processed" in text


def test_worst_images_ordering_by_rmse() -> None:
    classified = [
        {
            "a": 0.01,
            "label": "H^+",
            "mq_meas": 1.0,
            "candidates": [{"name": "H^+", "mq_target": 1.0, "rel_err": 0.0}],
        },
    ]
    line_row = ParabolaLineFitEvalRow(
        line_index=0,
        n_points=2,
        rmse_y_eff_px=1.0,
        mean_abs_residual_y_eff_px=0.5,
        rmse_y_eff_weighted_px=0.9,
    )

    def one(name: str, rmse: float) -> BatchImageStats:
        r = _result_with_eval([line_row], rmse, classified)
        return stats_from_run(Path(name), Path("out") / name, r)

    rows = [one("low.tif", 1.0), one("mid.tif", 3.0), one("high.tif", 5.0)]
    md = render_batch_summary_md(rows)
    assert "mean abs. rel. err. (ID'd)" in md
    assert "high.tif" in md
    hi = md.index("high.tif")
    mi = md.index("mid.tif")
    lo = md.index("low.tif")
    assert hi < mi < lo


def test_aggregate_counts_and_pooled_rel_err() -> None:
    classified_a = [
        {
            "a": 0.01,
            "label": "?",
            "mq_meas": 1.0,
            "candidates": [],
        },
        {
            "a": 0.02,
            "label": "C^6+",
            "mq_meas": 2.0,
            "candidates": [{"name": "C^6+", "mq_target": 2.0, "rel_err": 0.1}],
        },
    ]
    classified_b = [
        {
            "a": 0.03,
            "label": "O^8+",
            "mq_meas": 2.0,
            "candidates": [{"name": "O^8+", "mq_target": 2.0, "rel_err": 0.2}],
        },
    ]
    line_row = ParabolaLineFitEvalRow(
        line_index=0,
        n_points=2,
        rmse_y_eff_px=2.0,
        mean_abs_residual_y_eff_px=1.0,
        rmse_y_eff_weighted_px=1.5,
    )
    r1 = _result_with_eval([line_row], 2.0, classified_a)
    r2 = _result_with_eval([line_row], 2.0, classified_b)
    rows = [
        stats_from_run(Path("a.tif"), Path("o/a"), r1),
        stats_from_run(Path("b.tif"), Path("o/b"), r2),
    ]
    md = render_batch_summary_md(rows)
    assert "| Total parabolae (summed) | 3 |" in md
    assert "| Identified (summed) | 2 |" in md
    assert "| Unidentified (summed) | 1 |" in md
    pooled = (0.1 + 0.2) / 2.0
    assert f"{pooled:.4f}" in md


def test_write_batch_summary_md_creates_file(tmp_path: Path) -> None:
    classified = [
        {
            "a": 0.01,
            "label": "H^+",
            "mq_meas": 1.0,
            "candidates": [{"name": "H^+", "mq_target": 1.0, "rel_err": 0.0}],
        },
    ]
    line_row = ParabolaLineFitEvalRow(
        line_index=0,
        n_points=2,
        rmse_y_eff_px=1.0,
        mean_abs_residual_y_eff_px=0.5,
        rmse_y_eff_weighted_px=0.9,
    )
    r = _result_with_eval([line_row], 1.5, classified)
    row = stats_from_run(Path("x.tif"), tmp_path / "x", r)
    out = write_batch_summary_md(tmp_path, [row])
    assert out == tmp_path / "summary.md"
    assert out.is_file()
    body = out.read_text(encoding="utf-8")
    assert "x.tif" in body
    assert "# Batch processing summary" in body


def test_stats_fallback_rmse_from_line_fits() -> None:
    classified = [
        {
            "a": 0.01,
            "label": "?",
            "mq_meas": 1.0,
            "candidates": [],
        },
    ]
    line_rows = [
        ParabolaLineFitEvalRow(
            line_index=0,
            n_points=2,
            rmse_y_eff_px=10.0,
            mean_abs_residual_y_eff_px=1.0,
            rmse_y_eff_weighted_px=9.0,
        ),
        ParabolaLineFitEvalRow(
            line_index=1,
            n_points=2,
            rmse_y_eff_px=20.0,
            mean_abs_residual_y_eff_px=2.0,
            rmse_y_eff_weighted_px=18.0,
        ),
    ]
    payload = build_run_eval(
        line_rows,
        None,
        classified,
        bright_spot_yx=(300, 100),
        peak_extraction_start_row=100,
    )
    result = {
        "eval": payload.model_dump(mode="json"),
        "timings": {},
    }
    s = stats_from_run(Path("n.tif"), Path("o/n"), result)
    assert abs(s.effective_global_rmse_px - 15.0) < 1e-9
