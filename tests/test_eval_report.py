from oblisk.reporting.eval_report import ParabolaLineFitEvalRow, build_run_eval


def test_build_run_eval_species_counts_and_deltas() -> None:
    line_rows = [
        ParabolaLineFitEvalRow(
            line_index=0,
            n_points=3,
            rmse_y_eff_px=1.0,
            mean_abs_residual_y_eff_px=0.8,
            rmse_y_eff_weighted_px=0.9,
        ),
    ]
    classified = [
        {
            "a": 0.002,
            "label": "?",
            "mq_meas": 2.1,
            "candidates": [],
        },
        {
            "a": 0.004,
            "label": "Si^14+",
            "mq_meas": 2.0,
            "candidates": [
                {"name": "Si^14+", "mq_target": 2.0, "rel_err": 0.01},
            ],
        },
        {
            "a": 0.006,
            "label": "C^6+",
            "mq_meas": 2.01,
            "candidates": [
                {"name": "C^6+", "mq_target": 2.0, "rel_err": 0.005},
            ],
        },
    ]
    bright_y, bright_x = 310, 420
    start_row = bright_y - 210
    payload = build_run_eval(
        line_rows,
        0.95,
        classified,
        bright_spot_yx=(bright_y, bright_x),
        peak_extraction_start_row=start_row,
    )
    ibs = payload.initial_to_bright_spot
    assert ibs is not None
    assert ibs.distance_px == 210.0
    assert payload.species_summary.n_classified_parabolae == 3
    assert payload.species_summary.n_unidentified == 1
    assert payload.species_summary.n_carbon_identified == 1
    assert len(payload.species_identified) == 2
    si_row = next(r for r in payload.species_identified if r.label == "Si^14+")
    assert si_row.delta_mq == 0.0
    c_row = next(r for r in payload.species_identified if r.label == "C^6+")
    assert abs(c_row.delta_mq - 0.01) < 1e-12
