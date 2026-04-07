from oblisk.reporting.eval_report import (
    InitialToBrightSpotEval,
    ParabolaLineFitEvalRow,
    RunEvalPayload,
    SpeciesIdentifiedEvalRow,
    SpeciesSummaryEval,
)
from oblisk.analysis.spectra import SpectrumGeometry
from oblisk.reporting.results import build_result_json


def test_result_json_includes_timings() -> None:
    classified = [{"a": 0.001, "label": "H^1+", "candidates": []}]
    geometry = SpectrumGeometry(
        x0_fit=100.0,
        y0_fit=50.0,
        theta_fit=0.0,
        gamma_fit=0.0,
        delta_fit=0.0,
    )
    spectra_result = type("SpectraResult", (), {"spectra": []})()
    timings = {
        "load_crop": 0.1,
        "denoise": 0.05,
        "peak_extraction": 0.2,
        "build_lines": 0.1,
        "merge_lines": 0.15,
        "smooth_lines": 0.02,
        "fit_origin": 1.5,
        "score_parabolas": 0.8,
        "detect_good_a": 0.01,
        "get_xp_range": 0.05,
        "classify": 0.001,
        "build_spectra": 0.3,
    }
    eval_payload = RunEvalPayload(
        parabola_line_fits=[
            ParabolaLineFitEvalRow(
                line_index=0,
                n_points=10,
                rmse_y_eff_px=0.5,
                mean_abs_residual_y_eff_px=0.4,
                rmse_y_eff_weighted_px=0.45,
            ),
        ],
        global_rmse_y_eff_px=0.48,
        species_summary=SpeciesSummaryEval(
            n_classified_parabolae=1,
            n_unidentified=0,
            n_carbon_identified=0,
        ),
        species_identified=[
            SpeciesIdentifiedEvalRow(
                parabola_index=0,
                label="H^1+",
                a=0.001,
                mq_meas=1.0,
                mq_target=1.0,
                delta_mq=0.0,
                rel_err=0.0,
            ),
        ],
        initial_to_bright_spot=InitialToBrightSpotEval(
            bright_spot_yx=(200, 150),
            initial_spot_yx=(50, 150),
            distance_px=150.0,
        ),
    )
    result = build_result_json(
        classified, spectra_result, geometry, timings, [], eval_payload
    )
    assert "eval" in result
    assert result["eval"]["species_summary"]["n_unidentified"] == 0
    assert "timings" in result
    t = result["timings"]
    assert "total" in t
    assert abs(t["total"] - sum(timings.values())) < 0.001
    for step in timings:
        assert step in t
        assert t[step] == round(timings[step], 4)
    assert result["algorithm_log"] == []
    assert result["geometry"]["common_vertex_x"] == 100.0
    assert result["geometry"]["common_vertex_y"] == 50.0
