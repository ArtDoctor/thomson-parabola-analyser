import numpy as np
from matplotlib import pyplot as plt

from oblisk.analysis.curvature_peaks import (
    compute_peak_half_width_bounds,
    detect_good_a_values,
    plot_a_score_and_peaks,
)
from oblisk.analysis.geometry import from_rotated_frame
from oblisk.analysis.spectra import (
    BackgroundRoi,
    SpectrumGeometry,
    build_spectra_result,
    plot_energy_spectra,
)


def _draw_band(
    image: np.ndarray,
    geometry: SpectrumGeometry,
    a_center: float,
    band_half_width: float,
    amplitude: float,
) -> None:
    xp_values = np.linspace(-80.0, 80.0, 320)
    a_values = np.linspace(a_center - band_half_width, a_center + band_half_width, 9)

    for a_value in a_values:
        yp_values = a_value * (xp_values**2)
        x_img, y_img = from_rotated_frame(
            xp_values,
            yp_values,
            geometry.x0_fit,
            geometry.y0_fit,
            geometry.theta_fit,
        )
        for x_val, y_val in zip(x_img, y_img):
            x_idx = int(round(x_val))
            y_idx = int(round(y_val))
            if x_idx < 1 or x_idx >= image.shape[1] - 1 or y_idx < 1 or y_idx >= image.shape[0] - 1:
                continue
            image[y_idx - 1:y_idx + 2, x_idx - 1:x_idx + 2] += amplitude


def test_smoke_half_width_spectra() -> None:
    geometry = SpectrumGeometry(
        x0_fit=110.0,
        y0_fit=24.0,
        theta_fit=0.0,
        gamma_fit=0.0,
        delta_fit=0.0,
    )

    a_grid = np.linspace(0.0015, 0.0058, 600)
    scores = 1.0 + 62.0 * np.exp(-0.5 * ((a_grid - 0.0024) / 0.00012) ** 2)
    scores += 45.0 * np.exp(-0.5 * ((a_grid - 0.0045) / 0.00018) ** 2)

    good_a, peak_indices = detect_good_a_values(
        a_grid,
        scores,
        prominence_rel=0.03,
        height_rel=0.01,
        min_distance=20,
    )
    peak_left_a, peak_right_a, half_height_scores = compute_peak_half_width_bounds(
        a_grid,
        scores,
        peak_indices,
    )
    integration_windows_a = np.column_stack((peak_left_a, peak_right_a))
    assert np.all(peak_left_a <= a_grid[peak_indices]), "Left bounds must stay left of their peaks."
    assert np.all(peak_right_a >= a_grid[peak_indices]), "Right bounds must stay right of their peaks."
    assert np.all(peak_right_a[:-1] <= peak_left_a[1:]), "Half-width windows must not overlap."

    close_scores = np.full_like(a_grid, 8.0)
    close_scores = close_scores + 50.0 * np.exp(
        -0.5 * ((a_grid - 0.00305) / 0.00010) ** 2
    )
    close_scores = close_scores + 47.0 * np.exp(
        -0.5 * ((a_grid - 0.00355) / 0.00010) ** 2
    )
    _, close_peak_indices = detect_good_a_values(
        a_grid,
        close_scores,
        prominence_rel=0.01,
        height_rel=0.01,
        min_distance=10,
    )
    assert close_peak_indices.size >= 2, "Expected at least two peaks in close-peak smoke case."
    close_peak_indices = close_peak_indices[:2]
    close_left_a, close_right_a, _ = compute_peak_half_width_bounds(
        a_grid,
        close_scores,
        close_peak_indices,
    )
    valley_idx = close_peak_indices[0] + int(
        np.argmin(close_scores[close_peak_indices[0]: close_peak_indices[1] + 1])
    )
    valley_a = a_grid[valley_idx]
    assert close_right_a[0] <= valley_a + 1e-12, "Left peak must stop at the inter-peak valley."
    assert close_left_a[1] >= valley_a - 1e-12, "Right peak must start at the inter-peak valley."

    weak_floor_scores = np.full_like(a_grid, 23.0)
    weak_floor_scores += 2.0 * np.exp(-0.5 * ((a_grid - 0.0022) / 0.00010) ** 2)
    weak_floor_scores += 2.5 * np.exp(-0.5 * ((a_grid - 0.0028) / 0.00010) ** 2)
    weak_floor_scores += 6.0 * np.exp(-0.5 * ((a_grid - 0.0044) / 0.00014) ** 2)
    _, weak_floor_peak_indices = detect_good_a_values(
        a_grid,
        weak_floor_scores,
        prominence_rel=0.01,
        height_rel=0.01,
        min_distance=12,
        floor_margin_abs=3.0,
    )
    assert weak_floor_peak_indices.size == 1, "Expected low peaks near the score floor to be filtered out."
    assert abs(float(a_grid[weak_floor_peak_indices[0]]) - 0.0044) < 1.5e-4, (
        "Expected the stronger peak to survive the score-floor filter."
    )

    plot_a_score_and_peaks(
        a_grid,
        scores,
        good_a,
        peak_indices,
        peak_left_a=peak_left_a,
        peak_right_a=peak_right_a,
        half_height_scores=half_height_scores,
        title="Synthetic score(a) smoke test",
    )
    plt.close("all")

    image = np.full((220, 220), 2.0, dtype=float)
    _draw_band(image, geometry, a_center=float(good_a[0]), band_half_width=0.00018, amplitude=18.0)
    _draw_band(image, geometry, a_center=float(good_a[1]), band_half_width=0.00025, amplitude=14.0)

    classified = [
        {
            "a": float(good_a[0]),
            "label": "H^1+",
            "candidates": [
                {"name": "H^1+", "mq_target": 1.0, "rel_err": 0.0},
            ],
        },
        {
            "a": float(good_a[1]),
            "label": "C^2+",
            "candidates": [
                {"name": "C^2+", "mq_target": 6.0, "rel_err": 0.0},
            ],
        },
    ]

    result = build_spectra_result(
        image=image,
        classified=classified,
        geometry=geometry,
        match_tol=0.08,
        background_roi=BackgroundRoi(x0=0, x1=40, y0=160, y1=210),
        xp_bounds_px=(-80.0, 80.0),
        sample_count=400,
        integration_windows_a=integration_windows_a,
        integration_a_samples=33,
        num_energy_bins=120,
        smoothing_sigma=1.5,
    )

    assert len(result.spectra) == 2, "Expected one spectrum per synthetic peak."
    assert result.energy_centers_keV.size == 120, "Expected common energy bins."
    assert all(spectrum.weights.size > 0 for spectrum in result.spectra), "Expected non-empty weights."
    assert all(
        np.any(spectrum.normalized_signal > 0.0) for spectrum in result.spectra
    ), "Expected non-zero integrated spectra."

    plot_energy_spectra(result, title="Synthetic half-width integrated spectra smoke test")
    plt.close("all")
