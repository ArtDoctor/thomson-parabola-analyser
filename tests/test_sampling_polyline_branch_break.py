import numpy as np
import pytest

import oblisk.analysis.spectra_band_integration as spectra_band_module
from oblisk.analysis.spectra import (
    SamplePolyline,
    SpectrumGeometry,
    _longest_finite_polyline_segment,
    _sample_single_spectrum,
)


def test_sampling_polyline_inserts_nan_between_xp_branches() -> None:
    """Matplotlib would otherwise draw a chord from the last -Xp sample to the first +Xp."""
    image = np.full((256, 256), 50.0, dtype=float)
    geometry = SpectrumGeometry(
        x0_fit=128.0,
        y0_fit=128.0,
        theta_fit=0.0,
        gamma_fit=0.0,
        delta_fit=0.0,
    )
    _, _, polyline = _sample_single_spectrum(
        image=image,
        a_rot=0.002,
        mass_number=1,
        charge_state=1,
        geometry=geometry,
        background_mean=0.0,
        xp_bounds_px=(-40.0, 40.0),
        eps_px=3.0,
        sample_count=200,
        local_mean_radius_px=2,
        integration_window_a=None,
        spectrum_index=None,
        pixel_ownership=None,
    )
    assert polyline is not None
    assert np.any(np.isnan(polyline.x))
    assert np.any(np.isnan(polyline.y))


def test_sampling_polyline_single_xp_sign_has_no_nan_break() -> None:
    image = np.full((256, 256), 50.0, dtype=float)
    geometry = SpectrumGeometry(
        x0_fit=128.0,
        y0_fit=128.0,
        theta_fit=0.0,
        gamma_fit=0.0,
        delta_fit=0.0,
    )
    _, _, polyline = _sample_single_spectrum(
        image=image,
        a_rot=0.002,
        mass_number=1,
        charge_state=1,
        geometry=geometry,
        background_mean=0.0,
        xp_bounds_px=(10.0, 80.0),
        eps_px=3.0,
        sample_count=120,
        local_mean_radius_px=2,
        integration_window_a=None,
        spectrum_index=None,
        pixel_ownership=None,
    )
    assert polyline is not None
    assert not np.any(np.isnan(polyline.x))


def test_sampling_polyline_inserts_nan_after_out_of_bounds_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.full((256, 256), 50.0, dtype=float)
    geometry = SpectrumGeometry(
        x0_fit=128.0,
        y0_fit=128.0,
        theta_fit=0.0,
        gamma_fit=0.0,
        delta_fit=0.0,
        k1_fit=1.0,
        k2_fit=0.0,
        img_center_x=128.0,
        img_center_y=128.0,
        img_diag=float(np.hypot(256.0, 256.0)),
    )

    def gap_distort(
        x_u: np.ndarray,
        y_u: np.ndarray,
        cx: float,
        cy: float,
        k1: float,
        r_norm: float,
        n_iter: int = 5,
        k2: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        del cy, k1, r_norm, n_iter, k2
        x_out = np.asarray(x_u, dtype=float).copy()
        y_out = np.asarray(y_u, dtype=float).copy()
        gap_mask = (x_out > cx + 35.0) & (x_out < cx + 70.0)
        y_out[gap_mask] += 500.0
        return x_out, y_out

    monkeypatch.setattr(spectra_band_module, "distort_points", gap_distort)

    _, _, polyline = _sample_single_spectrum(
        image=image,
        a_rot=0.002,
        mass_number=1,
        charge_state=1,
        geometry=geometry,
        background_mean=0.0,
        xp_bounds_px=(10.0, 120.0),
        eps_px=3.0,
        sample_count=240,
        local_mean_radius_px=2,
        integration_window_a=None,
        spectrum_index=None,
        pixel_ownership=None,
    )

    assert polyline is not None
    assert np.any(np.isnan(polyline.x))
    assert np.any(np.isnan(polyline.y))


def test_longest_finite_polyline_segment_keeps_dominant_branch() -> None:
    polyline = SamplePolyline(
        x=np.array([10.0, 11.0, 12.0, np.nan, 30.0, 31.0], dtype=float),
        y=np.array([100.0, 101.0, 102.0, np.nan, 200.0, 201.0], dtype=float),
    )

    longest = _longest_finite_polyline_segment(polyline)

    assert longest is not None
    seg_x, seg_y = longest
    np.testing.assert_allclose(seg_x, np.array([10.0, 11.0, 12.0], dtype=float))
    np.testing.assert_allclose(seg_y, np.array([100.0, 101.0, 102.0], dtype=float))
