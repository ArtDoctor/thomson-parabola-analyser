import numpy as np

from oblisk.analysis.geometry import (
    dominant_xp_sign_from_points,
    distort_points,
    longest_finite_polyline_segment,
    perspective_reference_from_xp,
    tilt_inverse_Yp,
    undistort_points,
    visible_polyline_with_nan_breaks,
)


def test_dominant_xp_sign_from_points_prefers_positive_support() -> None:
    points_list = [
        np.array(
            [
                [0.0, 10.0],
                [4.0, 12.0],
                [12.0, 16.0],
                [20.0, 24.0],
            ],
            dtype=float,
        )
    ]

    sign = dominant_xp_sign_from_points(
        points_list=points_list,
        x0_fit=0.0,
        y0_fit=0.0,
        theta_fit=0.0,
    )

    assert sign == 1


def test_dominant_xp_sign_from_points_prefers_negative_support() -> None:
    points_list = [
        np.array(
            [
                [-25.0, 11.0],
                [-18.0, 15.0],
                [-10.0, 18.0],
                [1.0, 20.0],
            ],
            dtype=float,
        )
    ]

    sign = dominant_xp_sign_from_points(
        points_list=points_list,
        x0_fit=0.0,
        y0_fit=0.0,
        theta_fit=0.0,
    )

    assert sign == -1


def test_visible_polyline_with_nan_breaks_splits_reentry_runs() -> None:
    visible = visible_polyline_with_nan_breaks(
        x_arr=np.array([10.0, 12.0, 300.0, 14.0, 16.0], dtype=float),
        y_arr=np.array([10.0, 12.0, 300.0, 14.0, 16.0], dtype=float),
        width=100,
        height=100,
    )

    assert visible is not None
    x_vis, y_vis = visible
    np.testing.assert_allclose(
        x_vis,
        np.array([10.0, 12.0, np.nan, 14.0, 16.0], dtype=float),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        y_vis,
        np.array([10.0, 12.0, np.nan, 14.0, 16.0], dtype=float),
        equal_nan=True,
    )


def test_longest_finite_polyline_segment_keeps_main_run() -> None:
    longest = longest_finite_polyline_segment(
        x_arr=np.array([10.0, 12.0, 14.0, np.nan, 50.0, 52.0], dtype=float),
        y_arr=np.array([100.0, 110.0, 120.0, np.nan, 10.0, 20.0], dtype=float),
    )

    assert longest is not None
    x_seg, y_seg = longest
    np.testing.assert_allclose(x_seg, np.array([10.0, 12.0, 14.0], dtype=float))
    np.testing.assert_allclose(y_seg, np.array([100.0, 110.0, 120.0], dtype=float))


def test_distort_points_round_trips_large_radius_inputs() -> None:
    cx = 565.0
    cy = 557.0
    radius_norm = float(np.hypot(1130.0, 1114.0) * 0.5)
    k1 = 0.1423599135079507
    k2 = 0.02252122889622504

    x_u = np.array([1660.4872992610565], dtype=float)
    y_u = np.array([2028.4563738759127], dtype=float)
    x_d, y_d = distort_points(x_u, y_u, cx, cy, k1, radius_norm, k2=k2)

    assert np.all(np.isfinite(x_d))
    assert np.all(np.isfinite(y_d))

    x_roundtrip, y_roundtrip = undistort_points(
        x_d,
        y_d,
        cx,
        cy,
        k1,
        radius_norm,
        k2=k2,
    )
    np.testing.assert_allclose(x_roundtrip, x_u, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(y_roundtrip, y_u, rtol=0.0, atol=1e-6)


def test_tilt_inverse_yp_stays_consistent_across_subsets_with_shared_reference() -> None:
    xp_full = np.linspace(-40.0, 140.0, 181, dtype=float)
    xp_sub = xp_full[25:140:3]
    reference = perspective_reference_from_xp(xp_full)

    yp_full = tilt_inverse_Yp(
        xp_full,
        a=0.0012,
        gamma=-0.2,
        delta=0.15,
        perspective_reference=reference,
    )
    yp_sub = tilt_inverse_Yp(
        xp_sub,
        a=0.0012,
        gamma=-0.2,
        delta=0.15,
        perspective_reference=reference,
    )

    np.testing.assert_allclose(yp_sub, yp_full[25:140:3], rtol=0.0, atol=1e-12)
