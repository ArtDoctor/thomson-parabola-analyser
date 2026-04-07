import numpy as np

from oblisk.analysis.geometry import distort_points
from oblisk.analysis.overlay import (
    ProjectionGeometry,
    project_parabola_curve,
    project_origin_point,
    project_polyline_segments,
)


def test_project_polyline_segments_serializes_points_as_xy_objects() -> None:
    segments = project_polyline_segments(
        np.array([10.0, 12.0, np.nan, 20.0, 22.0], dtype=float),
        np.array([30.0, 32.0, np.nan, 40.0, 42.0], dtype=float),
    )

    assert len(segments) == 2
    p0 = [point.model_dump(mode="json") for point in segments[0].to_payload()]
    assert p0 == [
        {"x": 10.0, "y": 30.0},
        {"x": 12.0, "y": 32.0},
    ]
    p1 = [point.model_dump(mode="json") for point in segments[1].to_payload()]
    assert p1 == [
        {"x": 20.0, "y": 40.0},
        {"x": 22.0, "y": 42.0},
    ]


def test_project_parabola_curve_63400_stays_single_visible_segment() -> None:
    geometry = ProjectionGeometry(
        x0_fit=210.3920363718853,
        y0_fit=234.49874736292256,
        theta_fit=-0.2562551424266636,
        gamma_fit=0.2,
        delta_fit=0.07499999999999998,
        k1_fit=0.1423599135079507,
        k2_fit=0.02252122889622504,
        img_center_x=565.0,
        img_center_y=557.0,
        img_diag=float(np.hypot(1130.0, 1114.0)),
    )

    curve = project_parabola_curve(
        a_value=0.0018630737586808547,
        geometry=geometry,
        xp_min=0.0,
        xp_max=948.0476719198526,
        image_shape=(1114, 1130),
        n_samples=10_000,
    )

    assert curve is not None
    assert len(curve.segments) == 1
    assert len(curve.segments[0].x) > 400


def test_project_origin_point_applies_same_distortion_as_curves() -> None:
    geometry = ProjectionGeometry(
        x0_fit=41.660218,
        y0_fit=249.911011,
        theta_fit=0.001364,
        gamma_fit=0.0,
        delta_fit=0.0,
        k1_fit=-0.001819,
        k2_fit=-0.000539,
        img_center_x=250.0,
        img_center_y=250.0,
        img_diag=float(np.hypot(500.0, 500.0)),
    )

    origin_x, origin_y = project_origin_point(geometry)
    expected_x, expected_y = distort_points(
        np.asarray([geometry.x0_fit], dtype=float),
        np.asarray([geometry.y0_fit], dtype=float),
        geometry.img_center_x,
        geometry.img_center_y,
        geometry.k1_fit,
        max(geometry.img_diag * 0.5, 1.0),
        k2=geometry.k2_fit,
    )

    np.testing.assert_allclose(origin_x, expected_x[0], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(origin_y, expected_y[0], rtol=0.0, atol=1e-12)
