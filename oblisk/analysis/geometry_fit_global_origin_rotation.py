import numpy as np
from scipy.optimize import least_squares

from oblisk.analysis.geometry_fit_origin_workspace import GlobalOriginFitWorkspace
from oblisk.analysis.geometry_fit_types import PathStep


def solve_rotation_only(
    ws: GlobalOriginFitWorkspace,
    x0_init: float,
    y0_init: float,
    theta_init: float,
    gamma_init: float,
    delta_init: float,
    k1_init: float,
    k2_init: float,
    max_nfev: int,
) -> tuple[float, float, float, float, float, float, np.ndarray, list[PathStep]]:
    def residuals_rot_only(p: np.ndarray) -> np.ndarray:
        theta_fit_local = float(p[0])
        gamma_fit_local = float(p[1])
        delta_fit_local = float(p[2])
        k1_fit_local = float(p[3]) if ws.fit_k1 else 0.0
        k2_fit_local = float(p[4]) if ws.fit_k1 else 0.0
        _, data_res = ws.compute_a_and_data_residuals(
            x0_init,
            y0_init,
            theta_fit_local,
            gamma_fit_local,
            delta_fit_local,
            k1_fit_local,
            k2_fit_local,
        )
        return data_res * ws.weights_per_point

    if ws.fit_k1:
        p0 = np.array(
            [theta_init, gamma_init, delta_init, k1_init, k2_init],
            dtype=float,
        )
        lower = np.array([-np.pi, -0.9, -0.9, -2.0, -2.0], dtype=float)
        upper = np.array([np.pi, 0.9, 0.9, 2.0, 2.0], dtype=float)
    else:
        p0 = np.array([theta_init, gamma_init, delta_init], dtype=float)
        lower = np.array([-np.pi, -0.9, -0.9], dtype=float)
        upper = np.array([np.pi, 0.9, 0.9], dtype=float)

    path: list[PathStep] = [
        ws.path_step(
            x0_init,
            y0_init,
            float(p0[0]),
            float(p0[1]),
            float(p0[2]),
            float(p0[3]) if ws.fit_k1 else 0.0,
            float(p0[4]) if ws.fit_k1 else 0.0,
        )
    ]

    def cb_rot_only(xk: np.ndarray, *_unused: object) -> None:
        path.append(
            ws.path_step(
                x0_init,
                y0_init,
                float(xk[0]),
                float(xk[1]),
                float(xk[2]),
                float(xk[3]) if ws.fit_k1 else 0.0,
                float(xk[4]) if ws.fit_k1 else 0.0,
            )
        )

    res = least_squares(
        residuals_rot_only,
        p0,
        bounds=(lower, upper),
        max_nfev=max_nfev,
        verbose=0,
        loss="huber",
        f_scale=3.0,
        callback=cb_rot_only,
    )

    theta_fit = float(res.x[0])
    gamma_fit = float(res.x[1])
    delta_fit = float(res.x[2])
    k1_fit = float(res.x[3]) if ws.fit_k1 else 0.0
    k2_fit = float(res.x[4]) if ws.fit_k1 else 0.0

    a_list, _ = ws.compute_a_and_data_residuals(
        x0_init,
        y0_init,
        theta_fit,
        gamma_fit,
        delta_fit,
        k1_fit,
        k2_fit,
    )
    return (
        x0_init,
        y0_init,
        theta_fit,
        gamma_fit,
        delta_fit,
        k1_fit,
        k2_fit,
        a_list,
        path,
    )
