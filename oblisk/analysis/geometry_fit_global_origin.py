import numpy as np
from scipy.optimize import least_squares

from oblisk.analysis.geometry_fit_global_origin_rotation import solve_rotation_only
from oblisk.analysis.geometry_fit_origin_workspace import GlobalOriginFitWorkspace
from oblisk.analysis.geometry_fit_types import PathStep


def fit_global_origin_with_rotation(
    filtered_lines: list[list[list[int]]],
    x0_init: float,
    y0_init: float,
    theta_init: float = 0.0,
    gamma_init: float = 0.0,
    delta_init: float = 0.0,
    max_nfev: int = 20000,
    fit_rotation_only: bool = False,
    k1_init: float = 0.0,
    k2_init: float = 0.0,
    img_center: tuple[float, float] | None = None,
    img_diag: float | None = None,
) -> tuple[float, float, float, float, float, float, float, np.ndarray, list[PathStep]]:
    """
    Fit a shared origin (x0, y0), shared rotation theta, horizontal perspective γ,
    a linear tilt δ, and radial distortion k1, k2 so that in a warped frame

        Yp = a_i * basis + shear,
        where basis, shear = tilt_basis_and_shear(Xp, γ, δ),

    where (Xp, Yp) are coordinates in the rotated frame about (x0, y0),
    after optional radial undistortion of the raw pixel coordinates.

    Returns
    -------
    x0_fit, y0_fit, theta_fit, gamma_fit, delta_fit, k1_fit, k2_fit, a_list, path
    """
    ws = GlobalOriginFitWorkspace(
        filtered_lines,
        x0_init,
        y0_init,
        k1_init,
        k2_init,
        img_center,
        img_diag,
    )

    if fit_rotation_only:
        return solve_rotation_only(
            ws,
            x0_init,
            y0_init,
            theta_init,
            gamma_init,
            delta_init,
            k1_init,
            k2_init,
            max_nfev,
        )

    if ws.fit_k1:
        p0_stage1 = np.array(
            [x0_init, y0_init, theta_init, k1_init, k2_init],
            dtype=float,
        )
        lower_stage1 = np.array([-np.inf, -np.inf, -np.pi, -2.0, -2.0], dtype=float)
        upper_stage1 = np.array([np.inf, np.inf, np.pi, 2.0, 2.0], dtype=float)
    else:
        p0_stage1 = np.array([x0_init, y0_init, theta_init], dtype=float)
        lower_stage1 = np.array([-np.inf, -np.inf, -np.pi], dtype=float)
        upper_stage1 = np.array([np.inf, np.inf, np.pi], dtype=float)

    path_stage1: list[PathStep] = [
        ws.path_step(
            float(p0_stage1[0]),
            float(p0_stage1[1]),
            float(p0_stage1[2]),
            0.0,
            0.0,
            float(p0_stage1[3]) if ws.fit_k1 else 0.0,
            float(p0_stage1[4]) if ws.fit_k1 else 0.0,
        )
    ]

    def cb_stage1(xk: np.ndarray, *_unused: object) -> None:
        path_stage1.append(
            ws.path_step(
                float(xk[0]),
                float(xk[1]),
                float(xk[2]),
                0.0,
                0.0,
                float(xk[3]) if ws.fit_k1 else 0.0,
                float(xk[4]) if ws.fit_k1 else 0.0,
            )
        )

    res_stage1 = least_squares(
        ws.residuals_stage1,
        p0_stage1,
        bounds=(lower_stage1, upper_stage1),
        max_nfev=max_nfev,
        verbose=0,
        loss="huber",
        f_scale=3.0,
        callback=cb_stage1,
    )

    x0_stage1 = float(res_stage1.x[0])
    y0_stage1 = float(res_stage1.x[1])
    theta_stage1 = float(res_stage1.x[2])
    k1_stage1 = float(res_stage1.x[3]) if ws.fit_k1 else 0.0
    k2_stage1 = float(res_stage1.x[4]) if ws.fit_k1 else 0.0

    profile_max_nfev = 200
    path_profile: list[PathStep] = []
    bounds_profile = (
        np.array([-np.inf, -np.inf, -np.pi], dtype=float),
        np.array([np.inf, np.inf, np.pi], dtype=float),
    )

    def _run_profile_grid(
        gamma_values: np.ndarray,
        delta_values: np.ndarray,
        k1_fixed: float,
        k2_fixed: float,
        x0_ref: float,
        y0_ref: float,
        p0_geo: np.ndarray,
        best_so_far: dict[str, float],
        best_cost_so_far: float,
    ) -> tuple[dict[str, float], float]:
        best = dict(best_so_far)
        best_cost = best_cost_so_far
        for gv in gamma_values:
            for dv in delta_values:
                gf = float(gv)
                df = float(dv)

                def _res_profile(
                    p: np.ndarray,
                    _g: float = gf,
                    _d: float = df,
                ) -> np.ndarray:
                    return ws.fixed_tilt_profile_residuals(
                        p,
                        gamma_fixed=_g,
                        delta_fixed=_d,
                        k1_fixed=k1_fixed,
                        k2_fixed=k2_fixed,
                        x0_ref=x0_ref,
                        y0_ref=y0_ref,
                    )

                init_res = _res_profile(p0_geo)
                init_cost = float(0.5 * np.sum(init_res * init_res))
                if best_cost > 0 and init_cost > 3.0 * best_cost:
                    continue

                res_p = least_squares(
                    _res_profile,
                    p0_geo,
                    bounds=bounds_profile,
                    max_nfev=profile_max_nfev,
                    verbose=0,
                    loss="huber",
                    f_scale=3.0,
                )

                cost_p = float(res_p.cost)
                path_profile.append(
                    ws.path_step(
                        float(res_p.x[0]),
                        float(res_p.x[1]),
                        float(res_p.x[2]),
                        gf,
                        df,
                        k1_fixed,
                        k2_fixed,
                    )
                )
                if cost_p < best_cost:
                    best_cost = cost_p
                    best = {
                        "x0": float(res_p.x[0]),
                        "y0": float(res_p.x[1]),
                        "theta": float(res_p.x[2]),
                        "gamma": gf,
                        "delta": df,
                        "k1": k1_fixed,
                        "k2": k2_fixed,
                    }
        return best, best_cost

    coarse_gamma = np.array([-0.25, -0.10, 0.0, 0.10, 0.25], dtype=float)
    coarse_delta = np.array([-0.70, -0.45, -0.20, 0.0, 0.20, 0.45, 0.70], dtype=float)

    best_profile: dict[str, float] = {
        "x0": x0_stage1,
        "y0": y0_stage1,
        "theta": theta_stage1,
        "gamma": 0.0,
        "delta": 0.0,
        "k1": k1_stage1,
        "k2": k2_stage1,
    }
    p0_profile = np.array([x0_stage1, y0_stage1, theta_stage1], dtype=float)
    best_profile_cost = float(
        0.5
        * np.sum(
            ws.fixed_tilt_profile_residuals(
                p0_profile,
                gamma_fixed=0.0,
                delta_fixed=0.0,
                k1_fixed=k1_stage1,
                k2_fixed=k2_stage1,
                x0_ref=x0_stage1,
                y0_ref=y0_stage1,
            )
            ** 2
        )
    )
    path_profile.append(
        ws.path_step(
            x0_stage1,
            y0_stage1,
            theta_stage1,
            0.0,
            0.0,
            k1_stage1,
            k2_stage1,
        )
    )

    best_profile, best_profile_cost = _run_profile_grid(
        coarse_gamma,
        coarse_delta,
        k1_stage1,
        k2_stage1,
        x0_stage1,
        y0_stage1,
        p0_profile,
        best_profile,
        best_profile_cost,
    )

    fine_gamma = np.linspace(
        max(best_profile["gamma"] - 0.10, -0.85),
        min(best_profile["gamma"] + 0.10, 0.85),
        3,
    )
    fine_delta = np.linspace(
        max(best_profile["delta"] - 0.15, -0.85),
        min(best_profile["delta"] + 0.15, 0.85),
        5,
    )
    p0_fine = np.array(
        [best_profile["x0"], best_profile["y0"], best_profile["theta"]],
        dtype=float,
    )
    best_profile, best_profile_cost = _run_profile_grid(
        fine_gamma,
        fine_delta,
        k1_stage1,
        k2_stage1,
        x0_stage1,
        y0_stage1,
        p0_fine,
        best_profile,
        best_profile_cost,
    )

    path_refine: list[PathStep] = []

    def _run_geo_fit(
        seed: dict[str, float],
        nfev: int,
    ) -> tuple[dict[str, float], list[PathStep]]:
        ga_fixed = seed["gamma"]
        de_fixed = seed["delta"]

        def _residuals_geo(p: np.ndarray) -> np.ndarray:
            x0_f = float(p[0])
            y0_f = float(p[1])
            th_f = float(p[2])
            k1_f = float(p[3]) if ws.fit_k1 else seed["k1"]
            k2_f = float(p[4]) if ws.fit_k1 else seed["k2"]

            _, d_res = ws.compute_a_and_data_residuals(
                x0_f,
                y0_f,
                th_f,
                ga_fixed,
                de_fixed,
                k1_f,
                k2_f,
            )

            x_sh = (x0_f - seed["x0"]) / ws.r_x
            y_sh = (y0_f - seed["y0"]) / ws.r_y
            o_res = ws.prior_scale * np.array([x_sh, y_sh], dtype=float)

            parts: list[np.ndarray] = [o_res]
            if ws.fit_k1:
                parts.append(ws.prior_scale * np.array([(k1_f - seed["k1"]) / ws.r_k1]))
                parts.append(ws.prior_scale * np.array([(k2_f - seed["k2"]) / ws.r_k2]))

            return np.concatenate([d_res * ws.weights_per_point] + parts)

        if ws.fit_k1:
            p0_g = np.array(
                [seed["x0"], seed["y0"], seed["theta"], seed["k1"], seed["k2"]],
                dtype=float,
            )
            lo = np.array([-np.inf, -np.inf, -np.pi, -2.0, -2.0], dtype=float)
            hi = np.array([np.inf, np.inf, np.pi, 2.0, 2.0], dtype=float)
        else:
            p0_g = np.array([seed["x0"], seed["y0"], seed["theta"]], dtype=float)
            lo = np.array([-np.inf, -np.inf, -np.pi], dtype=float)
            hi = np.array([np.inf, np.inf, np.pi], dtype=float)

        local_path: list[PathStep] = [
            ws.path_step(
                p0_g[0],
                p0_g[1],
                p0_g[2],
                ga_fixed,
                de_fixed,
                p0_g[3] if ws.fit_k1 else seed["k1"],
                p0_g[4] if ws.fit_k1 else seed["k2"],
            )
        ]

        def _cb_geo(xk: np.ndarray, *_unused: object) -> None:
            local_path.append(
                ws.path_step(
                    float(xk[0]),
                    float(xk[1]),
                    float(xk[2]),
                    ga_fixed,
                    de_fixed,
                    float(xk[3]) if ws.fit_k1 else seed["k1"],
                    float(xk[4]) if ws.fit_k1 else seed["k2"],
                )
            )

        res_g = least_squares(
            _residuals_geo,
            p0_g,
            bounds=(lo, hi),
            max_nfev=nfev,
            verbose=0,
            loss="huber",
            f_scale=3.0,
            callback=_cb_geo,
        )

        result = {
            "x0": float(res_g.x[0]),
            "y0": float(res_g.x[1]),
            "theta": float(res_g.x[2]),
            "gamma": ga_fixed,
            "delta": de_fixed,
            "k1": float(res_g.x[3]) if ws.fit_k1 else seed["k1"],
            "k2": float(res_g.x[4]) if ws.fit_k1 else seed["k2"],
        }
        return result, local_path

    current_best = dict(best_profile)
    n_coord_iters = 2

    def _data_rmse(d: dict[str, float]) -> float:
        return ws.rmse_for_params(
            d["x0"],
            d["y0"],
            d["theta"],
            d["gamma"],
            d["delta"],
            d["k1"],
            d["k2"],
        )

    for _cd_iter in range(n_coord_iters):
        geo_result, geo_path = _run_geo_fit(current_best, max_nfev)
        path_refine.extend(geo_path)

        if _data_rmse(geo_result) < _data_rmse(current_best):
            current_best = geo_result

        re_gamma = np.linspace(
            max(current_best["gamma"] - 0.08, -0.85),
            min(current_best["gamma"] + 0.08, 0.85),
            3,
        )
        re_delta = np.linspace(
            max(current_best["delta"] - 0.12, -0.85),
            min(current_best["delta"] + 0.12, 0.85),
            5,
        )
        p0_re = np.array(
            [current_best["x0"], current_best["y0"], current_best["theta"]],
            dtype=float,
        )
        reprofile_best, _ = _run_profile_grid(
            re_gamma,
            re_delta,
            current_best["k1"],
            current_best["k2"],
            current_best["x0"],
            current_best["y0"],
            p0_re,
            current_best,
            float("inf"),
        )
        if _data_rmse(reprofile_best) < _data_rmse(current_best):
            current_best = reprofile_best

    final_geo, final_path = _run_geo_fit(current_best, max_nfev)
    path_refine.extend(final_path)
    if _data_rmse(final_geo) < _data_rmse(current_best):
        current_best = final_geo

    x0_fit = current_best["x0"]
    y0_fit = current_best["y0"]
    theta_fit = current_best["theta"]
    gamma_fit = current_best["gamma"]
    delta_fit = current_best["delta"]
    k1_fit = current_best["k1"]
    k2_fit = current_best["k2"]
    path = path_stage1 + path_profile + path_refine

    a_list, _ = ws.compute_a_and_data_residuals(
        x0_fit,
        y0_fit,
        theta_fit,
        gamma_fit,
        delta_fit,
        k1_fit,
        k2_fit,
    )

    return (
        x0_fit,
        y0_fit,
        theta_fit,
        gamma_fit,
        delta_fit,
        k1_fit,
        k2_fit,
        a_list,
        path,
    )
