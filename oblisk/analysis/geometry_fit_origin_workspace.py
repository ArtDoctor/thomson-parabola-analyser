import numpy as np

from oblisk.analysis.geometry import (
    perspective_reference_from_xp,
    tilt_basis_and_shear,
    to_rotated_frame,
    undistort_points,
)
from oblisk.analysis.geometry_fit_types import PathStep


class GlobalOriginFitWorkspace:
    def __init__(
        self,
        filtered_lines: list[list[list[int]]],
        x0_init: float,
        y0_init: float,
        k1_init: float,
        k2_init: float,
        img_center: tuple[float, float] | None,
        img_diag: float | None,
    ) -> None:
        x_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []
        l_list: list[np.ndarray] = []
        for i, line in enumerate(filtered_lines):
            arr = np.asarray(line, dtype=float)
            if arr.ndim != 2 or arr.shape[1] < 3:
                raise ValueError(
                    f"Each line must be an array of shape (N, >=3). "
                    f"Got shape {arr.shape} for line index {i}"
                )
            x_list.append(arr[:, 1])
            y_list.append(arr[:, 2])
            l_list.append(np.full(arr.shape[0], i, dtype=int))

        self.X = np.concatenate(x_list)
        self.Y = np.concatenate(y_list)
        self.L = np.concatenate(l_list)
        self.num_lines = len(filtered_lines)
        self.n_points = int(self.X.size)
        if self.n_points == 0:
            raise ValueError("No points provided in filtered_lines")

        if img_center is not None and img_diag is not None:
            self.cx, self.cy = img_center
            self.r_norm = max(img_diag * 0.5, 1.0)
            self.fit_k1 = True
        else:
            self.cx, self.cy, self.r_norm = 0.0, 0.0, 1.0
            self.fit_k1 = False

        line_counts = np.bincount(self.L, minlength=self.num_lines).astype(float)
        line_counts[line_counts == 0.0] = 1.0
        self.weights_per_point = 1.0 / np.sqrt(line_counts[self.L])

        span_x = float(self.X.max() - self.X.min())
        span_y = float(self.Y.max() - self.Y.min())
        origin_span_frac = 0.2
        self.r_x = max(origin_span_frac * span_x, 1e-6)
        self.r_y = max(origin_span_frac * span_y, 1e-6)
        self.prior_scale = float(np.sqrt(float(self.n_points)))
        self.r_k1 = 0.5
        self.r_k2 = 0.5
        self.x0_init = x0_init
        self.y0_init = y0_init
        self.k1_init = k1_init
        self.k2_init = k2_init

    def compute_a_and_data_residuals(
        self,
        x0_fit: float,
        y0_fit: float,
        theta_fit: float,
        gamma_fit: float,
        delta_fit: float,
        k1_fit: float = 0.0,
        k2_fit: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.fit_k1 and (abs(k1_fit) > 1e-15 or abs(k2_fit) > 1e-15):
            x_u, y_u = undistort_points(
                self.X,
                self.Y,
                self.cx,
                self.cy,
                k1_fit,
                self.r_norm,
                k2=k2_fit,
            )
        else:
            x_u, y_u = self.X, self.Y

        xp, yp = to_rotated_frame(x_u, y_u, x0_fit, y0_fit, theta_fit)
        perspective_reference = perspective_reference_from_xp(xp)
        basis, shear = tilt_basis_and_shear(
            xp,
            gamma_fit,
            delta_fit,
            perspective_reference=perspective_reference,
        )
        yp_shifted = yp - shear

        eps = 1e-12
        numer = np.bincount(self.L, weights=basis * yp_shifted, minlength=self.num_lines)
        denom = np.bincount(self.L, weights=basis * basis, minlength=self.num_lines) + eps
        a_arr = numer / denom

        y_pred = a_arr[self.L] * basis + shear
        data_res = y_pred - yp
        return a_arr, data_res

    def rmse_for_params(
        self,
        x0_fit: float,
        y0_fit: float,
        theta_fit: float,
        gamma_fit: float,
        delta_fit: float,
        k1_fit: float = 0.0,
        k2_fit: float = 0.0,
    ) -> float:
        _, data_res = self.compute_a_and_data_residuals(
            x0_fit,
            y0_fit,
            theta_fit,
            gamma_fit,
            delta_fit,
            k1_fit,
            k2_fit,
        )
        data_res_weighted = data_res * self.weights_per_point
        return float(np.sqrt(np.mean(data_res_weighted * data_res_weighted)))

    def path_step(
        self,
        x0_fit: float,
        y0_fit: float,
        theta_fit: float,
        gamma_fit: float,
        delta_fit: float,
        k1_fit: float = 0.0,
        k2_fit: float = 0.0,
    ) -> PathStep:
        return {
            "x0": float(x0_fit),
            "y0": float(y0_fit),
            "theta": float(theta_fit),
            "gamma": float(gamma_fit),
            "delta": float(delta_fit),
            "k1": float(k1_fit),
            "k2": float(k2_fit),
            "rmse": self.rmse_for_params(
                x0_fit,
                y0_fit,
                theta_fit,
                gamma_fit,
                delta_fit,
                k1_fit,
                k2_fit,
            ),
        }

    def residuals_stage1(self, p: np.ndarray) -> np.ndarray:
        x0_fit = float(p[0])
        y0_fit = float(p[1])
        theta_fit = float(p[2])
        k1_fit = float(p[3]) if self.fit_k1 else 0.0
        k2_fit = float(p[4]) if self.fit_k1 else 0.0

        _, data_res = self.compute_a_and_data_residuals(
            x0_fit,
            y0_fit,
            theta_fit,
            0.0,
            0.0,
            k1_fit,
            k2_fit,
        )

        x_shift = (x0_fit - self.x0_init) / self.r_x
        y_shift = (y0_fit - self.y0_init) / self.r_y
        origin_res = self.prior_scale * np.array([x_shift, y_shift], dtype=float)

        prior_parts: list[np.ndarray] = [origin_res]
        if self.fit_k1:
            k1_shift = (k1_fit - self.k1_init) / self.r_k1
            k1_res = self.prior_scale * np.array([k1_shift], dtype=float)
            prior_parts.append(k1_res)
            k2_shift = (k2_fit - self.k2_init) / self.r_k2
            k2_res = self.prior_scale * np.array([k2_shift], dtype=float)
            prior_parts.append(k2_res)

        data_res_weighted = data_res * self.weights_per_point
        return np.concatenate([data_res_weighted] + prior_parts)

    def fixed_tilt_profile_residuals(
        self,
        p: np.ndarray,
        gamma_fixed: float,
        delta_fixed: float,
        k1_fixed: float,
        k2_fixed: float,
        x0_ref: float,
        y0_ref: float,
    ) -> np.ndarray:
        x0_fit = float(p[0])
        y0_fit = float(p[1])
        theta_fit = float(p[2])
        _, data_res = self.compute_a_and_data_residuals(
            x0_fit,
            y0_fit,
            theta_fit,
            gamma_fixed,
            delta_fixed,
            k1_fixed,
            k2_fixed,
        )

        x_shift = (x0_fit - x0_ref) / self.r_x
        y_shift = (y0_fit - y0_ref) / self.r_y
        origin_res = self.prior_scale * np.array([x_shift, y_shift], dtype=float)
        data_res_weighted = data_res * self.weights_per_point
        return np.concatenate([data_res_weighted, origin_res])
