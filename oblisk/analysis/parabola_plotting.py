from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from oblisk.analysis.geometry import (
    PerspectiveReference,
    distort_points,
    from_rotated_frame,
    perspective_reference_from_xp,
    tilt_inverse_Yp,
    to_rotated_frame,
    undistort_points,
)
from oblisk.analysis.geometry_fit_types import PathStep
from oblisk.analysis.overlay import (
    ProjectionGeometry,
    build_detected_projected_curves,
    project_origin_point,
)
from oblisk.plot_display import show_or_save


def plot_lines_rotated(
    filtered_lines: list[list[list[int]]],
    x0: float,
    y0: float,
    title: str,
    image: np.ndarray | None = None,
    show_error_curve: bool = True,
    x0_fit: float | None = None,
    y0_fit: float | None = None,
    theta_fit: float | None = None,
    gamma_fit: float = 0.0,
    delta_fit: float = 0.0,
    a_list: np.ndarray | None = None,
    path: list[PathStep] | None = None,
    save_paths: list[Path | None] | None = None,
    output_indices: list[int] | None = None,
    k1_fit: float = 0.0,
    k2_fit: float = 0.0,
    img_center: tuple[float, float] | None = None,
    img_diag: float | None = None,
    perspective_reference: PerspectiveReference | None = None,
) -> None:
    """
    Visualize experimental lines and their fitted parabolas in the original
    image coordinates, using shared origin, rotation and perspective/tilt
    parameters.
    """
    if x0_fit is None or y0_fit is None or theta_fit is None:
        raise ValueError("x0_fit, y0_fit and theta_fit must be provided")

    def _plot_single_figure(
        with_background: bool,
        fig_save_path: Path | None = None,
    ) -> None:
        plt.figure(figsize=(12, 8))
        if with_background and image is not None:
            plt.imshow(image, cmap="gray", origin="lower")

        line_handles = []
        line_labels = []

        for idx, line in enumerate(filtered_lines):
            line_arr = np.asarray(line, dtype=float)
            if line_arr.ndim != 2 or line_arr.shape[1] < 3:
                continue

            x_vals = line_arr[:, 1]
            y_vals = line_arr[:, 2]

            (handle,) = plt.plot(x_vals, y_vals, linewidth=2)

            x_vals_fit = x_vals
            y_vals_fit = y_vals
            if (
                (abs(k1_fit) > 1e-15 or abs(k2_fit) > 1e-15)
                and img_center is not None
                and img_diag is not None
            ):
                x_vals_fit, y_vals_fit = undistort_points(
                    x_vals,
                    y_vals,
                    img_center[0],
                    img_center[1],
                    k1_fit,
                    max(img_diag * 0.5, 1.0),
                    k2=k2_fit,
                )

            Xp_data, Yp_data = to_rotated_frame(
                x_vals_fit,
                y_vals_fit,
                x0_fit,
                y0_fit,
                theta_fit,
            )
            assert Xp_data.shape == x_vals.shape == y_vals.shape

            Xp_min, Xp_max = float(Xp_data.min()), float(Xp_data.max())
            if Xp_min == Xp_max:
                Xp_fit = np.array([Xp_min - 1.0, Xp_max + 1.0])
            else:
                Xp_fit = np.linspace(Xp_min, Xp_max, 400)

            if a_list is not None and idx < len(a_list):
                a_i = float(a_list[idx])
            else:
                a_i = 0.0

            line_reference = (
                perspective_reference
                if perspective_reference is not None
                else perspective_reference_from_xp(Xp_data)
            )
            Yp_fit = tilt_inverse_Yp(
                Xp_fit,
                a_i,
                gamma_fit,
                delta_fit,
                perspective_reference=line_reference,
            )

            xf, yf = from_rotated_frame(Xp_fit, Yp_fit, x0_fit, y0_fit, theta_fit)
            if (
                (abs(k1_fit) > 1e-15 or abs(k2_fit) > 1e-15)
                and img_center is not None
                and img_diag is not None
            ):
                xf, yf = distort_points(
                    xf,
                    yf,
                    img_center[0],
                    img_center[1],
                    k1_fit,
                    max(img_diag * 0.5, 1.0),
                    k2=k2_fit,
                )

            plt.plot(
                xf,
                yf,
                linestyle="--",
                linewidth=0.9,
                color=handle.get_color(),
            )

            mid_idx = len(line_arr) // 2
            mid_x, mid_y = x_vals[mid_idx], y_vals[mid_idx]
            text_color = "white" if image is not None and with_background else "black"
            box_face = "black" if text_color == "white" else "white"

            plt.text(
                mid_x,
                mid_y,
                f"{idx}",
                fontsize=7,
                color=text_color,
                ha="center",
                va="center",
                bbox=dict(facecolor=box_face, alpha=0.7, boxstyle="round,pad=0.2"),
            )

            line_handles.append(handle)
            line_labels.append(f"Line {idx}   a={a_i:.3g}")

        (init_origin_handle,) = plt.plot(x0, y0, "ro", markersize=5)
        fitted_origin_handle = plt.plot(
            x0_fit,
            y0_fit,
            marker="x",
            markersize=6,
            linewidth=0,
            color="cyan",
        )[0]

        path_handle = None
        if path is not None and len(path) > 1:
            xs_path = [step["x0"] for step in path]
            ys_path = [step["y0"] for step in path]
            path_handle = plt.plot(
                xs_path,
                ys_path,
                "o-",
                color="yellow",
                linewidth=2,
                markersize=4,
                alpha=1.0,
            )[0]

        legend_handles = list(line_handles)
        legend_labels = list(line_labels)
        legend_handles.append(init_origin_handle)
        legend_labels.append("Initial origin (guess)")
        legend_handles.append(fitted_origin_handle)
        legend_labels.append("Fitted shared origin")

        if path_handle is not None:
            legend_handles.append(path_handle)
            legend_labels.append("Optimization path (x0, y0)")

        plt.title(
            f"{title}\nθ = {theta_fit:.4f} rad, γ = {gamma_fit:.3f}, δ = {delta_fit:.3f}"
        )
        plt.xlabel("x (pixels)")
        plt.ylabel("y (pixels)")
        plt.legend(legend_handles, legend_labels, loc="upper left", fontsize=8)
        plt.tight_layout()
        show_or_save(fig_save_path)

    if output_indices is None:
        output_indices = [0, 1, 2]

    if 0 in output_indices:
        _plot_single_figure(
            with_background=True,
            fig_save_path=save_paths[0] if save_paths and len(save_paths) > 0 else None,
        )
    if 1 in output_indices:
        _plot_single_figure(
            with_background=False,
            fig_save_path=save_paths[1] if save_paths and len(save_paths) > 1 else None,
        )

    if 2 in output_indices and show_error_curve and path is not None and len(path) > 0:
        rmses = [step["rmse"] for step in path]
        plt.figure(figsize=(6, 4))
        plt.plot(range(len(rmses)), rmses, marker="o")
        plt.yscale("log")
        plt.xlabel("Iteration")
        plt.ylabel("RMSE (warped frame, weighted)")
        plt.title(f"Error vs Iteration - {title}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        show_or_save(save_paths[2] if save_paths and len(save_paths) > 2 else None)


def plot_detected_parabolas(
    image: np.ndarray,
    good_a: np.ndarray,
    x0_fit: float,
    y0_fit: float,
    theta_fit: float,
    gamma_fit: float,
    delta_fit: float,
    Xp_min: float,
    Xp_max: float,
    n_samples_per_parabola: int = 800,
    title: str = "Detected parabolas from intensity scan",
    save_path: Path | None = None,
    k1_fit: float = 0.0,
    k2_fit: float = 0.0,
    img_center: tuple[float, float] | None = None,
    img_diag: float | None = None,
    perspective_reference: PerspectiveReference | None = None,
) -> None:
    curve_linewidth = 0.7
    curve_alpha = 0.88
    plt.figure(figsize=(12, 8))
    plt.imshow(image, cmap="gray", origin="lower")

    H, W = image.shape[:2]
    projection_geometry = ProjectionGeometry(
        x0_fit=float(x0_fit),
        y0_fit=float(y0_fit),
        theta_fit=float(theta_fit),
        gamma_fit=float(gamma_fit),
        delta_fit=float(delta_fit),
        k1_fit=float(k1_fit),
        k2_fit=float(k2_fit),
        img_center_x=float(img_center[0]) if img_center is not None else 0.0,
        img_center_y=float(img_center[1]) if img_center is not None else 0.0,
        img_diag=float(img_diag) if img_diag is not None else 1.0,
        perspective_reference=(
            perspective_reference
            if perspective_reference is not None
            else PerspectiveReference()
        ),
    )
    origin_x, origin_y = project_origin_point(projection_geometry)

    if good_a is None or len(good_a) == 0:
        plt.plot(origin_x, origin_y, "ro", markersize=5, label="shared origin")
        plt.xlabel("x (pixels)")
        plt.ylabel("y (pixels)")
        plt.title(title)
        plt.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        show_or_save(save_path)
        return

    for i, (a_value, curve) in enumerate(
        build_detected_projected_curves(
            a_values=np.asarray(good_a, dtype=float),
            geometry=projection_geometry,
            xp_span=(float(Xp_min), float(Xp_max)),
            image_shape=(int(H), int(W)),
            n_samples=int(n_samples_per_parabola),
        )
    ):
        for segment_index, segment in enumerate(curve.segments):
            plt.plot(
                segment.x,
                segment.y,
                "--",
                linewidth=curve_linewidth,
                alpha=curve_alpha,
                label=f"a_{i} = {float(a_value):.3g}" if segment_index == 0 else None,
                clip_on=True,
            )

    plt.plot(origin_x, origin_y, "ro", markersize=5, label="shared origin")
    plt.xlim(0, W)
    plt.ylim(0, H)
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.title(title)
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    show_or_save(save_path)
