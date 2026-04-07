from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from oblisk.analysis.overlay import (
    ProjectionGeometry,
    build_classified_projected_curves,
    project_parabola_curve,
    project_origin_point,
)
from oblisk.analysis.geometry import PerspectiveReference
from oblisk.plot_display import show_or_save


def plot_grayscale_panel(
    image: np.ndarray,
    title: str,
    save_path: Path | None = None,
) -> None:
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap="gray", origin="lower")
    plt.title(title)
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.tight_layout()
    show_or_save(save_path)


def plot_original_denoised_rgb_overlay(
    image_original: np.ndarray,
    image_denoised: np.ndarray,
    title: str,
    save_path: Path | None = None,
) -> None:
    if image_original.shape != image_denoised.shape:
        raise ValueError("image_original and image_denoised must have the same shape")
    vmax = float(max(int(np.max(image_original)), int(np.max(image_denoised)), 1))
    r = image_original.astype(np.float64) / vmax
    b = image_denoised.astype(np.float64) / vmax
    g = np.zeros_like(r)
    rgb = np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)

    plt.figure(figsize=(10, 8))
    plt.imshow(rgb, origin="lower", interpolation="nearest")
    plt.title(title)
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.tight_layout()
    show_or_save(save_path)


def plot_peaks_overlay(
    image: np.ndarray,
    peaks_by_row: list[list[tuple[int, int]]],
    title: str,
    save_path: Path | None = None,
) -> None:
    xs: list[int] = []
    ys: list[int] = []
    for row in peaks_by_row:
        for col, row_idx in row:
            xs.append(col)
            ys.append(row_idx)
    plt.figure(figsize=(12, 8))
    plt.imshow(image, cmap="gray", origin="lower")
    if xs:
        plt.scatter(xs, ys, s=3, c="cyan", alpha=0.65, linewidths=0)
    plt.title(title)
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.tight_layout()
    show_or_save(save_path)


def plot_lines_with_labels(
    filtered_lines: list[list[list[int]]],
    image: np.ndarray,
    save_path: Path | None = None,
    title: str = "Filtered & Merged Lines with Boxed Labels",
) -> None:
    plt.figure(figsize=(12, 8))
    plt.imshow(image, cmap="gray", origin="lower")

    line_handles, line_labels = [], []

    for idx, line in enumerate(filtered_lines):
        if len(line) < 2:
            continue
        arr = np.array(line)

        handle, = plt.plot(arr[:, 1], arr[:, 2], linewidth=1, label=f"Line {idx}")
        mid_idx = max(0, min(len(arr) // 2, len(arr) - 1))
        mid_x, mid_y = float(arr[mid_idx, 1]), float(arr[mid_idx, 2])
        plt.text(
            mid_x,
            mid_y,
            str(idx),
            fontsize=9,
            color="white",
            ha="center",
            va="center",
            bbox=dict(facecolor="black", alpha=0.6, boxstyle="round,pad=0.3"),
        )

        line_handles.append(handle)
        line_labels.append(f"Line {idx}")

    plt.title(title)
    if line_handles:
        plt.legend(line_handles, line_labels, loc="lower right", fontsize=8, ncol=2)
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.tight_layout()
    show_or_save(save_path)


def plot_classified_rot(
    classified: list[dict],
    hydrogen_line: dict,
    Xp_span: tuple[float, float],
    x0_fit: float,
    y0_fit: float,
    theta_fit: float,
    gamma_fit: float,
    delta_fit: float,
    title: str,
    image: np.ndarray | None = None,
    save_path: Path | None = None,
    k1_fit: float = 0.0,
    k2_fit: float = 0.0,
    img_center: tuple[float, float] | None = None,
    img_diag: float | None = None,
    perspective_reference: PerspectiveReference | None = None,
) -> None:
    curve_linewidth = 0.7
    curve_alpha = 0.88

    plt.figure(figsize=(14, 11))
    if image is not None:
        plt.imshow(image, cmap="gray", origin="lower")

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

    handles: list = []
    labels: list[str] = []
    curve_xs: list[np.ndarray] = []
    curve_ys: list[np.ndarray] = []

    image_shape = (
        (int(image.shape[0]), int(image.shape[1]))
        if image is not None
        else (
            int(max(1, round(img_center[1] * 2))) if img_center is not None else 1,
            int(max(1, round(img_center[0] * 2))) if img_center is not None else 1,
        )
    )

    for entry_index, curve in build_classified_projected_curves(
        classified=classified,
        geometry=projection_geometry,
        xp_span=(float(Xp_span[0]), float(Xp_span[1])),
        image_shape=image_shape,
        n_samples=600,
    ):
        match = classified[entry_index]
        a_i = float(match["a"])
        curve_xs.extend([segment.x for segment in curve.segments])
        curve_ys.extend([segment.y for segment in curve.segments])

        fit_handle = None
        for segment in curve.segments:
            handle, = plt.plot(
                segment.x,
                segment.y,
                linestyle="--",
                linewidth=curve_linewidth,
                alpha=curve_alpha,
                clip_on=True,
            )
            if fit_handle is None:
                fit_handle = handle
        if fit_handle is None or curve.label_anchor is None:
            continue

        lx, ly = curve.label_anchor
        plt.text(
            lx,
            ly,
            match.get("label", "?"),
            fontsize=6,
            color="white",
            ha="center",
            va="center",
            bbox=dict(facecolor="black", alpha=0.75, boxstyle="round,pad=0.12"),
        )

        candidates = match.get("candidates", [])
        max_legend_cands = 6
        if candidates:
            shown = candidates[:max_legend_cands]
            cand_legend = ", ".join(
                f"{candidate['name']} (Δ={100.0 * float(candidate['rel_err']):.1f}%)"
                for candidate in shown
            )
            if len(candidates) > max_legend_cands:
                cand_legend += f", … (+{len(candidates) - max_legend_cands})"
            legend_text = f"grp{entry_index}  a'={a_i:.3g}  → {cand_legend}"
        else:
            mq_meas = float(match.get("mq_meas", float("nan")))
            legend_text = f"grp{entry_index}  a'={a_i:.3g}  → ?  (m/q≈{mq_meas:.2f})"

        handles.append(fit_handle)
        labels.append(legend_text)

    def _truncate_legend_line(text: str, max_chars: int = 96) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1] + "\u2026"

    labels = [_truncate_legend_line(text) for text in labels]

    hydrogen_curve = project_parabola_curve(
        a_value=float(hydrogen_line["a"]),
        geometry=projection_geometry,
        xp_min=float(Xp_span[0]),
        xp_max=float(Xp_span[1]),
        image_shape=image_shape,
        n_samples=600,
    )
    if hydrogen_curve is not None:
        hydrogen_handle = None
        for segment in hydrogen_curve.segments:
            curve_xs.append(segment.x)
            curve_ys.append(segment.y)
            handle, = plt.plot(
                segment.x,
                segment.y,
                linestyle=":",
                linewidth=curve_linewidth,
                alpha=curve_alpha,
                clip_on=True,
            )
            if hydrogen_handle is None:
                hydrogen_handle = handle
        if hydrogen_handle is None:
            hydrogen_handle = Line2D([], [], linestyle=":", linewidth=curve_linewidth, color="gray")
        if hydrogen_curve.label_anchor is None:
            tx = float(hydrogen_curve.visible_x[len(hydrogen_curve.visible_x) // 2])
            ty = float(hydrogen_curve.visible_y[len(hydrogen_curve.visible_y) // 2])
        else:
            tx, ty = hydrogen_curve.label_anchor
        plt.text(
            tx,
            ty,
            "H^1+ (ref)",
            fontsize=6,
            color="white",
            ha="center",
            va="bottom",
            bbox=dict(facecolor="black", alpha=0.6, boxstyle="round,pad=0.12"),
        )
    else:
        hydrogen_handle = Line2D([], [], linestyle=":", linewidth=0.9, color="gray")

    plt.plot(origin_x, origin_y, marker="x", markersize=4, linewidth=0, color="cyan")

    if image is not None:
        plt.xlim(0, image.shape[1])
        plt.ylim(0, image.shape[0])
    elif curve_xs and curve_ys:
        all_x = np.concatenate(curve_xs)
        all_y = np.concatenate(curve_ys)
        all_x = np.concatenate([all_x, np.asarray([origin_x], dtype=float)])
        all_y = np.concatenate([all_y, np.asarray([origin_y], dtype=float)])
        margin = 20.0
        plt.xlim(float(np.min(all_x)) - margin, float(np.max(all_x)) + margin)
        plt.ylim(float(np.min(all_y)) - margin, float(np.max(all_y)) + margin)
    plt.title(
        f"{title}\n"
        f"shared vertex=({x0_fit:.1f},{y0_fit:.1f}),  "
        f"θ={theta_fit:.4f} rad, γ={gamma_fit:.3f}, δ={delta_fit:.3f}",
        fontsize=8,
        pad=2,
    )
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    fig = plt.gcf()
    ax = plt.gca()
    ax.tick_params(axis="both", which="major", labelsize=5)
    n_leg = len(handles) + 1
    ncol = 3 if n_leg > 6 else 2 if n_leg > 2 else 1
    nrows = (n_leg + ncol - 1) // ncol
    bottom_margin = min(0.42, max(0.13, 0.07 + 0.026 * float(nrows)))
    fig.subplots_adjust(left=0.03, right=0.995, top=0.945, bottom=bottom_margin)
    ax.legend(
        handles + [hydrogen_handle],
        labels + ["H reference"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=ncol,
        fontsize=5,
        borderaxespad=0.25,
        framealpha=0.94,
        handlelength=1.2,
        columnspacing=0.9,
    )
    show_or_save(save_path)


def plot_classified_rot_numbered(
    classified: list[dict],
    hydrogen_line: dict,
    Xp_span: tuple[float, float],
    x0_fit: float,
    y0_fit: float,
    theta_fit: float,
    gamma_fit: float,
    delta_fit: float,
    title: str,
    label_to_num: dict[str, int] | None = None,
    image: np.ndarray | None = None,
    save_path: Path | None = None,
    perspective_reference: PerspectiveReference | None = None,
) -> None:
    plt.figure(figsize=(18, 14), dpi=200)
    if image is not None:
        plt.imshow(image, cmap="gray", origin="lower")

    image_shape = (
        (int(image.shape[0]), int(image.shape[1]))
        if image is not None
        else (1, 1)
    )
    projection_geometry = ProjectionGeometry(
        x0_fit=float(x0_fit),
        y0_fit=float(y0_fit),
        theta_fit=float(theta_fit),
        gamma_fit=float(gamma_fit),
        delta_fit=float(delta_fit),
        perspective_reference=(
            perspective_reference
            if perspective_reference is not None
            else PerspectiveReference()
        ),
    )
    origin_x, origin_y = project_origin_point(projection_geometry)

    for entry_index, curve in build_classified_projected_curves(
        classified=classified,
        geometry=projection_geometry,
        xp_span=(float(Xp_span[0]), float(Xp_span[1])),
        image_shape=image_shape,
        n_samples=500,
    ):
        for segment in curve.segments:
            plt.plot(segment.x, segment.y, linestyle="--", linewidth=0.9)

        label_text = "?"
        candidates = classified[entry_index].get("candidates", [])
        if candidates and label_to_num is not None:
            primary_name = candidates[0].get("name")
            if primary_name in label_to_num:
                label_text = str(label_to_num[primary_name])

        if curve.label_anchor is None:
            mid_index = len(curve.visible_x) // 2
            tx = float(curve.visible_x[mid_index])
            ty = float(curve.visible_y[mid_index])
        else:
            tx, ty = curve.label_anchor
        plt.text(
            tx,
            ty,
            label_text,
            fontsize=8,
            color="white",
            ha="center",
            va="center",
            bbox=dict(facecolor="black", alpha=0.75, boxstyle="round,pad=0.2"),
        )

    hydrogen_curve = project_parabola_curve(
        a_value=float(hydrogen_line["a"]),
        geometry=projection_geometry,
        xp_min=float(Xp_span[0]),
        xp_max=float(Xp_span[1]),
        image_shape=image_shape,
        n_samples=600,
    )
    if hydrogen_curve is not None:
        for segment in hydrogen_curve.segments:
            plt.plot(segment.x, segment.y, linestyle=":", linewidth=0.9)
        mid_h = len(hydrogen_curve.visible_x) // 2
        plt.text(
            float(hydrogen_curve.visible_x[mid_h]),
            float(hydrogen_curve.visible_y[mid_h]),
            "H^1+ (ref)",
            fontsize=8,
            color="white",
            ha="center",
            va="bottom",
            bbox=dict(facecolor="black", alpha=0.6, boxstyle="round,pad=0.2"),
        )

    plt.plot(origin_x, origin_y, marker="x", markersize=5, linewidth=0, color="cyan")
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.title(
        f"{title}\n"
        f"shared vertex=({x0_fit:.1f},{y0_fit:.1f}),  "
        f"θ={theta_fit:.4f} rad, γ={gamma_fit:.3f}, δ={delta_fit:.3f}"
    )
    plt.tight_layout()
    show_or_save(save_path)
