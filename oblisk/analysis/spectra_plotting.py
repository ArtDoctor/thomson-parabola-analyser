import logging
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter1d

from oblisk.analysis.overlay import (
    ProjectionGeometry,
    project_origin_point,
    project_polyline_segments,
)
from oblisk.analysis.spectra_core import _build_absolute_log_curves, _dedupe_names
from oblisk.analysis.spectra_models import AbsoluteSpectrumCurve, SpectraResult, SpectrumGeometry
from oblisk.analysis.species import A_BY_SYM, parse_species
from oblisk.plot_display import show_or_save

logger = logging.getLogger(__name__)


def plot_log_spectra_shared_absolute(
    result: SpectraResult,
    num_bins: int = 400,
    smoothing_sigma: float = 2.0,
    min_percentile: float = 0.5,
    max_percentile: float = 99.5,
    min_energy_floor_keV: float = 1e-2,
    title: str = "Energy spectra per ion - log x-axis, shared ABSOLUTE y-scale",
    save_path: Path | None = None,
) -> None:
    valid_spectra = [spectrum.energies_keV for spectrum in result.spectra if spectrum.energies_keV.size > 0]
    if not valid_spectra:
        logger.warning("No valid spectra available to plot.")
        return

    energy_all = np.concatenate(valid_spectra)
    energy_all = energy_all[np.isfinite(energy_all) & (energy_all > 0.0)]
    if energy_all.size == 0:
        logger.warning("No spectra available to plot after filtering.")
        return

    e_min_log_keV = max(float(np.percentile(energy_all, min_percentile)), min_energy_floor_keV)
    e_max_log_keV = float(np.percentile(energy_all, max_percentile))

    centers, curves, global_ymax = _build_absolute_log_curves(
        result=result,
        e_min_log_keV=e_min_log_keV,
        e_max_log_keV=e_max_log_keV,
        num_bins=num_bins,
        smoothing_sigma=smoothing_sigma,
    )

    num_spectra = len(curves)
    cols = int(np.ceil(np.sqrt(num_spectra)))
    rows = int(np.ceil(num_spectra / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.8 * rows), sharex=True, sharey=True)

    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    axes = axes.reshape(rows, cols)

    for index, curve in enumerate(curves):
        row, col = divmod(index, cols)
        ax = axes[row, col]
        ax.semilogx(centers, curve.y_abs, linewidth=1.6)

        extras: list[str] = []
        if curve.same_mq:
            extras.append("same m/q: " + ", ".join(curve.same_mq))
        nearby_filtered = [name for name in curve.nearby if name not in curve.same_mq]
        if nearby_filtered:
            extras.append("nearby: " + ", ".join(nearby_filtered))

        panel_title = curve.label
        if extras:
            panel_title += " - " + " | ".join(extras)
        ax.set_title(panel_title, fontsize=10)
        if row == rows - 1:
            ax.set_xlabel("Energy [keV] (log)")
        if col == 0:
            ax.set_ylabel("Intensity (arb. units / keV)")
        ax.grid(True, which="both", linewidth=0.3)
        ax.set_xlim(e_min_log_keV, e_max_log_keV)
        ax.set_ylim(0.0, global_ymax * 1.05)

    for empty_index in range(num_spectra, rows * cols):
        row, col = divmod(empty_index, cols)
        axes[row, col].axis("off")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    show_or_save(save_path)


def plot_single_numbered_log_spectra(
    result: SpectraResult,
    num_bins: int = 500,
    smoothing_sigma: float = 2.0,
    min_percentile: float = 0.5,
    max_percentile: float = 99.5,
    min_energy_floor_keV: float = 1e-2,
    figsize: tuple[float, float] = (12.0, 8.0),
    dpi: int = 200,
    print_caption_map: bool = True,
    save_path: Path | None = None,
) -> dict[str, int]:
    valid_spectra = [spectrum.energies_keV for spectrum in result.spectra if spectrum.energies_keV.size > 0]
    if not valid_spectra:
        logger.warning("No valid spectra available to plot.")
        return {}

    energy_all = np.concatenate(valid_spectra)
    energy_all = energy_all[np.isfinite(energy_all) & (energy_all > 0.0)]
    if energy_all.size == 0:
        logger.warning("No spectra available to plot after filtering.")
        return {}

    e_min_log_keV = max(float(np.percentile(energy_all, min_percentile)), min_energy_floor_keV)
    e_max_log_keV = float(np.percentile(energy_all, max_percentile))

    centers, curves, global_ymax = _build_absolute_log_curves(
        result=result,
        e_min_log_keV=e_min_log_keV,
        e_max_log_keV=e_max_log_keV,
        num_bins=num_bins,
        smoothing_sigma=smoothing_sigma,
    )

    if not curves:
        raise ValueError("No spectra available to plot.")

    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    annotations: list[tuple[float, float, str]] = []

    for curve in curves:
        ax.semilogx(centers, curve.y_abs, linewidth=1.6)
        if np.any(np.isfinite(curve.y_abs)) and float(np.nanmax(curve.y_abs)) > 0.0:
            peak_index = int(np.nanargmax(curve.y_abs))
            x_peak = float(centers[peak_index])
            y_peak = float(curve.y_abs[peak_index])
        else:
            x_peak = float(np.sqrt(e_min_log_keV * e_max_log_keV))
            y_peak = 0.0
        annotations.append((x_peak, y_peak, str(curve.index)))

    ax.set_xlim(e_min_log_keV, e_max_log_keV)
    ax.set_ylim(0.0, global_ymax * 1.05)
    ax.grid(True, which="both", linewidth=0.3)
    ax.set_xlabel("Energy [keV] (log)")
    ax.set_ylabel("Intensity (arb. units / keV)")

    for x_peak, y_peak, idx_str in annotations:
        ax.text(
            x_peak,
            y_peak,
            idx_str,
            fontsize=11,
            fontweight="bold",
            color="white",
            ha="center",
            va="bottom",
            bbox=dict(facecolor="black", alpha=0.6, boxstyle="round,pad=0.35"),
        )

    plt.tight_layout()
    show_or_save(save_path)

    label_to_index = {curve.label: curve.index for curve in curves if curve.label}
    if print_caption_map:
        for curve in curves:
            symbol, charge_state = parse_species(curve.label) if curve.label else (None, None)
            mass_number = A_BY_SYM[symbol] if symbol in A_BY_SYM else None
            q_over_m = (charge_state / mass_number) if (charge_state is not None and mass_number) else float("nan")

            possible_species = [curve.label] if curve.label else []
            possible_species.extend(curve.same_mq)
            possible_species.extend([name for name in curve.nearby if name not in possible_species])
            possible_species = _dedupe_names([name for name in possible_species if name])

            species_str = (
                ", ".join(possible_species) if possible_species else "unknown"
            )
            logger.debug(
                "%d = q/m: %.4f e/amu; possible species: %s",
                curve.index,
                q_over_m,
                species_str,
            )

    return label_to_index


def plot_spectra_linear_energy_logy(
    result: SpectraResult,
    e_min_keV: float = 100.0,
    e_max_keV: float = 10000.0,
    num_bins: int = 1000,
    smoothing_sigma_bins: float = 2.0,
    edge_sigma_keep: float = 3.0,
    clip_min: float = 1.0,
    plot_e_max_keV: float = 4000.0,
    figsize: tuple[float, float] = (12.0, 8.0),
    dpi: int = 200,
    title: str = "Intensity, smoothed and all values below 1 set to 0",
    save_path: Path | None = None,
) -> None:
    if not result.spectra or all(spectrum.energies_keV.size == 0 for spectrum in result.spectra):
        logger.warning("No spectra available to plot.")
        return

    bins_keV = np.linspace(e_min_keV, e_max_keV, num_bins + 1, dtype=float)
    centers_keV = 0.5 * (bins_keV[:-1] + bins_keV[1:])
    bin_widths_keV = np.diff(bins_keV)
    centers_meV = centers_keV / 1000.0
    edge_keep = int(np.ceil(edge_sigma_keep * smoothing_sigma_bins))

    curves: list[AbsoluteSpectrumCurve] = []
    global_ymax = 0.0
    for index, spectrum in enumerate(result.spectra, start=1):
        energies = np.asarray(spectrum.energies_keV, dtype=float)
        weights = np.asarray(spectrum.weights, dtype=float)

        if energies.size == 0:
            y_abs = np.zeros_like(centers_keV)
        else:
            mask = (energies >= e_min_keV) & (energies <= e_max_keV)
            if not np.any(mask):
                y_abs = np.zeros_like(centers_keV)
            else:
                histogram, _ = np.histogram(energies[mask], bins=bins_keV, weights=weights[mask])
                density = histogram.astype(float) / bin_widths_keV
                if density.size > 0:
                    smoothed = gaussian_filter1d(density, sigma=smoothing_sigma_bins, mode="nearest")
                    keep = min(edge_keep, density.size // 2)
                    if keep > 0:
                        smoothed[:keep] = density[:keep]
                        smoothed[-keep:] = density[-keep:]
                    y_abs = smoothed
                else:
                    y_abs = density

        y_abs = np.where(y_abs < clip_min, 0.0, y_abs)
        if np.any(np.isfinite(y_abs)):
            global_ymax = max(global_ymax, float(np.nanmax(y_abs)))

        curves.append(
            AbsoluteSpectrumCurve(
                index=index,
                label=spectrum.label,
                nearby=list(spectrum.nearby),
                same_mq=list(spectrum.same_mq),
                energies_keV=spectrum.energies_keV,
                weights=spectrum.weights,
                y_abs=y_abs,
            )
        )

    if global_ymax <= 0.0:
        global_ymax = 1.0

    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    for curve in curves:
        ax.plot(centers_meV, curve.y_abs, linewidth=1.6, label=str(curve.index))

    ax.set_xlim(e_min_keV / 1000.0, plot_e_max_keV / 1000.0)
    ax.set_ylim(clip_min, global_ymax * 1.1)
    ax.set_yscale("log")
    ax.grid(True, which="both", linewidth=0.3)
    ax.set_xlabel("Energy [MeV] (linear)")
    ax.set_ylabel("Intensity (arb. units / keV, log scale)")
    ax.set_title(title)
    ax.legend(title="Index", fontsize=8)
    plt.tight_layout()
    show_or_save(save_path)


def plot_sampling_overlay(
    image: np.ndarray,
    result: SpectraResult,
    geometry: SpectrumGeometry,
    title: str = "Parabola samples used for spectra",
    save_path: Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image, cmap="gray", origin="lower")
    origin_x, origin_y = project_origin_point(
        ProjectionGeometry(
            x0_fit=float(geometry.x0_fit),
            y0_fit=float(geometry.y0_fit),
            theta_fit=float(geometry.theta_fit),
            gamma_fit=float(geometry.gamma_fit),
            delta_fit=float(geometry.delta_fit),
            k1_fit=float(geometry.k1_fit),
            k2_fit=float(geometry.k2_fit),
            img_center_x=float(geometry.img_center_x),
            img_center_y=float(geometry.img_center_y),
            img_diag=float(geometry.img_diag),
            perspective_reference=geometry.perspective_reference,
        )
    )

    roi = result.background_roi
    ax.add_patch(
        Rectangle(
            (roi.x0, roi.y0),
            roi.x1 - roi.x0,
            roi.y1 - roi.y0,
            facecolor="yellow",
            edgecolor="none",
            alpha=0.10,
            label="Background ROI",
        )
    )

    for spectrum in result.spectra:
        if spectrum.polyline is None:
            continue
        segments = project_polyline_segments(spectrum.polyline.x, spectrum.polyline.y)
        if not segments:
            continue
        for segment_index, segment in enumerate(segments):
            ax.plot(
                segment.x,
                segment.y,
                linewidth=0.9,
                label=spectrum.label if segment_index == 0 else None,
            )

    ax.plot(
        [origin_x],
        [origin_y],
        marker="x",
        markersize=5,
        color="cyan",
        linewidth=0,
        label="Vertex",
    )
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.tight_layout()
    show_or_save(save_path)


def plot_energy_spectra(
    result: SpectraResult,
    title: str | None = None,
    save_path: Path | None = None,
) -> None:
    if result.energy_centers_keV.size == 0:
        raise ValueError("No spectra were computed, so there is no common energy axis to plot.")

    num_spectra = len(result.spectra)
    cols = int(np.ceil(np.sqrt(num_spectra)))
    rows = int(np.ceil(num_spectra / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4.5 * cols, 3.6 * rows),
        sharex=True,
        sharey=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    axes = axes.reshape(rows, cols)

    for index, spectrum in enumerate(result.spectra):
        row, col = divmod(index, cols)
        ax = axes[row, col]
        ax.plot(result.energy_centers_keV, spectrum.normalized_signal, linewidth=1.6)

        extras: list[str] = []
        if spectrum.same_mq:
            extras.append("same m/q: " + ", ".join(spectrum.same_mq))
        nearby_filtered = [name for name in spectrum.nearby if name not in spectrum.same_mq]
        if nearby_filtered:
            extras.append("nearby: " + ", ".join(nearby_filtered))

        panel_title = spectrum.label
        if extras:
            panel_title += " - " + " | ".join(extras)
        ax.set_title(panel_title, fontsize=10)

        if row == rows - 1:
            ax.set_xlabel("Energy [keV]")
        if col == 0:
            ax.set_ylabel("Relative intensity")

        ax.grid(True, linewidth=0.3)
        ax.set_ylim(0.0, 1.05)

    for empty_index in range(num_spectra, rows * cols):
        row, col = divmod(empty_index, cols)
        axes[row, col].axis("off")

    if title is not None:
        fig.suptitle(title, fontsize=12)
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    else:
        plt.tight_layout()
    show_or_save(save_path)


def plot_spectra_summary(
    image: np.ndarray,
    result: SpectraResult,
    geometry: SpectrumGeometry,
    spectra_title: str | None = None,
) -> None:
    plot_sampling_overlay(image=image, result=result, geometry=geometry)
    if result.energy_centers_keV.size > 0:
        plot_energy_spectra(result=result, title=spectra_title)
