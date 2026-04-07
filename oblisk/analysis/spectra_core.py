from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

from oblisk.analysis.background import BackgroundRoi, compute_background_mean
from oblisk.analysis.geometry import xp_span_px_from_image, xp_span_px_from_points
from oblisk.analysis.species import (
    A_BY_SYM,
    candidate_from_mapping,
    nearby_names,
    parse_species,
    same_mq_names,
)
from oblisk.analysis.spectra_band_integration import _sample_single_spectrum
from oblisk.analysis.spectra_models import (
    AbsoluteSpectrumCurve,
    ClassifiedLine,
    IonSpectrum,
    SpectraResult,
    SpectrumGeometry,
)
from oblisk.analysis.spectra_pixels import _build_pixel_ownership


def _classified_line_from_mapping(raw_line: Mapping[str, Any]) -> ClassifiedLine:
    points_value = raw_line["points"] if "points" in raw_line else None
    points = None if points_value is None else np.asarray(points_value, dtype=float)
    raw_candidates = raw_line["candidates"] if "candidates" in raw_line else []
    candidates = [candidate_from_mapping(candidate) for candidate in raw_candidates]
    label = str(raw_line["label"]) if "label" in raw_line else ""

    return ClassifiedLine(
        a=float(raw_line["a"]),
        label=label,
        candidates=candidates,
        points=points,
    )


def _normalize_classified_lines(
    classified: Sequence[ClassifiedLine | Mapping[str, Any]],
) -> list[ClassifiedLine]:
    normalized: list[ClassifiedLine] = []
    for line in classified:
        if isinstance(line, ClassifiedLine):
            normalized.append(line)
        else:
            normalized.append(_classified_line_from_mapping(line))
    return normalized


def infer_xp_bounds_px(
    image: np.ndarray,
    classified: Sequence[ClassifiedLine | Mapping[str, Any]],
    geometry: SpectrumGeometry,
    pad_px: float = 0.0,
) -> tuple[float, float]:
    normalized = _normalize_classified_lines(classified)
    points_list = [
        line.points for line in normalized if line.points is not None and len(line.points) > 0
    ]
    if points_list:
        return xp_span_px_from_points(
            points_list=points_list,
            x0_fit=geometry.x0_fit,
            y0_fit=geometry.y0_fit,
            theta_fit=geometry.theta_fit,
            pad_px=pad_px,
        )
    return xp_span_px_from_image(
        image=image,
        x0_fit=geometry.x0_fit,
        y0_fit=geometry.y0_fit,
        theta_fit=geometry.theta_fit,
        pad_px=pad_px,
    )


def build_spectra_result(
    image: np.ndarray,
    classified: Sequence[ClassifiedLine | Mapping[str, Any]],
    geometry: SpectrumGeometry,
    match_tol: float,
    background_roi: BackgroundRoi | None = None,
    xp_bounds_px: tuple[float, float] | None = None,
    eps_px: float = 3.0,
    sample_count: int = 2500,
    local_mean_radius_px: int = 2,
    integration_windows_a: np.ndarray | None = None,
    integration_a_samples: int = 33,
    num_energy_bins: int = 500,
    smoothing_sigma: float = 2.0,
) -> SpectraResult:
    normalized = _normalize_classified_lines(classified)
    image_float = image.astype(float)
    roi = background_roi if background_roi is not None else BackgroundRoi()
    clipped_roi, background_mean = compute_background_mean(image=image_float, roi=roi)
    final_xp_bounds_px = (
        xp_bounds_px if xp_bounds_px is not None else infer_xp_bounds_px(image, normalized, geometry)
    )

    pixel_ownership = _build_pixel_ownership(
        image=image_float,
        normalized=normalized,
        geometry=geometry,
        integration_windows_a=integration_windows_a,
        eps_px=eps_px,
    )

    spectra: list[IonSpectrum] = []
    all_energies: list[np.ndarray] = []

    for line_index, line in enumerate(normalized):
        if not line.candidates:
            continue

        best_match = line.candidates[0]
        symbol, charge_state = parse_species(best_match.name)
        if symbol is None or charge_state is None:
            continue
        if symbol not in A_BY_SYM:
            continue

        nearby = [name for name in nearby_names(line.candidates, max_rel=match_tol) if name != best_match.name]
        same_mq = [name for name in same_mq_names(line.candidates) if name != best_match.name]

        integration_window = None
        iw = integration_windows_a
        if iw is not None and line_index < len(iw) and len(iw[line_index]) == 2:
            left_a = float(iw[line_index][0])
            right_a = float(iw[line_index][1])
            integration_window = (left_a, right_a)

        energies_keV, weights, polyline = _sample_single_spectrum(
            image=image_float,
            a_rot=float(line.a),
            mass_number=A_BY_SYM[symbol],
            charge_state=charge_state,
            geometry=geometry,
            background_mean=background_mean,
            xp_bounds_px=final_xp_bounds_px,
            eps_px=eps_px,
            sample_count=sample_count,
            local_mean_radius_px=local_mean_radius_px,
            integration_window_a=integration_window,
            integration_a_samples=integration_a_samples,
            spectrum_index=line_index if (pixel_ownership is not None and integration_window is not None) else None,
            pixel_ownership=pixel_ownership,
        )

        spectrum = IonSpectrum(
            label=best_match.name,
            symbol=symbol,
            mass_number=A_BY_SYM[symbol],
            charge_state=charge_state,
            a_rot=float(line.a),
            nearby=nearby,
            same_mq=same_mq,
            energies_keV=energies_keV,
            weights=weights,
            polyline=polyline,
        )
        spectra.append(spectrum)
        if len(energies_keV) > 0:
            all_energies.append(energies_keV)

    if not all_energies:
        return SpectraResult(
            background_roi=clipped_roi,
            background_mean=background_mean,
            xp_bounds_px=final_xp_bounds_px,
            spectra=spectra,
        )

    energy_all = np.concatenate(all_energies)
    finite_energy_all = energy_all[np.isfinite(energy_all)]
    energy_max = float(np.percentile(finite_energy_all, 99.5))
    bins = np.linspace(0.0, energy_max, num_energy_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    for spectrum in spectra:
        if len(spectrum.energies_keV) == 0:
            spectrum.normalized_signal = np.zeros_like(centers)
            continue

        histogram, _ = np.histogram(
            np.clip(spectrum.energies_keV, 0.0, energy_max),
            bins=bins,
            weights=spectrum.weights,
        )
        smoothed = gaussian_filter1d(histogram.astype(float), sigma=smoothing_sigma, mode="nearest")
        max_value = float(np.max(smoothed)) if np.any(smoothed) else 1.0
        spectrum.normalized_signal = smoothed / max_value if max_value > 0.0 else smoothed

    return SpectraResult(
        background_roi=clipped_roi,
        background_mean=background_mean,
        xp_bounds_px=final_xp_bounds_px,
        energy_centers_keV=centers,
        spectra=spectra,
    )


def _dedupe_names(values: Sequence[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def build_label_to_index_map(result: SpectraResult) -> dict[str, int]:
    return {
        spectrum.label: index
        for index, spectrum in enumerate(result.spectra, start=1)
        if spectrum.label
    }


def _build_absolute_log_curves(
    result: SpectraResult,
    e_min_log_keV: float,
    e_max_log_keV: float,
    num_bins: int,
    smoothing_sigma: float,
) -> tuple[np.ndarray, list[AbsoluteSpectrumCurve], float]:
    if e_min_log_keV <= 0.0:
        raise ValueError("e_min_log_keV must be positive for logarithmic binning.")
    if e_max_log_keV <= e_min_log_keV:
        raise ValueError("e_max_log_keV must be larger than e_min_log_keV.")

    bins = np.logspace(np.log10(e_min_log_keV), np.log10(e_max_log_keV), num_bins + 1)
    centers = np.sqrt(bins[:-1] * bins[1:])

    curves: list[AbsoluteSpectrumCurve] = []
    global_ymax = 0.0
    for index, spectrum in enumerate(result.spectra, start=1):
        if spectrum.energies_keV.size == 0:
            y_abs = np.zeros_like(centers)
        else:
            ek = spectrum.energies_keV
            in_range = (ek >= e_min_log_keV) & (ek <= e_max_log_keV) & np.isfinite(ek)
            energies_in = spectrum.energies_keV[in_range]
            weights_in = spectrum.weights[in_range]
            histogram, _ = np.histogram(energies_in, bins=bins, weights=weights_in)
            density = histogram.astype(float) / np.diff(bins)
            y_abs = gaussian_filter1d(density, sigma=smoothing_sigma, mode="nearest")

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

    return centers, curves, global_ymax
