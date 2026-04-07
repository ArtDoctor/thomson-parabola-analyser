# Core Processing Pipeline

The core processing pipeline orchestrates the transformation of a raw Thomson parabola image into physical energy spectra. It integrates all individual components (YOLO, UNet, Physics formulas) into a cohesive workflow.

## 1. Image Preprocessing
- **ROI Cropping:** Uses the YOLO model (`thomson-cutter.onnx`) to discard irrelevant edges and borders.
- **Denoising:** Uses the UNet denoiser (`unet_denoise.py`) to suppress background radiation noise while preserving continuous ion traces.

## 2. Background Subtraction
- Extracts a background intensity estimate from a region devoid of ion traces. This scalar value (`bg_mean`) is subtracted from the image during energy sampling to ensure accurate flux integration.

## 3. Trace Detection & Smoothing
- **Morphological Operations:** Enhances the contrast of continuous streaks against the background.
- **Line Tracking:** Traces the center of intensity for each observed streak.
- **Smoothing (`utils/line_processing.py`):** Uses a sliding window median approach to detect and correct outlier points in the traced lines, applying a moving average to ensure physical continuity.

## 4. Frame Alignment
- The detector plate is rarely perfectly aligned with the magnetic and electric axes.
- The pipeline mathematically fits a global coordinate origin $(x_0, y_0)$ and a rotation angle $\theta$. All detected points are transformed from pixel space $(x, y)$ into a canonical physical frame $(X', Y')$ where parabolas follow $Y' = a X'^2$ perfectly.

## 5. Parabola Fitting and Classification
- For each smoothed line, a curvature parameter $a$ is fitted.
- This parameter is compared to theoretical $a$ values derived from the spectrometer geometry and known ion species ($A/Z$ ratios).
- Lines are assigned to species (e.g., $C^{4+}$, $H^+$, $O^{6+}$) based on the closest match within acceptable tolerances.

## 6. Energy Spectra Extraction
- **Sampling:** Intensity is sampled along the exact theoretical trajectory of the classified parabola, ensuring even faint signals are captured correctly.
- **Energy Mapping:** The physical distance $X'$ of each sample from the origin is converted to kinetic energy using the established electromagnetic deflection formulas (`utils/energy.py`).
- **Plotting:** Final spectra are aggregated and plotted, yielding $\text{d}N/\text{d}E$ histograms for physical analysis.
