# Thomson Spectrometer Physics

This document outlines the physical principles and mathematical formulations used to analyze Thomson parabola spectrometer images in the Oblisk pipeline.

## Overview

A Thomson parabola spectrometer uses parallel or crossed electric ($\mathbf{E}$) and magnetic ($\mathbf{B}$) fields to deflect ions. Ions enter the fields through a pinhole, and their deflection on a downstream detector plane depends on their velocity $v$ and charge-to-mass ratio $q/m$. 

Because the electric deflection scales with $1/v^2$ and the magnetic deflection scales with $1/v$, particles of the same $q/m$ but varying energies form a parabolic trace on the detector.

## Geometry and Fields

The standard configuration used in the pipeline includes:
- **Magnetic deflector:** Length $L_{iB} = 8$ cm, field $B_x \approx 0.144$ T.
- **Electric deflector:** Length $L_{iE} = 8$ cm, field $E_x \approx 0.13$ MV/m.
- **Drift distance:** Varies by detector configuration (e.g., 15x15 cm vs. 6x6 cm Image Plates).

The coordinate frame of the detector is oriented such that magnetic deflection occurs along the $x$-axis and electric deflection along the $y$-axis.

## Parabola Equation

In the detector plane (after transforming to the aligned physical frame $X', Y'$), the trace of a specific ion species follows the equation:
$$ Y' = a \cdot X'^2 $$

The curvature parameter $a$ depends solely on the spectrometer geometry and the ion's charge-to-mass ratio. It is calculated via a kinematic factor $K$:
$$ K = \frac{(q/m) \cdot E \cdot L_{iE} \cdot (L_{iE}/2 + L_{fE})}{B^2 \cdot L_{iB}^2 \cdot (L_{iB}/2 + L_{fB})^2} $$

where:
- $L_{fB}$ is the drift length from the end of the magnetic field to the detector.
- $L_{fE}$ is the drift length from the end of the electric field to the detector.

In the pipeline's image coordinates, $a$ is adjusted by the pixel scale (meters per pixel) to directly relate pixel coordinates: $a_{px} = K / (\text{pixels per meter})$.

## Image Tilts and Rotations

Real spectrometer images often contain misalignment (translation and rotation). Before extracting physical quantities, the pipeline fits a coordinate transformation:
- **Global Origin $(x_0, y_0)$:** The theoretical pinhole projection (zero deflection point) on the image.
- **Rotation $\theta$:** The angle between the image axes and the physical $X', Y'$ deflection axes.

All observed points $(x, y)$ are transformed to the canonical frame $(X', Y')$ using these parameters.

## Energy Calculation

Kinetic energy is derived from the magnetic deflection $X'$ (which is orthogonal to the electric field deflection). The velocity $v$ is:
$$ v = \frac{q \cdot B \cdot L_{iB} \cdot (L_{iB}/2 + L_{fB})}{m \cdot X'} $$

From $v$, the kinetic energy $E_k$ in Joules is:
$$ E_k = \frac{1}{2} m v^2 $$
This value is subsequently converted to electron-volts (keV or MeV) for spectra plotting. The calculations are handled in `oblisk/analysis/energy.py` and `oblisk/analysis/physics.py`.
