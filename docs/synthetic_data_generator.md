# Synthetic Data Generator

The synthetic data generator provides ground-truth Thomson parabola signatures used for validating the analytical pipeline, evaluating physics tolerances, and training the UNet denoiser. 

## Overview

The core is a fast, parallelized Monte Carlo particle tracker written in C++. It simulates the trajectories of various ion species (e.g., H+, C4+, O6+) as they travel through the spectrometer's electric and magnetic fields and hit the downstream detector.

## C++ Core Engine

Located in `synthetic_data/`:

- `thomson.cpp`: The full-featured tracker supporting multiple physics options (adaptive stepping, fringe fields, relativistic Boris pusher, and Gaussian beam profiles).
- `thomson_optimized.cpp`: A highly optimized version providing ~220x speedup. It achieves this by replacing iterative Boris pusher steps with exact analytical solutions (helical motion in $\mathbf{B}$, parabolic acceleration in $\mathbf{E}$, linear drift) in uniform field regions.
- `thomson_shared.h`: Contains shared physical constants and field definitions.

### Integration (Boris Pusher)
When simulating complex features like fringe fields (smooth `tanh` field boundaries), the tracker uses the Boris pusher algorithm. This algorithm is exact, symplectic, and correctly handles $\mathbf{E} \times \mathbf{B}$ particle rotation at a default time step of $5 \times 10^{-13}$ s.

## Python Wrappers and Tools

The Python scripts in `synthetic_data/` wrap the C++ engine to generate comprehensive datasets:

- **`generate_synthetic_dataset.py`**: Automates the execution of the C++ binary across randomized configurations to create a diverse dataset.
- **`synthetic_data/utils/noise.py` & `noise_adder.ipynb`**: Applies realistic IP (Image Plate) noise, background gradients, and artifacts to the clean Monte Carlo hit maps, bridging the domain gap between simulation and real experimental data.
- **`synthetic_data/utils/hits_to_img.py`**: Converts the raw 2D coordinate text files (`hits.txt`) into 2D histogram images.

## Outputs

The simulation produces a file (`hits.txt`) where each row contains:
`<deflection_y>  <deflection_x>  <K_MeV>  [<particle_label>]`

These hits are subsequently rendered into high-resolution TIFF/PNG images which look exactly like theoretical Thomson parabolas, used as standard references across the project.
