# Thomson Spectrometer Simulation

A fast, parallelised Monte Carlo particle tracker for a [Thomson spectrometer](https://en.wikipedia.org/wiki/Thomson_problem) setup. Ions are propagated through crossed magnetic and electric fields using the Boris pusher algorithm and their hit positions recorded on a downstream detector plane.

## Physics

The simulated geometry consists of (along the beam axis `z`):


| Region                        | Length      | Field            |
| ----------------------------- | ----------- | ---------------- |
| Magnetic deflector            | 8 cm        | `Bx = 0.144 T`   |
| 1 cm gap                      | —           | —                |
| Electric deflector            | 8 cm        | `Ex = 0.13 MV/m` |
| Drift to detector             | ~21.5 cm    | —                |
| **Total (source → detector)** | **37.5 cm** | —                |


Particles are deflected in orthogonal directions by the two fields, so a particle's position on the detector encodes both its **charge-to-mass ratio** and its **kinetic energy** — the classic Thomson parabola signature.

### Integration

Particle equations of motion are integrated with the **Boris pusher** (exact, symplectic rotation), at a default time step of `DT = 5×10⁻¹³ s`. An optional adaptive stepper coarsens the step by 10× during free-drift regions far from any field boundary.

## Files


| File                             | Description                                                                 |
| -------------------------------- | --------------------------------------------------------------------------- |
| `thomson.cpp`                    | Full-featured simulation with CLI options (particle species, physics flags) |
| `thomson_optimized.cpp`          | Performance-optimized version with analytic transport (~220× faster)        |
| `thomson_shared.h`               | Shared header: constants, types, field functions (used by both versions)    |
| `run.sh`                         | Build `thomson.cpp` and run it, writing hits to `hits.txt`                  |
| `run_optimized.sh`               | Build `thomson_optimized.cpp` and run it, writing hits to `hits.txt`        |
| `generate_synthetic_dataset.py`  | Builds paired clean/noisy training data used by the UNet workflow           |
| `generate_eval_synth.py`         | Builds the broader `eval_synth/` evaluation dataset with JSON ground truth  |
| `spot_utils.py`                  | Synthetic bright-spot generation used to mimic the beam-origin feature      |
| `utils/`                         | Rendering and noise helpers for synthetic image generation                  |
| `test_regression.sh`             | Checks that the default `thomson.cpp` run stays deterministic               |
| `test_physics.sh`                | Physics validation: each optional feature is compared against baseline      |
| `test_optimized.sh`              | Compares optimized vs original output across all configurations             |
| `hits.txt`                       | Output hit file (generated at runtime)                                      |


## Building & Running

### Canonical image generation

If you need a synthetic detector image for debugging, tests, evaluation setup, or fixture refreshes, use:

```bash
bash synthetic_data/generate_image.sh
```

This is the canonical image-generation path for the repository. It runs the optimized Thomson simulator and then renders `hits.txt` into `synthetic_data/detector_image.png` plus `synthetic_data/noisy_detector_image.png`.

Do not create new repository synthetic-image fixtures by hand-drawing idealized parabolas unless the user explicitly asks for that. Tests and checked-in fixtures should be derived from `generate_image.sh` so they stay aligned with the simulator physics and rendering pipeline.

### Quick start

From the repository root:

```bash
bash synthetic_data/run.sh
```

From inside `synthetic_data/`, the shorter `bash run.sh` form works too.

This compiles `thomson.cpp` with `-O3 -fopenmp`, runs the simulation, writes results to `synthetic_data/hits.txt`, then removes the binary.

> **Note:** `run.sh` uses `-march=native` for maximum performance on your machine. Remove that flag if you intend to share the binary.

### Manual build

```bash
cd synthetic_data
g++ -O3 -march=native -fopenmp thomson.cpp -o thomson
./thomson [options] > hits.txt
```

### Passing arguments through `run.sh`

All arguments passed to `run.sh` are forwarded to the binary:

```bash
bash synthetic_data/run.sh -relativistic -fringe -particle C4 -particle O6
```

### Running the optimized version

```bash
bash synthetic_data/run_optimized.sh [same options as above]
```

The optimized binary accepts the same CLI flags and produces output in the same format.

## CLI Options (`thomson`)


| Flag             | Description                                                            |
| ---------------- | ---------------------------------------------------------------------- |
| `-particle SPEC` | Simulate this particle species (repeatable). Default: `C1`             |
| `-relativistic`  | Use relativistic Boris pusher                                          |
| `-fringe`        | Replace sharp field boundaries with smooth tanh fringe profiles        |
| `-adaptive`      | Enable adaptive time stepping (10× coarser step in field-free regions) |
| `-beam`          | Sample initial positions from a Gaussian beam (σ = 0.5 mm)             |
| `-h` / `--help`  | Print usage and exit                                                   |


### Particle specification (`SPEC`)

The format is `<element><charge>`, e.g.:

- `C4` — Carbon, 4+ charge state
- `H` — Proton (charge defaults to +1)
- `O6` — Oxygen 6+
- `Si2` — Silicon 2+
- `e` — Electron (charge defaults to −1)

Supported elements: H, He, Li, Be, B, C, N, O, F, Ne, Si, P, S, Cl, Ar, Fe, e.

## Output Format

Each row of `hits.txt` corresponds to one particle that reached the detector:

```
<deflection_y>  <deflection_x>  <K_MeV>  [<particle_label>]
```

- Column 1: vertical deflection (y, in metres) — driven by the **electric** field
- Column 2: horizontal deflection (x, in metres) — driven by the **magnetic** field
- Column 3: initial kinetic energy in MeV
- Column 4 (thomson only): particle label, e.g. `C4`

Progress and hit counts are written to `stderr`.

## Simulation Parameters


| Parameter                   | Value                       |
| --------------------------- | --------------------------- |
| Particles per species       | 200 000                     |
| Energy range                | 0.030 – 0.889 MeV (uniform) |
| Beam divergence (σ)         | 1 mrad                      |
| Beam spot size (σ, `-beam`) | 0.5 mm                      |
| Max integration steps       | 2 000 000                   |
| Default time step           | 5×10⁻¹³ s                   |


## Testing

### Regression test

Verifies that `thomson.cpp` (no flags, default C¹⁺) still produces deterministic output across repeated runs:

```bash
bash synthetic_data/test_regression.sh
```

### Physics validation

Runs each optional feature individually and checks that the mean deflection does not deviate from baseline by more than the expected tolerance:

```bash
bash synthetic_data/test_physics.sh
```


| Feature         | Tolerance |
| --------------- | --------- |
| `-relativistic` | < 0.01 %  |
| `-fringe`       | < 5 %     |
| `-adaptive`     | < 0.01 %  |
| `-beam`         | < 1 %     |


### Optimized vs original comparison

Runs both `thomson.cpp` and `thomson_optimized.cpp` with shared RNG seeds across 6 configurations, comparing mean deflections:

```bash
bash synthetic_data/test_optimized.sh
```


| Configuration       | Tolerance |
| ------------------- | --------- |
| Baseline (no flags) | < 0.01 %  |
| `-relativistic`     | < 0.01 %  |
| `-adaptive`         | < 0.01 %  |
| `-beam`             | < 1 %     |
| `-fringe`           | < 5 %     |
| Multi-particle      | < 0.01 %  |


## Optimized version

`thomson_optimized.cpp` implements three performance optimizations:

1. **Analytic transport** — replaces millions of Boris timesteps per particle with exact analytic solutions for each field region (helical motion in B, parabolic acceleration in E, linear drift)
2. **Thread-local vector pre-allocation** — reserves memory for thread-local hit vectors to avoid heap contention
3. **Precomputed particle constants** — stores `q/m` and `mc²` once at startup instead of recomputing per-step

The analytic transport provides the dominant speedup (~220×) for non-fringe modes. Fringe mode falls back to the Boris pusher since smooth tanh profiles don't have closed-form solutions.

## Requirements

- C++11-compatible compiler (GCC or Clang recommended)
- OpenMP (usually included with GCC via `-fopenmp`)
- Standard UNIX shell tools (`bash`, `awk`, `sort`, `diff`) for the test scripts
