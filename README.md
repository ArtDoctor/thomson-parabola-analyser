# Oblisk

Oblisk is an end-to-end Thomson parabola spectrometer analysis repository.

The main runtime path is:

1. load a detector image
2. crop the active detector region with YOLO
3. denoise with UNet or morphological opening
4. detect and merge trace candidates
5. fit shared geometry and classify species
6. build spectra and export plots plus `res.json`

## Quickstart

Create the local environment and install dependencies:

```bash
bash setup.sh
source venv/bin/activate
```

Run one image through the CLI (point at your detector file — here `example.png` is a convenient local example when present):

```bash
python main.py example.png --output-dir outputs --add-plots
```

## Inputs And Models

- `eval_synth/` is the broader synthetic evaluation dataset generated from `synthetic_data/`.
- The runtime expects the detector model at `yolo-tune/thomson-cutter.onnx` and the UNet checkpoint at `unet-denoiser/unet_denoise_best.pth` by default.
- Those runtime model paths can be overridden with `OBLISK_YOLO_MODEL_PATH` and `OBLISK_UNET_CHECKPOINT_PATH`.
- While decision to keep models in the repository and git isn't exactly a best practice, introducing a proper model tracking and versioning framework would take a lot of time, but wouldn't bring any major improvements. The models are not going to be updated often, so they are kept as plain files.

## Outputs

For a single-image run, the output directory contains:

- `res.json`: structured result payload with classifications, spectra, timings, and evaluation summaries
- `plots/`: diagnostic PNGs when `--add-plots` or `--add-plots-full` is used

For batch runs, the CLI also writes batch summary material under the chosen output directory.

## Testing

Run the full pytest suite (non-interactive matplotlib backend, venv activated):

```bash
bash test.sh
```

End-to-end coverage includes `synthetic_data/detector_image.png` (gitignored until you generate it). Create or refresh that file with:

```bash
bash synthetic_data/generate_image.sh
```

Type-checking and linting are separate: `./mypy.sh` and `flake8 .`. Tests marked optional `integration` skip when matching local fixture data is absent.

## Web App

The browser UI lives under `web/` and uses the same core Python package. The backend is in `web/backend/` and the frontend is in `web/frontend/`.

## Repository Layout

- `oblisk/`: main Python package for runtime analysis, preprocessing, reporting, and shared services
- `tests/`: unit, smoke, synthetic-regression, and integration coverage
- `docs/`: pipeline notes, subsystem docs, attempts, and supporting research material
- `synthetic_data/`: simulator and dataset-generation utilities
- `unet-denoiser/`: UNet training workflow and artifacts
- `yolo-tune/`: detector-model training assets and exported ONNX runtime model
- `web/`: FastAPI backend and Vite frontend
- `eval_synth/`: local synthetic evaluation dataset (optional; generated)

## More Detail

Start with these documents if you need subsystem-level context:

- `docs/core_pipeline.md`
- `docs/physics.md`
- `docs/unet_denoiser.md`
- `docs/yolo_detector.md`
- `docs/synthetic_data_generator.md`
- `docs/web_application.md`
