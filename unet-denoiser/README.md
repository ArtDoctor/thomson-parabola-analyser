# UNet Denoiser Workflow

This directory contains the UNet training workflow and the checkpoint used by the main Oblisk runtime.

## What lives here

- `unet_denoise_best.pth`: runtime checkpoint loaded by the main pipeline by default
- `train.py`: scriptable training entrypoint
- `train.ipynb`: notebook version of the same workflow
- `utils/`: dataset, model, loss, and inference helpers used by training
- `clean/` and `noisy/`: paired synthetic training images
- `test/`: example noisy images used for post-training inference snapshots
- `output/`: generated plots and inference previews
- `checkpoints/`: training checkpoints

## Runtime relationship

The main analysis pipeline loads `unet_denoise_best.pth` from this directory by default. That path is configurable through `OBLISK_UNET_CHECKPOINT_PATH`.

## Training

From the repository root:

```bash
python unet-denoiser/train.py
```

The script now resolves dataset, checkpoint, output, and test paths relative to `unet-denoiser/` instead of depending on the current shell working directory. It also avoids unconditional CUDA AMP so it can run on CPU-only environments.
