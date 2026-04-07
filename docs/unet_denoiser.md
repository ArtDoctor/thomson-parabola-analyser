# UNet Denoiser

The UNet denoiser is a deep learning component designed to reduce background noise and artifacts in raw Thomson parabola spectrometer images, ensuring clean signal extraction for the downstream trace detection algorithms.

## Architecture

The model uses a `DeeperUNet` architecture, which is a variant of the standard U-Net. It features symmetric encoding and decoding paths with skip connections, optimized for localized feature extraction such as detecting continuous ion tracks over noisy backgrounds.

## Inference Pipeline (`oblisk/runtime_unet.py`)

1. **Preprocessing:** 
   - Images are converted to grayscale. 
   - Very large images (e.g., above 2000px longest side) are optionally downscaled to fit memory limits, processed, and then upscaled back using bilinear interpolation.
2. **Patch-based Processing:**
   - Because full-resolution spectrometer images can exceed GPU memory, inference is performed using a sliding window approach.
   - **Patch Size:** 512x512 pixels.
   - **Stride:** 256 pixels.
   - **Blending:** A Hann window is applied to each patch before accumulation to eliminate seam artifacts at patch boundaries.
3. **Hardware:** Automatically utilizes CUDA if available, falling back to CPU.

## Training

The network is trained using synthetic noisy images (generated via the Monte Carlo tracker and noise addition scripts). Model weights are saved as `.pth` checkpoints. The best performing model is typically stored at `unet-denoiser/unet_denoise_best.pth`.

The training-side workflow remains in `unet-denoiser/`, with:

- `train.py` as the script entrypoint
- `train.ipynb` as the notebook workflow
- `utils/` as the training-only dataset/model/loss helpers
