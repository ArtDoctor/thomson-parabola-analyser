from pathlib import Path

import numpy as np
import torch

from oblisk.runtime_unet import denoise_image as _denoise_image
from oblisk.runtime_unet import preload_unet_denoiser as _preload_unet_denoiser
from oblisk.runtime_yolo import Coords, cut_detector_image as _cut_detector_image


def cut_detector_image(
    image: np.ndarray,
    model_path: Path | None = None,
) -> Coords:
    return _cut_detector_image(image, model_path=model_path)


def preload_unet_denoiser(
    checkpoint_path: Path | None = None,
    device: torch.device | None = None,
) -> None:
    _preload_unet_denoiser(checkpoint_path=checkpoint_path, device=device)


def denoise_image(
    image: np.ndarray,
    checkpoint_path: Path | None = None,
    patch_size: int = 512,
    stride: int = 256,
    device: torch.device | None = None,
) -> np.ndarray:
    return _denoise_image(
        image,
        checkpoint_path=checkpoint_path,
        patch_size=patch_size,
        stride=stride,
        device=device,
    )
