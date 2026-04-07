import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from oblisk.config import default_unet_checkpoint_path


class DeeperUNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1) -> None:
        super().__init__()

        def double_conv(in_c: int, out_c: int, norm: bool = True) -> nn.Sequential:
            layers: list[nn.Module] = [
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=not norm)
            ]
            if norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=not norm))
            if norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.d1 = double_conv(in_channels, 64, norm=False)
        self.d2 = double_conv(64, 128)
        self.d3 = double_conv(128, 256)
        self.d4 = double_conv(256, 512)

        self.pool = nn.MaxPool2d(2)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.u1 = double_conv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.u2 = double_conv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.u3 = double_conv(128, 64, norm=False)
        self.out = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.d1(x)
        x2 = self.d2(self.pool(x1))
        x3 = self.d3(self.pool(x2))
        x4 = self.d4(self.pool(x3))

        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.u1(x)
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.u2(x)
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.u3(x)
        return self.out(x)


def _create_window(window_size: int, device: torch.device) -> torch.Tensor:
    window_1d = torch.hann_window(window_size, device=device)
    window_2d = window_1d.unsqueeze(1) * window_1d.unsqueeze(0)
    return window_2d.unsqueeze(0).unsqueeze(0)


def _default_checkpoint_path() -> Path:
    return default_unet_checkpoint_path()


def _load_model(checkpoint_path: Path, device: torch.device) -> DeeperUNet:
    model = DeeperUNet()
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


_cached_model: DeeperUNet | None = None
_cached_device: torch.device | None = None
_infer_lock = threading.Lock()


def preload_unet_denoiser(
    checkpoint_path: Path | None = None,
    device: torch.device | None = None,
) -> None:
    global _cached_model, _cached_device
    with _infer_lock:
        if _cached_model is not None:
            return
        path = checkpoint_path if checkpoint_path is not None else _default_checkpoint_path()
        if not path.exists():
            raise FileNotFoundError(
                f"UNet checkpoint not found at {path}. "
                "Train the model first with unet-denoiser/train.ipynb"
            )
        dev = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        _cached_model = _load_model(path, dev)
        _cached_device = dev


def _denoise_with_model(
    model: DeeperUNet,
    image: np.ndarray,
    device: torch.device,
    patch_size: int,
    stride: int,
    use_patching: bool = False,
) -> np.ndarray:
    if image.dtype == np.uint8:
        img_float = image.astype(np.float32) / 255.0
    else:
        img_float = image.astype(np.float32)
        if img_float.max() > 1.0:
            img_float = img_float / 255.0

    img_tensor = T.functional.to_tensor(img_float).unsqueeze(0).to(device)
    _, _, h, w = img_tensor.shape

    if not use_patching:
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        padded_img = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode="reflect")
        with torch.no_grad():
            output_tensor = model(padded_img)
            output_tensor = torch.clamp(output_tensor, 0.0, 1.0)
        final_output = output_tensor[:, :, :h, :w]
        output_np = np.clip(final_output.squeeze().cpu().numpy(), 0, 1)
        return (output_np * 255).astype(np.uint8)

    pad_h_extra = (stride - h % stride) % stride
    pad_w_extra = (stride - w % stride) % stride
    pad_top = patch_size // 2
    pad_bottom = (patch_size // 2) + pad_h_extra
    pad_left = patch_size // 2
    pad_right = (patch_size // 2) + pad_w_extra

    padded_img = F.pad(img_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")
    _, _, padded_h, padded_w = padded_img.shape

    output_tensor = torch.zeros_like(padded_img)
    weight_accumulator = torch.zeros((1, 1, padded_h, padded_w), device=device)
    window = _create_window(patch_size, device)

    with torch.no_grad():
        for y in range(0, padded_h - patch_size + 1, stride):
            for x in range(0, padded_w - patch_size + 1, stride):
                patch = padded_img[:, :, y:y + patch_size, x:x + patch_size]
                pred_patch = model(patch)
                pred_patch = torch.clamp(pred_patch, 0.0, 1.0)
                output_tensor[:, :, y:y + patch_size, x:x + patch_size] += pred_patch * window
                weight_accumulator[:, :, y:y + patch_size, x:x + patch_size] += window

    output_tensor = output_tensor / (weight_accumulator + 1e-5)
    final_output = output_tensor[:, :, pad_top:pad_top + h, pad_left:pad_left + w]
    output_np = np.clip(final_output.squeeze().cpu().numpy(), 0, 1)
    return (output_np * 255).astype(np.uint8)


def denoise_image(
    image: np.ndarray,
    checkpoint_path: Path | None = None,
    patch_size: int = 512,
    stride: int = 256,
    device: torch.device | None = None,
) -> np.ndarray:
    with _infer_lock:
        if _cached_model is not None and _cached_device is not None:
            return _denoise_with_model(
                _cached_model, image, _cached_device, patch_size, stride
            )

    path = checkpoint_path if checkpoint_path is not None else _default_checkpoint_path()
    if not path.exists():
        raise FileNotFoundError(
            f"UNet checkpoint not found at {path}. "
            "Train the model first with unet-denoiser/train.ipynb"
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with _infer_lock:
        if _cached_model is not None and _cached_device is not None:
            return _denoise_with_model(
                _cached_model, image, _cached_device, patch_size, stride
            )
        model = _load_model(path, device)
        try:
            return _denoise_with_model(model, image, device, patch_size, stride)
        finally:
            del model
