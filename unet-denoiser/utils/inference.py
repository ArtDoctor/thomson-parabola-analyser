import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from .model import DeeperUNet


def create_window(window_size: int, device: torch.device) -> torch.Tensor:
    window_1d = torch.hann_window(window_size)
    window_2d = window_1d.unsqueeze(1) * window_1d.unsqueeze(0)
    return window_2d.unsqueeze(0).unsqueeze(0).to(device)


def denoise_large_image(
    image_path: str,
    model: DeeperUNet,
    device: torch.device,
    patch_size: int = 512,
    stride: int = 256,
    use_patching: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    img = Image.open(image_path).convert("L")
    img = img.transpose(Image.FLIP_LEFT_RIGHT).rotate(90)
    img_tensor = T.functional.to_tensor(img).unsqueeze(0).to(device)

    _, channels, h, w = img_tensor.shape

    # For 1200x1200 images, passing the full image is much faster
    if not use_patching:
        # UNet has 4 pooling layers, so sides must be divisible by 16 (2^4 = 16)
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        
        # Pad with reflect mode to avoid boundary artifacts
        padded_img = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode="reflect")
        
        with torch.no_grad():
            output_tensor = model(padded_img)
            output_tensor = torch.clamp(output_tensor, 0.0, 1.0)
            
        final_output = output_tensor[:, :, :h, :w]
        
        input_np = img_tensor.squeeze().cpu().numpy()
        output_np = final_output.squeeze().cpu().numpy()
        
        return input_np, np.clip(output_np, 0, 1)

    # Patch-based inference keeps memory bounded for very large images.
    pad_h_extra = (stride - h % stride) % stride
    pad_w_extra = (stride - w % stride) % stride

    pad_top = patch_size // 2
    pad_bottom = (patch_size // 2) + pad_h_extra
    pad_left = patch_size // 2
    pad_right = (patch_size // 2) + pad_w_extra

    padded_img = F.pad(
        img_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect"
    )
    _, _, padded_h, padded_w = padded_img.shape

    output_tensor = torch.zeros_like(padded_img)
    weight_accumulator = torch.zeros((1, 1, padded_h, padded_w), device=device)

    window = create_window(patch_size, device)

    with torch.no_grad():
        for y in range(0, padded_h - patch_size + 1, stride):
            for x in range(0, padded_w - patch_size + 1, stride):
                patch = padded_img[:, :, y:y + patch_size, x:x + patch_size]
                pred_patch = model(patch)
                pred_patch = torch.clamp(pred_patch, 0.0, 1.0)
                output_tensor[:, :, y:y + patch_size, x:x + patch_size] += (
                    pred_patch * window
                )
                weight_accumulator[:, :, y:y + patch_size, x:x + patch_size] += (
                    window
                )

    output_tensor = output_tensor / (weight_accumulator + 1e-5)
    final_output = output_tensor[:, :, pad_top:pad_top + h, pad_left:pad_left + w]

    input_np = img_tensor.squeeze().cpu().numpy()
    output_np = final_output.squeeze().cpu().numpy()

    return input_np, np.clip(output_np, 0, 1)
