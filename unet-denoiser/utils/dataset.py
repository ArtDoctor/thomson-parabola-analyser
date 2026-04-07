import os
import glob
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


def _apply_affine_shear_pair(
    noisy_tensor: torch.Tensor,
    clean_tensor: torch.Tensor,
    shear_x_deg: float,
    shear_y_deg: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # torchvision: [shear parallel to x-axis, shear parallel to y-axis], degrees
    shear = [float(shear_x_deg), float(shear_y_deg)]
    noisy_out = TF.affine(
        noisy_tensor,
        angle=0.0,
        translate=[0, 0],
        scale=1.0,
        shear=shear,
        interpolation=TF.InterpolationMode.BILINEAR,
        fill=[0.0],
    )
    clean_out = TF.affine(
        clean_tensor,
        angle=0.0,
        translate=[0, 0],
        scale=1.0,
        shear=shear,
        interpolation=TF.InterpolationMode.BILINEAR,
        fill=[0.0],
    )
    return noisy_out, clean_out


def _column_vertical_scale_maps(h: int, w: int, strength: float) -> tuple[np.ndarray, np.ndarray]:
    w_eff = float(max(w - 1, 1))
    x_coords = np.arange(w, dtype=np.float32)
    s = 1.0 + strength * (2.0 * x_coords / w_eff - 1.0)
    s = np.maximum(s, 0.25)
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )
    cy = (h - 1) / 2.0
    map_x = xx.astype(np.float32)
    map_y = ((yy - cy) / s[np.newaxis, :] + cy).astype(np.float32)
    return map_x, map_y


def _remap_1ch_tensor(tensor_1hw: torch.Tensor, map_x: np.ndarray, map_y: np.ndarray) -> torch.Tensor:
    ch = tensor_1hw.squeeze(0).detach().cpu().numpy().astype(np.float32)
    out = cv2.remap(
        ch,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    out = np.clip(out, 0.0, 1.0)
    return torch.from_numpy(out).unsqueeze(0).to(dtype=tensor_1hw.dtype, device=tensor_1hw.device)


def _apply_column_vertical_scale_pair(
    noisy_tensor: torch.Tensor,
    clean_tensor: torch.Tensor,
    strength: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, h, w = noisy_tensor.shape
    map_x, map_y = _column_vertical_scale_maps(h, w, strength)
    noisy_r = _remap_1ch_tensor(noisy_tensor, map_x, map_y)
    clean_r = _remap_1ch_tensor(clean_tensor, map_x, map_y)
    return noisy_r, clean_r


class DenoisingDataset(Dataset):
    def __init__(
        self,
        clean_dir: str,
        noisy_dir: str,
        crop_size: int = 512,
        scale_min: float = 0.9,
        scale_max: float = 1.1,
        aug_shear_prob: float = 0.5,
        aug_shear_x_deg_max: float = 6.0,
        aug_shear_y_deg_max: float = 6.0,
        aug_column_scale_prob: float = 0.5,
        aug_column_scale_strength_max: float = 0.04,
    ) -> None:
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.noisy_paths = glob.glob(os.path.join(noisy_dir, "*.png"))
        self.crop_size = crop_size
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.aug_shear_prob = aug_shear_prob
        self.aug_shear_x_deg_max = aug_shear_x_deg_max
        self.aug_shear_y_deg_max = aug_shear_y_deg_max
        self.aug_column_scale_prob = aug_column_scale_prob
        self.aug_column_scale_strength_max = aug_column_scale_strength_max

    def __len__(self) -> int:
        return len(self.noisy_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        noisy_path = self.noisy_paths[idx]

        basename = os.path.basename(noisy_path)
        clean_name = basename.split("_noise_")[0] + ".png"
        clean_path = os.path.join(self.clean_dir, clean_name)

        noisy_img = Image.open(noisy_path).convert("L")
        clean_img = Image.open(clean_path).convert("L")

        min_hw = min(noisy_img.height, noisy_img.width)
        effective_min = max(self.scale_min, self.crop_size / min_hw)
        scale = random.uniform(effective_min, self.scale_max)

        new_h = max(int(noisy_img.height * scale), self.crop_size)
        new_w = max(int(noisy_img.width * scale), self.crop_size)

        noisy_img = noisy_img.resize((new_w, new_h), Image.BILINEAR)
        clean_img = clean_img.resize((new_w, new_h), Image.BILINEAR)

        i, j, h, w = T.RandomCrop.get_params(
            noisy_img, output_size=(self.crop_size, self.crop_size)
        )

        noisy_img = T.functional.crop(noisy_img, i, j, h, w)
        clean_img = T.functional.crop(clean_img, i, j, h, w)

        noisy_tensor = T.functional.to_tensor(noisy_img)
        clean_tensor = T.functional.to_tensor(clean_img)

        # Random augmentation: flips and 90° rotation (identical on noisy and clean)
        if random.random() > 0.5:
            noisy_tensor = TF.hflip(noisy_tensor)
            clean_tensor = TF.hflip(clean_tensor)
        if random.random() > 0.5:
            noisy_tensor = TF.vflip(noisy_tensor)
            clean_tensor = TF.vflip(clean_tensor)
        num_rot90 = random.randint(0, 3)
        if num_rot90 > 0:
            noisy_tensor = torch.rot90(noisy_tensor, num_rot90, [1, 2])
            clean_tensor = torch.rot90(clean_tensor, num_rot90, [1, 2])

        # 2D affine shear + x-dependent vertical scale (same parameters on noisy and clean)
        if self.aug_shear_prob > 0 and random.random() < self.aug_shear_prob:
            shear_x = random.uniform(-self.aug_shear_x_deg_max, self.aug_shear_x_deg_max)
            shear_y = random.uniform(-self.aug_shear_y_deg_max, self.aug_shear_y_deg_max)
            if abs(shear_x) > 1e-6 or abs(shear_y) > 1e-6:
                noisy_tensor, clean_tensor = _apply_affine_shear_pair(
                    noisy_tensor, clean_tensor, shear_x, shear_y
                )
        if self.aug_column_scale_prob > 0 and random.random() < self.aug_column_scale_prob:
            strength = random.uniform(
                -self.aug_column_scale_strength_max,
                self.aug_column_scale_strength_max,
            )
            if abs(strength) > 1e-6:
                noisy_tensor, clean_tensor = _apply_column_vertical_scale_pair(
                    noisy_tensor, clean_tensor, strength
                )

        noisy_tensor = torch.clamp(noisy_tensor, 0.0, 1.0)
        clean_tensor = torch.clamp(clean_tensor, 0.0, 1.0)

        return noisy_tensor, clean_tensor
