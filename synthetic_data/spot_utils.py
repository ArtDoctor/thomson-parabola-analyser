import random
from typing import Any

import cv2
import numpy as np


def create_2d_gaussian(
    size: tuple[int, int],
    center: tuple[float, float],
    sigma_x: float | tuple[float, float],
    sigma_y: float,
    amplitude: float,
    angle: float = 0.0,
    one_sided: bool = False,
    taper: bool = False,
) -> np.ndarray:
    x = np.arange(0, size[1], 1, float)
    y = np.arange(0, size[0], 1, float)
    x, y = np.meshgrid(x, y)
    
    x0, y0 = center
    
    theta = np.radians(angle)
    x_rot = (x - x0) * np.cos(theta) - (y - y0) * np.sin(theta)
    y_rot = (x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)
    
    if isinstance(sigma_x, tuple):
        sx = np.where(x_rot > 0, sigma_x[0], sigma_x[1])
    else:
        sx = np.full_like(x_rot, sigma_x, dtype=np.float64)

    if taper:
        taper_factor = np.clip(1.0 - (x_rot / (2.0 * sx)), 0.1, 1.0)
        sigma_y_eff = sigma_y * taper_factor
    else:
        sigma_y_eff = sigma_y

    g = amplitude * np.exp(-((x_rot**2) / (2 * sx**2) + (y_rot**2) / (2 * sigma_y_eff**2)))
    
    if one_sided:
        g[x_rot < 0] = 0
        
    return g


def generate_grain_mask(image_size: tuple[int, int]) -> np.ndarray:
    h, w = image_size
    grain = np.random.uniform(0.4, 1.3, (h, w)).astype(np.float32)
    dark_speckles = np.random.random((h, w))
    grain[dark_speckles < 0.25] *= np.random.uniform(0.0, 0.4) 
    clumps = np.random.uniform(0.8, 1.2, (h // 4, w // 4)).astype(np.float32)
    clumps = cv2.resize(clumps, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32, copy=False)
    return grain * clumps


def generate_synthetic_object(
    image_size: tuple[int, int] = (466, 466),
    center: tuple[int, int] = (233, 233),
    core_sigma: float = 4.5,
    core_amplitude: float = 255,
    coma_angle: float = 0.0,
    coma_amplitude: float = 160,
    lines_params: list[dict[str, float]] | None = None,
    ghost_params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = np.zeros(image_size, dtype=np.float32)
    
    # Spatial Mask for noise scaling
    yy, xx = np.mgrid[0:image_size[0], 0:image_size[1]]
    dist_from_center = np.sqrt((xx - center[0])**2 + (yy - center[1])**2)
    noise_strength = np.clip((dist_from_center - 5) / 60.0, 0.0, 1.0) 
    
    # 0. Background Blob
    bg_blob = create_2d_gaussian(image_size, center, 190, 190, 45, coma_angle)
    img += bg_blob

    # 1. The Core (Generated for both the main image and a separate return)
    core = create_2d_gaussian(image_size, center, core_sigma, core_sigma, core_amplitude)
    img += core
    
    # Core-only image (clipped to 8-bit)
    core_img = np.clip(core, 0, 255).astype(np.uint8)

    # 2. The Coma
    coma = create_2d_gaussian(image_size, center, (55, 30), 48, coma_amplitude, coma_angle)
    raw_coma_grain = generate_grain_mask(image_size)
    blended_coma_grain = 1.0 + (raw_coma_grain - 1.0) * noise_strength
    coma *= blended_coma_grain
    img += coma

    # Clean version: everything so far (blob, core, coma) — no radiating lines
    img_clean = img.copy()

    # 3. Radiating Lines (noisy only; clean keeps blob/core/coma/ghost but not lines)
    if lines_params:
        for params in lines_params:
            line = create_2d_gaussian(
                image_size, center, params['length_sigma'], params['width_sigma'],
                params['intensity'], params['angle'], one_sided=True, taper=True
            )
            raw_line_grain = generate_grain_mask(image_size)
            blended_line_grain = 1.0 + (raw_line_grain - 1.0) * noise_strength
            line *= blended_line_grain
            img += line

    # 4. Ghost Circle
    if ghost_params:
        ghost = create_2d_gaussian(image_size, ghost_params['center'], 6, 6, ghost_params['amplitude'])
        img += ghost
        img_clean += ghost

    # 5. Global Sensor Grain
    global_noise = np.random.normal(0, 4, image_size).astype(np.float32)
    img += global_noise
    img_clean += global_noise

    img = np.clip(img, 0, 255).astype(np.uint8)
    img_clean = np.clip(img_clean, 0, 255).astype(np.uint8)

    return img, img_clean, core_img


def create_randomized_instance() -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    image_size = (466, 466)
    center = (image_size[1] // 2 + random.randint(-26, 26), image_size[0] // 2 + random.randint(-26, 26))
    core_sigma = random.uniform(3.0, 5.0) 
    coma_angle = random.uniform(0, 360)
    coma_amplitude = random.uniform(100, 180) 
    
    num_lines = random.randint(7, 15)
    lines_params = [{'angle': random.uniform(0, 360), 'length_sigma': random.uniform(100, 400), 
                     'width_sigma': random.uniform(1.5, 8.0), 'intensity': random.uniform(5, 20)} 
                    for _ in range(num_lines)]
        
    ghost_params = None
    if random.random() < 0.2:
        ghost_dist = random.uniform(30, 80) 
        ghost_angle = random.uniform(0, 360)
        ghost_params = {
            'center': (center[0] + ghost_dist * np.cos(np.radians(ghost_angle)),
                       center[1] + ghost_dist * np.sin(np.radians(ghost_angle))),
            'amplitude': random.uniform(30, 80)
        }

    img, img_clean, _ = generate_synthetic_object(
        image_size=image_size, center=center, core_sigma=core_sigma,
        coma_angle=coma_angle, coma_amplitude=coma_amplitude,
        lines_params=lines_params, ghost_params=ghost_params
    )
    return img, img_clean, center
