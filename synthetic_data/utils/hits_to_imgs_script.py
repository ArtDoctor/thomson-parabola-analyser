from pathlib import Path
import sys

from PIL import Image
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from synthetic_data.utils.hits_to_img import detector_limits_from_spot, hits_to_img
from synthetic_data.utils.noise import (
    apply_white_spots,
    apply_black_spots,
    apply_vertical_lines,
    apply_perlin_darkening,
    apply_perlin_whitening,
    apply_general_noise
)

ROOT_DIR = Path(__file__).resolve().parents[1]
HITS_PATH = ROOT_DIR / "hits.txt"
DETECTOR_IMAGE_PATH = ROOT_DIR / "detector_image.png"
NOISY_IMAGE_PATH = ROOT_DIR / "noisy_detector_image.png"

# Same corner framing as generate_synthetic_dataset (not default symmetric limits).
DEFAULT_SPOT_COL = 400
DEFAULT_SPOT_ROW = 1100
lim_x, lim_y = detector_limits_from_spot(DEFAULT_SPOT_COL, DEFAULT_SPOT_ROW)
img = hits_to_img(str(HITS_PATH), lim_x=lim_x, lim_y=lim_y)
img.save(DETECTOR_IMAGE_PATH)
img_array = np.array(img, dtype=np.float32)
# Apply noise sequentially
noisy_array = apply_white_spots(img_array)
noisy_array = apply_black_spots(noisy_array)
noisy_array = apply_vertical_lines(noisy_array)
noisy_array = apply_perlin_darkening(noisy_array)
noisy_array = apply_perlin_whitening(noisy_array)
noisy_array = apply_general_noise(noisy_array)
# Clip values to valid 8-bit range and cast
final_array = np.clip(noisy_array, 0, 255).astype(np.uint8)

# Save
out_img = Image.fromarray(final_array, mode='L')
out_img.save(NOISY_IMAGE_PATH)
