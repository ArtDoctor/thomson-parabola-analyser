from pathlib import Path
from typing import Any

from oblisk.config import Settings
from oblisk.runtime import Coords

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DETECTOR_IMAGE = REPO_ROOT / "synthetic_data" / "detector_image.png"


def synthetic_settings() -> Settings:
    return Settings()


def full_frame_coords(image: Any) -> Coords:
    height = int(image.shape[0])
    width = int(image.shape[1])
    return Coords(x1=0, y1=0, x2=width, y2=height, score=1.0)
