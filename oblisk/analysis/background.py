import numpy as np
from pydantic import BaseModel


class BackgroundRoi(BaseModel):
    x0: int = 200
    x1: int = 1000
    y0: int = 1000
    y1: int = 2000

    def clipped(self, width: int, height: int) -> "BackgroundRoi":
        return BackgroundRoi(
            x0=max(0, min(self.x0, width)),
            x1=max(0, min(self.x1, width)),
            y0=max(0, min(self.y0, height)),
            y1=max(0, min(self.y1, height)),
        )


def compute_background_mean(image: np.ndarray, roi: BackgroundRoi) -> tuple[BackgroundRoi, float]:
    image_float = image.astype(float)
    height, width = image_float.shape[:2]
    clipped_roi = roi.clipped(width=width, height=height)

    if clipped_roi.x0 >= clipped_roi.x1 or clipped_roi.y0 >= clipped_roi.y1:
        raise ValueError("Background ROI is empty after clipping to image bounds.")

    background_mean = float(
        image_float[clipped_roi.y0:clipped_roi.y1, clipped_roi.x0:clipped_roi.x1].mean()
    )
    return clipped_roi, background_mean
