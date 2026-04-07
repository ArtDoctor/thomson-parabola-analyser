import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict

from oblisk.config import Settings
from oblisk.processing.image_ops import (
    find_brightest_spot,
    standardize_orientation,
    standardize_parity,
)
from oblisk.reporting.pipeline_log import DenoiseLog, DetectorCropLog
from oblisk.runtime import cut_detector_image, denoise_image as unet_denoise_image


def _shift_1d_interval(start: int, end: int, limit: int) -> tuple[int, int]:
    """Fit ``[start, end)`` into ``[0, limit)``, preserving span when it fits."""
    span = end - start
    if span <= limit:
        if start < 0:
            end -= start
            start = 0
        if end > limit:
            start -= end - limit
            end = limit
        return start, end
    return 0, limit


def _square_yolo_crop(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> tuple[np.ndarray, tuple[int, int, int, int], int]:
    """Crop YOLO box to a square of side ``max(width, height)``, padding if needed.

    The square is centered on the YOLO box; when shifted to stay inside the image,
    padding can be one-sided (e.g. box flush with an image edge).

    Returns ``(patch, (px1, py1, px2, py2), side)`` where ``patch`` is ``side``×``side``
    and the tuple is the source rectangle in the original image (before zero-padding).
    """
    img_h, img_w = int(image.shape[0]), int(image.shape[1])
    w_box = x2 - x1
    h_box = y2 - y1
    if w_box == h_box:
        patch = image[y1:y2, x1:x2]
        return patch, (x1, y1, x2, y2), w_box

    side = max(w_box, h_box)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    nx1 = int(round(cx - side / 2.0))
    ny1 = int(round(cy - side / 2.0))
    nx2 = nx1 + side
    ny2 = ny1 + side

    px1, px2 = _shift_1d_interval(nx1, nx2, img_w)
    py1, py2 = _shift_1d_interval(ny1, ny2, img_h)

    roi = image[py1:py2, px1:px2]
    got_w = px2 - px1
    got_h = py2 - py1
    pad_w = side - got_w
    pad_h = side - got_h
    if pad_w == 0 and pad_h == 0:
        return roi, (px1, py1, px2, py2), side

    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    padded = np.pad(
        roi,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0,
    )
    return padded, (px1, py1, px2, py2), side


def _apply_inner_margin_crop(image: np.ndarray, margin_px: int) -> np.ndarray:
    if margin_px <= 0:
        return image
    height = int(image.shape[0])
    width = int(image.shape[1])
    if height <= 2 * margin_px or width <= 2 * margin_px:
        return image
    return image[margin_px:height - margin_px, margin_px:width - margin_px]


class PreprocessedImage(BaseModel):
    """Result of GPU-dependent preprocessing (YOLO crop + UNet denoise)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_path: Path
    cropped: np.ndarray
    opened: np.ndarray
    brightest_spot: tuple[int, int]
    detector_config: str
    m_per_px_img: float
    orig_w: int
    orig_h: int
    denoise_title: str
    unet_resize_scale: float | None
    log_entries: list[BaseModel]
    timings: dict[str, float]


def preprocess_image(
    image_path: Path,
    settings: Settings,
    use_denoise_unet: bool = True,
) -> PreprocessedImage:
    """GPU-dependent stages: load, YOLO crop, UNet denoise."""

    log_entries: list[BaseModel] = []
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    test_img = np.array(Image.open(image_path))

    if np.max(test_img) > 255:
        test_img = test_img / 65535 * 255
    test_img = test_img.astype(np.uint8)
    if test_img.ndim == 3:
        if test_img.shape[2] == 3:
            test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
        elif test_img.shape[2] == 4:
            test_img = cv2.cvtColor(test_img, cv2.COLOR_RGBA2GRAY)

    coords = cut_detector_image(test_img)
    test_img, (sx1, sy1, sx2, sy2), square_side = _square_yolo_crop(
        test_img,
        coords.x1,
        coords.y1,
        coords.x2,
        coords.y2,
    )
    orig_w = square_side
    orig_h = square_side
    m_per_px_img = float(settings.fallback_meters_per_pixel)
    crop_diag = float(np.sqrt(orig_w**2 + orig_h**2))
    detector_config = "A" if crop_diag > 1800 else "B"
    log_entries.append(
        DetectorCropLog(
            x1=sx1,
            y1=sy1,
            x2=sx2,
            y2=sy2,
            yolo_confidence=coords.score,
        )
    )
    test_img = _apply_inner_margin_crop(test_img, settings.inner_margin_crop_px)
    test_img, orient_info = standardize_orientation(test_img)
    log_entries.append(orient_info)
    test_img, parity_info = standardize_parity(test_img)
    log_entries.append(parity_info)
    timings["load_crop"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    brightest_spot_raw = find_brightest_spot(test_img)
    brightest_spot = (
        int(brightest_spot_raw[0]),
        int(brightest_spot_raw[1]),
    )

    image = test_img
    unet_resize_scale: float | None = None
    if settings.denoise:
        if use_denoise_unet:
            height, width = image.shape[:2]
            max_dim = max(height, width)
            if max_dim > 2000:
                unet_resize_scale = 1200.0 / max_dim
                new_w = int(round(width * unet_resize_scale))
                new_h = int(round(height * unet_resize_scale))
                image_small = cv2.resize(
                    image,
                    (new_w, new_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                denoised_small = unet_denoise_image(image_small)
                opened = cv2.resize(
                    denoised_small,
                    (width, height),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                opened = unet_denoise_image(image)
            denoise_title = "UNet Denoised"
            log_entries.append(
                DenoiseLog(method="unet", unet_resize_scale=unet_resize_scale)
            )
        else:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (settings.denoise_kernel_size, settings.denoise_kernel_size),
            )
            opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            denoise_title = "Morphological Opening"
            log_entries.append(
                DenoiseLog(method="morph_open", unet_resize_scale=None)
            )
    else:
        opened = image
        denoise_title = "Original"
        log_entries.append(DenoiseLog(method="none", unet_resize_scale=None))
    timings["denoise"] = time.perf_counter() - t0

    return PreprocessedImage(
        image_path=image_path,
        cropped=test_img,
        opened=opened,
        brightest_spot=brightest_spot,
        detector_config=detector_config,
        m_per_px_img=m_per_px_img,
        orig_w=orig_w,
        orig_h=orig_h,
        denoise_title=denoise_title,
        unet_resize_scale=unet_resize_scale,
        log_entries=log_entries,
        timings=timings,
    )
