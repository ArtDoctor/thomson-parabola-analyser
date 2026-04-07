import cv2
import numpy as np

from oblisk.reporting.pipeline_log import (
    OrientationDecision,
    ParityDecision,
    QuadrantPixelCounts,
)


def find_brightest_spot(image: np.ndarray) -> tuple[int, ...]:
    image_array = np.array(image)
    brightest_index = np.unravel_index(np.argmax(image_array), image_array.shape)
    return tuple(int(value) for value in brightest_index)


def find_brightest_spot_robust(
    image: np.ndarray,
    blur_size: int = 7,
) -> tuple[int, int]:
    """Reduce sensitivity to isolated hot pixels via a blurred search image."""

    blurred = cv2.GaussianBlur(image.astype(np.float32), (blur_size, blur_size), 0)
    brightest_index = np.unravel_index(np.argmax(blurred), blurred.shape)
    return (int(brightest_index[0]), int(brightest_index[1]))


def standardize_orientation(
    image: np.ndarray,
) -> tuple[np.ndarray, OrientationDecision]:
    """Rotate so the dominant signal fan lands in the top-right quadrant."""

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, _, _, max_loc = cv2.minMaxLoc(blurred)
    ox, oy = int(max_loc[0]), int(max_loc[1])

    masked = image.copy()
    cv2.circle(masked, (ox, oy), radius=60, color=(0, 0, 0), thickness=-1)

    non_zero = masked[masked > 0]
    if len(non_zero) == 0:
        empty_counts = QuadrantPixelCounts(TR=0, TL=0, BL=0, BR=0)
        decision = OrientationDecision(
            origin_xy=(ox, oy),
            quadrant_counts=empty_counts,
            dominant_quadrant="?",
            rotation_applied="skipped_empty_signal",
        )
        return image, decision

    thresh_val = np.percentile(non_zero, 75)
    _, thresh = cv2.threshold(masked, thresh_val, 255, cv2.THRESH_BINARY)

    y_idx, x_idx = np.nonzero(thresh)

    q_tr = int(np.sum((x_idx > ox) & (y_idx < oy)))
    q_tl = int(np.sum((x_idx < ox) & (y_idx < oy)))
    q_bl = int(np.sum((x_idx < ox) & (y_idx > oy)))
    q_br = int(np.sum((x_idx > ox) & (y_idx > oy)))

    quadrants = {
        "TR": q_tr,
        "TL": q_tl,
        "BL": q_bl,
        "BR": q_br,
    }

    max_quad = max(quadrants, key=lambda key: quadrants[key])
    counts = QuadrantPixelCounts(TR=q_tr, TL=q_tl, BL=q_bl, BR=q_br)

    if max_quad == "TR":
        decision = OrientationDecision(
            origin_xy=(ox, oy),
            quadrant_counts=counts,
            dominant_quadrant=max_quad,
            rotation_applied="none",
        )
        return image, decision
    if max_quad == "BR":
        decision = OrientationDecision(
            origin_xy=(ox, oy),
            quadrant_counts=counts,
            dominant_quadrant=max_quad,
            rotation_applied="rotate_90_ccw",
        )
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), decision
    if max_quad == "BL":
        decision = OrientationDecision(
            origin_xy=(ox, oy),
            quadrant_counts=counts,
            dominant_quadrant=max_quad,
            rotation_applied="rotate_180",
        )
        return cv2.rotate(image, cv2.ROTATE_180), decision
    if max_quad == "TL":
        decision = OrientationDecision(
            origin_xy=(ox, oy),
            quadrant_counts=counts,
            dominant_quadrant=max_quad,
            rotation_applied="rotate_90_cw",
        )
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), decision

    decision = OrientationDecision(
        origin_xy=(ox, oy),
        quadrant_counts=counts,
        dominant_quadrant=max_quad,
        rotation_applied="none",
    )
    return image, decision


def standardize_parity(image: np.ndarray) -> tuple[np.ndarray, ParityDecision]:
    """Normalize left/right fan parity after orientation has been standardized."""

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, _, _, max_loc = cv2.minMaxLoc(blurred)
    ox, oy = max_loc

    masked = image.copy()
    cv2.circle(masked, (ox, oy), radius=60, color=(0, 0, 0), thickness=-1)

    non_zero = masked[masked > 0]
    if len(non_zero) == 0:
        parity = ParityDecision(
            pixels_high_angle=0,
            pixels_low_angle=0,
            mirrored_fan_detected=False,
            horizontal_flip_applied=False,
            reorientation_after_horizontal_flip=None,
            final_vertical_flip_applied=False,
        )
        return image, parity

    thresh_val = np.percentile(non_zero, 75)
    _, thresh = cv2.threshold(masked, thresh_val, 255, cv2.THRESH_BINARY)

    y_idx, x_idx = np.nonzero(thresh)

    dy = oy - y_idx
    dx = x_idx - ox

    valid_mask = (dx > 0) & (dy > 0)
    dx = dx[valid_mask]
    dy = dy[valid_mask]

    if len(dx) == 0:
        parity = ParityDecision(
            pixels_high_angle=0,
            pixels_low_angle=0,
            mirrored_fan_detected=False,
            horizontal_flip_applied=False,
            reorientation_after_horizontal_flip=None,
            final_vertical_flip_applied=False,
        )
        return image, parity

    angles = np.arctan2(dy, dx)

    min_angle = np.percentile(angles, 5)
    max_angle = np.percentile(angles, 95)
    mid_angle = (min_angle + max_angle) / 2.0

    pixels_high_angle = int(np.sum(angles > mid_angle))
    pixels_low_angle = int(np.sum(angles < mid_angle))

    mirrored = pixels_high_angle > pixels_low_angle
    reorient_after: OrientationDecision | None = None
    if mirrored:
        flipped = cv2.flip(image, 1)
        result, reorient_after = standardize_orientation(flipped)
    else:
        result = image

    out = cv2.flip(result, 0)
    parity = ParityDecision(
        pixels_high_angle=pixels_high_angle,
        pixels_low_angle=pixels_low_angle,
        mirrored_fan_detected=mirrored,
        horizontal_flip_applied=mirrored,
        reorientation_after_horizontal_flip=reorient_after,
        final_vertical_flip_applied=True,
    )
    return out, parity
