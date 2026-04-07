import numpy as np

from oblisk.processing.preprocessing import _shift_1d_interval, _square_yolo_crop


def test_shift_1d_interval_negative_start() -> None:
    assert _shift_1d_interval(-10, 90, 100) == (0, 100)


def test_shift_1d_interval_past_limit() -> None:
    assert _shift_1d_interval(30, 130, 100) == (0, 100)


def test_shift_1d_interval_span_exceeds_limit() -> None:
    assert _shift_1d_interval(-10, 120, 100) == (0, 100)


def test_square_crop_already_square() -> None:
    img = np.arange(25, dtype=np.uint8).reshape(5, 5)
    patch, rect, side = _square_yolo_crop(img, 1, 1, 4, 4)
    assert patch.shape == (3, 3)
    assert rect == (1, 1, 4, 4)
    assert side == 3


def test_square_crop_wide_band_bottom_pads_upward_in_image() -> None:
    img = np.ones((10, 10), dtype=np.uint8) * 7
    y1, y2 = 7, 10
    x1, x2 = 0, 10
    patch, rect, side = _square_yolo_crop(img, x1, y1, x2, y2)
    assert side == 10
    assert patch.shape == (10, 10)
    assert rect == (0, 0, 10, 10)
    assert np.all(patch[7:10, :] == 7)


def test_square_crop_needs_zero_pad_when_side_exceeds_image() -> None:
    img = np.ones((8, 6), dtype=np.uint8) * 9
    patch, rect, side = _square_yolo_crop(img, 0, 0, 6, 8)
    assert side == 8
    assert patch.shape == (8, 8)
    assert rect == (0, 0, 6, 8)
    assert int(np.sum(patch == 9)) == 6 * 8
    assert int(np.sum(patch == 0)) == 8 * 8 - 6 * 8
