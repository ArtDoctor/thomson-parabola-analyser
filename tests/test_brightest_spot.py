import numpy as np

from oblisk.processing.image_ops import find_brightest_spot


def test_brightest_spot_synthetic() -> None:
    test_img = np.zeros((100, 100))
    test_img[50, 50] = 1
    brightest_spot = find_brightest_spot(test_img)
    assert brightest_spot == (50, 50)
