import numpy as np

from panorama.render import planar_canvas, render


def test_black_pixels_are_valid_and_blending_preserves_constant_image():
    images = [np.full((80, 100, 3), 120, np.uint8), np.full((80, 100, 3), 120, np.uint8)]
    images[0][20:30, 20:30] = 0
    transforms = [np.eye(3), np.array([[1.0, 0, 50], [0, 1, 0], [0, 0, 1]])]
    canvas = planar_canvas(images, transforms, 100000)
    output, valid, coverage = render(images, transforms, canvas)
    assert valid[25, 25]
    assert output[25, 25].max() < 15
    assert coverage[40, 75] == 2
    assert np.max(np.abs(output[40, 90].astype(int) - 120)) < 2
