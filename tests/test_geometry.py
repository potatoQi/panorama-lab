import numpy as np
import pytest

from panorama.geometry import dlt, project, ransac


def test_minimal_dlt_with_large_pixel_coordinates():
    source = np.array([[1000, 2000], [3000, 1900], [3200, 4000], [900, 4200.0]])
    h = np.array([[1.1, 0.1, 80], [-0.04, 0.9, -30], [1e-5, -2e-5, 1.0]])
    destination = project(h, source)
    np.testing.assert_allclose(project(dlt(source, destination), source), destination, atol=1e-7)


def test_ransac_with_outliers_and_noise():
    rng = np.random.default_rng(10)
    source = rng.uniform(0, 900, (140, 2))
    h = np.array([[0.95, 0.12, 120], [-0.1, 1.02, 30], [0.0001, -0.00008, 1.0]])
    destination = project(h, source) + rng.normal(0, 0.25, source.shape)
    destination[:50] = rng.uniform(0, 900, (50, 2))
    fit = ransac(source, destination, seed=22)
    assert fit.inliers[50:].sum() >= 87
    assert fit.inliers[:50].sum() <= 2
    assert (
        np.median(
            np.linalg.norm(project(fit.homography, source[50:]) - project(h, source[50:]), axis=1)
        )
        < 0.3
    )
    np.testing.assert_allclose(fit.homography, ransac(source, destination, seed=22).homography)


def test_collinear_points_rejected():
    p = np.column_stack((np.arange(6), np.arange(6)))
    with pytest.raises(ValueError):
        dlt(p, p + 2)


def test_too_few_correspondences():
    with pytest.raises(ValueError):
        ransac(np.zeros((3, 2)), np.zeros((3, 2)))
