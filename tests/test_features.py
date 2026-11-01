import cv2
import numpy as np

from panorama.features import Features, detect, match
from panorama.geometry import project, ransac


def test_ratio_and_mutual_filtering():
    a = Features(np.zeros((4, 2)), np.array([[0, 0], [10, 10], [20, 20], [0.01, 0]], np.float32))
    b = Features(np.zeros((3, 2)), np.array([[0, 0], [10, 10], [20, 20]], np.float32))
    pairs, _ = match(a, b)
    assert set(map(tuple, pairs)) == {(0, 0), (1, 1), (2, 2)}


def test_handwritten_features_recover_translation():
    rng = np.random.default_rng(2)
    im = np.zeros((300, 400, 3), np.uint8)
    for _ in range(180):
        x, y = rng.integers([25, 25], [370, 270])
        cv2.circle(
            im,
            (int(x), int(y)),
            int(rng.integers(2, 9)),
            tuple(map(int, rng.integers(60, 255, 3))),
            -1,
        )
    moved = cv2.warpAffine(im, np.float32([[1, 0, 15], [0, 1, 9]]), (400, 300))
    a, b = detect(im, "harris", 600), detect(moved, "harris", 600)
    pairs, _ = match(a, b)
    assert len(pairs) > 20
    estimate = ransac(a.points[pairs[:, 0]], b.points[pairs[:, 1]])
    np.testing.assert_allclose(
        project(estimate.homography, np.array([[200, 150]])), [[215, 159]], atol=1
    )


def test_binary_matching_is_exact_hamming():
    rng = np.random.default_rng(31)
    descriptors = rng.integers(0, 256, (70, 32), dtype=np.uint8)
    changed = descriptors.copy()
    changed[:, 0] ^= 3
    a = Features(np.zeros((70, 2)), descriptors, True)
    b = Features(np.zeros((70, 2)), changed, True)
    indices, quality = match(a, b)
    assert set(map(tuple, indices)) == {(i, i) for i in range(70)}
    distances = np.unpackbits(descriptors[:, None] ^ changed[None, :], axis=2).sum(axis=2)
    expected = 2 / np.sort(distances, axis=1)[:, 1]
    np.testing.assert_allclose(quality, expected[indices[:, 0]])


def test_plain_sift_and_rootsift_share_points_but_not_descriptors():
    from panorama.protocols import paper_features

    rng = np.random.default_rng(19)
    image = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    for extractor in (detect, paper_features):
        plain = extractor(image, "sift-plain")
        root = extractor(image, "sift")
        assert len(plain.points) > 20
        np.testing.assert_array_equal(plain.points, root.points)
        assert np.linalg.norm(plain.descriptors[0]) > 100
        np.testing.assert_allclose(
            root.descriptors,
            np.sqrt(plain.descriptors / (plain.descriptors.sum(1, keepdims=True) + 1e-12)),
        )


def test_xfeat_sparse_translation_and_budget():
    from pathlib import Path

    import pytest

    pytest.importorskip("torch")
    from panorama.xfeat import CACHE

    if not (Path(CACHE) / "xfeat/xfeat.pt").exists():
        pytest.skip("Run python -m panorama.xfeat to enable pretrained integration check")
    image = np.random.default_rng(28).integers(0, 256, (241, 337, 3), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    moved = cv2.warpAffine(image, np.float32([[1, 0, 32], [0, 1, 32]]), (337, 241))
    a, b = detect(image, "xfeat", 300), detect(moved, "xfeat", 300)
    assert 10 < len(a.points) <= 300
    assert a.descriptors.shape == (len(a.points), 64)
    assert (a.points >= 0).all() and (a.points < [337, 241]).all()
    np.testing.assert_allclose(np.linalg.norm(a.descriptors, axis=1), 1, atol=1e-5)
    pairs, _ = match(a, b)
    assert len(pairs) >= 8
    estimate = ransac(a.points[pairs[:, 0]], b.points[pairs[:, 1]])
    np.testing.assert_allclose(project(estimate.homography, [[160, 120]]), [[192, 152]], atol=2)
