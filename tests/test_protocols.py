"""Protocol regressions: coordinate frames, official metric parity and failure accounting."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from panorama.benchmark import evaluate, summarize
from panorama.features import Features
from panorama.protocols import (
    SUPERPOINT,
    UDIS,
    corner_error,
    nearest_matches,
    paper_case,
    resized_homography,
    udis_metrics,
)


def test_homography_uses_both_resize_scales_and_resized_error():
    # A point (100, 100) maps to (210, 210) before resizing.
    truth = np.array([[2, 0, 10], [0, 2, 10], [0, 0, 1]], float)
    scaled = resized_homography(truth, (1280, 960), (2560, 1920))
    np.testing.assert_allclose(scaled, [[1, 0, 2.5], [0, 1, 2.5], [0, 0, 1]])
    assert corner_error(np.eye(3), scaled, 640, 480) == pytest.approx(np.sqrt(12.5))


def test_paper_nearest_neighbors_do_not_apply_ratio_or_mutual_filter():
    a = Features(np.zeros((2, 2)), np.array([[0.0, 0.0], [0.1, 0.0]], np.float32))
    b = Features(np.zeros((2, 2)), np.array([[0.0, 0.0], [10.0, 0.0]], np.float32))
    assert nearest_matches(a, b).tolist() == [[0, 0], [1, 0]]


def test_udis_matches_original_scikit_image_015_reference():
    fixture = json.loads((Path(__file__).parent / "fixtures/udis_metrics.json").read_text())
    rng = np.random.default_rng(fixture["rng_seed"])
    a = rng.integers(0, 256, (33, 39, 3), dtype=np.uint8)
    b = np.roll(a, 3, axis=1)
    b[:, :, 1] //= 2
    h = np.array([[1, 0, -2.25], [0, 1, 1.5], [0, 0, 1]], float)
    actual = udis_metrics(a, b, h)
    for key, value in fixture["expected"].items():
        assert actual[key] == pytest.approx(value, abs=1e-12)


def test_udis_failed_estimate_uses_identity_and_stays_in_average(tmp_path, monkeypatch):
    for name, value in [("a.png", 70), ("b.png", 100)]:
        cv2.imwrite(str(tmp_path / name), np.full((512, 512, 3), value, np.uint8))

    def fail(*args, **kwargs):
        raise ValueError("forced estimation failure")

    monkeypatch.setattr("panorama.protocols.detect", fail)
    item = {"dataset": "UDIS-D", "id": "one", "a": "a.png", "b": "b.png"}
    failed = paper_case((item, "sift", 512, str(tmp_path), UDIS))
    assert failed["status"] == "failed" and failed["fallback"] == "identity"
    np.testing.assert_equal(failed["homography"], np.eye(3))
    assert failed["psnr_db"] == pytest.approx(20 * np.log10(255 / 30), abs=1e-5)
    good = dict(failed, case="two", status="ok", psnr_db=30.0, ssim=0.9)
    good.pop("fallback")
    row = summarize([failed, good])[0]
    assert row["pairs"] == 2 and row["succeeded"] == 1 and row["identity_fallbacks"] == 1
    assert row["psnr_db"] == pytest.approx((failed["psnr_db"] + 30) / 2)
    assert row["ssim"] == pytest.approx((failed["ssim"] + 0.9) / 2)


def test_hpatches_accepts_four_inliers_without_photometric_gate(tmp_path, monkeypatch):
    for name in ("a.png", "b.png"):
        cv2.imwrite(str(tmp_path / name), np.zeros((480, 640, 3), np.uint8))
    features = Features(
        np.array([[30, 30], [600, 30], [30, 440], [600, 440]], float), np.eye(4, dtype=np.float32)
    )
    monkeypatch.setattr("panorama.protocols.paper_features", lambda *args: features)
    item = {
        "dataset": "HPatches",
        "id": "one",
        "a": "a.png",
        "b": "b.png",
        "homography": np.eye(3).tolist(),
    }
    row = paper_case((item, "sift", 640, str(tmp_path), SUPERPOINT))
    assert row["status"] == "ok" and row["inliers"] == 4
    assert row["corner_error_px"] < 1e-8


def test_paper_protocol_rejects_custom_resize_and_missing_input(tmp_path):
    manifest = tmp_path / "manifest.json"
    item = {"dataset": "UDIS-D", "id": "one", "a": "missing.png", "b": "missing.png"}
    manifest.write_text(json.dumps({"pairs": [item]}))
    with pytest.raises(ValueError, match="固定输入尺寸"):
        evaluate(manifest, tmp_path / "out", max_size=800)
    with pytest.raises(OSError, match="无法读取"):
        paper_case((item, "sift", 512, str(tmp_path), UDIS))
