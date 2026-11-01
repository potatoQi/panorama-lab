import numpy as np

from panorama.benchmark import overlap_metrics, summarize


def test_failure_counts_in_geometric_accuracy_denominator():
    base = {"dataset": "test", "method": "sift", "has_ground_truth": True, "seconds": 1}
    good = dict(
        base,
        status="ok",
        corner_error_px=2.0,
        inlier_ratio=0.8,
        inlier_rmse_px=1.0,
        overlap_psnr_db=30.0,
        overlap_ssim=0.9,
        overlap_fraction=0.5,
    )
    bad = dict(base, status="failed")
    row = summarize([good, bad])[0]
    assert row["success_rate"] == 0.5
    assert row["h_accuracy_3px"] == 0.5
    assert row["h_accuracy_1px"] == 0
    assert row["h_accuracy_5px"] == 0.5
    assert row["overlap_psnr_db"] == 30


def test_photometric_mask_does_not_remove_black_scene_content():
    image = np.zeros((64, 80, 3), np.uint8)
    image[20:40, 20:40] = 200
    scores = overlap_metrics(image, image.copy(), np.eye(3))
    assert scores["overlap_ssim"] > 0.999
    assert scores["overlap_fraction"] > 0.7


def test_all_failed_legacy_run_preserves_metric_schema():
    row = summarize(
        [
            {
                "dataset": "APAP",
                "method": "orb",
                "has_ground_truth": False,
                "seconds": 1,
                "status": "failed",
            }
        ]
    )[0]
    assert row["succeeded"] == 0
    assert row["overlap_psnr_db"] is None and row["overlap_ssim"] is None
