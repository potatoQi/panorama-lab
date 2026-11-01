"""Calibrate UDIS-D's complete identity baseline against UDIS++ Table 1."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from panorama.io import digest, write_json
from panorama.protocols import UDIS_COMMIT, read_cv, udis_metrics

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = ROOT / "data/udis-d.json"
    spec = json.loads(manifest.read_text())
    cases = spec["pairs"]
    if len(cases) != 1106 or len({p["id"] for p in cases}) != 1106:
        raise ValueError("校准必须使用完整 1106 对测试图像。")
    base = manifest.parent / spec["root"]
    cv2.setNumThreads(1)

    def score(item):
        a, b = read_cv(base / item["a"]), read_cv(base / item["b"])
        return dict(case=item["id"], **udis_metrics(a, b, np.eye(3)))

    with ThreadPoolExecutor(4) as pool:
        records = list(pool.map(score, cases))
    scores = {k: float(np.mean([r[k] for r in records])) for k in ("psnr_db", "ssim")}
    # Compare at the precision reported in the paper.
    assert round(scores["psnr_db"], 2) == 12.86
    assert round(scores["ssim"], 3) == 0.303
    result = {
        "reference": "https://arxiv.org/abs/2302.08207",
        "reference_table": "Table 1, I_3x3, all 1106 pairs",
        "code_commit": UDIS_COMMIT,
        "manifest_sha256": digest(manifest),
        "pairs": len(cases),
        "paper": {"psnr_db": 12.86, "ssim": 0.303},
        "measured": scores,
        "matches_published_precision": True,
        "metric_fixture": "tests/fixtures/udis_metrics.json",
        "records": records,
    }
    write_json(ROOT / "work/protocol_checks.json", result)
    print(scores)


if __name__ == "__main__":
    main()
