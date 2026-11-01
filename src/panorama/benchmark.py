"""Pairwise geometric and photometric evaluation with explicit failure accounting."""

import csv
import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity

from .features import METHODS, detect, match
from .geometry import project, ransac
from .io import code_state, digest, load_image, write_json
from .protocols import DEFAULT_PROTOCOL, LEGACY, PROTOCOLS, SUPERPOINT, UDIS, paper_case


def overlap_metrics(a, b, h):
    height, width = b.shape[:2]
    warped = cv2.warpPerspective(a, h, (width, height))
    support = (
        cv2.warpPerspective(
            np.ones(a.shape[:2], np.uint8), h, (width, height), flags=cv2.INTER_NEAREST
        )
        > 0
    )
    mask = binary_erosion(support, iterations=4)
    fraction = float(mask.mean())
    if mask.sum() < 100 or fraction < 0.05:
        raise ValueError("有效重叠区域小于目标图像的 5%。")
    # Evaluate before exposure correction and blending to avoid rewarding the renderer.
    gray_a = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY).astype(float)
    gray_b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(float)
    mse = float(np.mean((gray_a[mask] - gray_b[mask]) ** 2))
    psnr = float(10 * np.log10(255**2 / max(mse, 1e-12)))
    _, ssim_map = structural_similarity(gray_a, gray_b, data_range=255, full=True)
    return {
        "overlap_fraction": fraction,
        "overlap_psnr_db": psnr,
        "overlap_ssim": float(ssim_map[mask].mean()),
    }


def _case(job):
    if job[4] != LEGACY:
        return paper_case(job)
    record = _legacy_case(job[:4])
    record["protocol"] = LEGACY
    return record


def _legacy_case(job):
    item, method, max_size, base = job
    cv2.setNumThreads(1)
    start = time.perf_counter()
    record = {
        "dataset": item["dataset"],
        "case": item["id"],
        "method": method,
        "status": "failed",
        "matches": 0,
        "inliers": 0,
        "has_ground_truth": "homography" in item,
    }
    try:
        a, ma = load_image(Path(base) / item["a"], max_size)
        b, mb = load_image(Path(base) / item["b"], max_size)
        fa, fb = detect(a, method, 1800), detect(b, method, 1800)
        indices, _ = match(fa, fb, 0.78)
        src, dst = fa.points[indices[:, 0]], fb.points[indices[:, 1]]
        record["matches"] = len(indices)
        estimate = ransac(src, dst, seed=42)
        record.update(
            inliers=int(estimate.inliers.sum()),
            inlier_ratio=float(estimate.inliers.mean()),
            inlier_rmse_px=float(np.sqrt(np.mean(estimate.errors[estimate.inliers] ** 2))),
        )
        if record["inliers"] < 12 or record["inlier_ratio"] < 0.2:
            raise ValueError("少于 12 个内点或内点率低于 20%。")
        record["homography"] = estimate.homography.tolist()
        if "homography" in item:
            target = np.array(item["homography"], float)
            sa = np.diag([*ma["scale_xy"], 1.0])
            sb = np.diag([*mb["scale_xy"], 1.0])
            original_h = np.linalg.inv(sb) @ estimate.homography @ sa
            width, height = ma["original_size"]
            corners = np.array(
                [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], float
            )
            error = np.linalg.norm(
                project(original_h, corners) - project(target, corners), axis=1
            ).mean()
            if not np.isfinite(error):
                raise ValueError("角点投影无效。")
            record["corner_error_px"] = float(error)
        record.update(overlap_metrics(a, b, estimate.homography))
        record["status"] = "ok"
    except (ValueError, OSError, np.linalg.LinAlgError, cv2.error) as error:
        record["error"] = str(error)
    record["seconds"] = time.perf_counter() - start
    return record


def summarize(records):
    groups = sorted({(r["dataset"], r["method"]) for r in records})
    rows = []
    for dataset, method in groups:
        cases = [r for r in records if r["dataset"] == dataset and r["method"] == method]
        good = [r for r in cases if r["status"] == "ok"]
        row = {
            "dataset": dataset,
            "method": method,
            "pairs": len(cases),
            "succeeded": len(good),
            "success_rate": len(good) / len(cases),
            "mean_seconds": float(np.mean([r["seconds"] for r in cases])),
        }
        row["protocol"] = cases[0].get("protocol", LEGACY)
        for metric in (
            "inlier_ratio",
            "inlier_rmse_px",
            "overlap_psnr_db",
            "overlap_ssim",
            "overlap_fraction",
        ):
            values = [r[metric] for r in good if metric in r]
            if values:
                row[metric] = float(np.mean(values))
            elif row["protocol"] == LEGACY:
                row[metric] = None
        if row["protocol"] == UDIS:
            for metric in ("psnr_db", "ssim"):
                if any(r.get(metric) is None for r in cases):
                    raise ValueError("全量光度评测含未定义指标，请检查原始记录。")
                row[metric] = float(np.mean([r[metric] for r in cases]))
            row["identity_fallbacks"] = sum(r.get("fallback") == "identity" for r in cases)
        gt = [r for r in cases if r["has_ground_truth"]]
        if gt:
            errors = [
                r.get("corner_error_px", np.inf) if r["status"] == "ok" else np.inf for r in gt
            ]
            for threshold in (1, 3, 5):
                row[f"h_accuracy_{threshold}px"] = float(np.mean(np.asarray(errors) <= threshold))
        rows.append(row)
    return rows


def _prepare_worker(methods):
    if "xfeat" in methods:
        from .xfeat import extract

        # Exclude model initialization from per-pair timings, as with warm detectors.
        extract(np.zeros((480, 640, 3), np.uint8), 1000)


def evaluate(
    manifest,
    output,
    methods=tuple(METHODS),
    max_size=None,
    workers=4,
    limit=None,
    protocol="auto",
):
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"未知方法：{method}")
    manifest = Path(manifest).resolve()
    spec = json.loads(manifest.read_text())
    datasets = {p["dataset"] for p in spec["pairs"]}
    if protocol == "auto":
        if len(datasets) != 1:
            raise ValueError("自动协议要求清单只包含一个数据集。")
        protocol = DEFAULT_PROTOCOL.get(next(iter(datasets)), LEGACY)
    if protocol not in PROTOCOLS:
        raise ValueError(f"未知评测协议：{protocol}")
    if protocol == SUPERPOINT and datasets != {"HPatches"}:
        raise ValueError("SuperPoint 协议只适用于 HPatches。")
    if protocol == UDIS and datasets != {"UDIS-D"}:
        raise ValueError("UDIS++ 协议只适用于 UDIS-D。")
    if protocol != LEGACY and max_size is not None:
        raise ValueError(
            "论文协议固定输入尺寸，请移除 --max-size；自定义尺寸请用 --protocol legacy-v1。"
        )
    max_size = (max_size or 800) if protocol == LEGACY else (640 if protocol == SUPERPOINT else 512)
    cases = spec["pairs"][:limit] if limit else spec["pairs"]
    if not cases:
        raise ValueError("评测清单为空。")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "records.jsonl").exists():
        raise ValueError("输出目录已有记录，请指定新目录以免覆盖实验。")
    base = (manifest.parent / spec.get("root", ".")).resolve()
    jobs = [(case, method, max_size, str(base), protocol) for case in cases for method in methods]
    feature_metadata = {}
    if "xfeat" in methods:
        from .xfeat import provenance

        feature_metadata["xfeat"] = provenance()
    source_state = code_state()
    records = []
    start = time.perf_counter()
    # Limit BLAS threads before spawned workers import NumPy.
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    from multiprocessing import get_context

    with (
        (output / "records.jsonl").open("w") as log,
        ProcessPoolExecutor(
            max_workers=max(1, min(workers, 8)),
            mp_context=get_context("spawn"),
            initializer=_prepare_worker,
            initargs=(methods,),
        ) as pool,
    ):
        futures = [pool.submit(_case, job) for job in jobs]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()
            if len(records) % 20 == 0 or len(records) == len(jobs):
                print(f"评测 {len(records)}/{len(jobs)}", flush=True)
    records.sort(key=lambda r: (r["dataset"], r["case"], r["method"]))
    summary = summarize(records)
    write_json(output / "summary.json", summary)
    fields = list(dict.fromkeys(key for row in summary for key in row))
    with (output / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    write_json(
        output / "protocol.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "manifest": str(manifest),
            "manifest_sha256": digest(manifest),
            "code": source_state,
            "protocol_id": protocol,
            "evaluation": PROTOCOLS[protocol],
            "source": spec.get("source"),
            "dataset_protocol": spec.get("protocol"),
            "split": spec.get("split"),
            "methods": list(methods),
            "feature_metadata": feature_metadata,
            "max_size": max_size,
            "max_features": PROTOCOLS[protocol]["max_features"],
            "ratio": None if protocol == SUPERPOINT else 0.78,
            "ransac_threshold_px": 3.0,
            "seed": 42,
            "workers": workers,
            "pairs": len(cases),
            "available_pairs": len(spec["pairs"]),
            "is_subset": len(cases) != len(spec["pairs"]),
            "seconds": time.perf_counter() - start,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "geometry": PROTOCOLS[protocol].get(
                "geometry", "Legacy original-image corner error where available."
            ),
            "photometry": PROTOCOLS[protocol].get("photometry"),
        },
    )
    return summary
