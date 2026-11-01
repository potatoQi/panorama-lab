"""Shared stitching service used by the CLI and UI."""

import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from .alignment import Edge, align_planar, align_spherical, exposure_gains, spanning_tree
from .features import METHODS, detect, match
from .geometry import ransac
from .io import code_state, digest, image_paths, load_image, preview_matches, save_rgb, write_json
from .render import planar_canvas, render, spherical_canvas


@dataclass
class Config:
    feature: str = "sift"
    projection: str = "spherical"
    blending: str = "multiband"
    max_size: int = 1200
    max_features: int = 1800
    ratio: float = 0.78
    threshold: float = 3.0
    seed: int = 42
    max_megapixels: float = 12.0
    optimize: bool = True
    exposure: bool = True

    def validate(self):
        if self.feature not in METHODS:
            raise ValueError(f"未知特征算法：{self.feature}")
        if self.projection not in {"spherical", "planar"}:
            raise ValueError("投影必须为 spherical / planar。")
        if self.blending not in {"multiband", "feather"}:
            raise ValueError("融合必须为 multiband / feather。")
        if not (128 <= self.max_size <= 4096 and 100 <= self.max_features <= 10000):
            raise ValueError("处理边长应为 128–4096，特征数量应为 100–10000。")
        if not (
            0 < self.ratio < 1 and 0 < self.threshold < 30 and 0.1 <= self.max_megapixels <= 50
        ):
            raise ValueError("匹配或画布参数超出支持范围。")


def stitch(folder, output, config=None, progress=None):
    config = config or Config()
    config.validate()
    progress = progress or (lambda fraction, text: None)
    paths = image_paths(folder)
    if len(paths) > 120:
        raise ValueError("当前全配对实现每次最多支持 120 张图片，请分组处理。")
    output = Path(output).expanduser().resolve()
    if output == Path(folder).expanduser().resolve():
        raise ValueError("输出目录不能与输入目录相同。")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise ValueError("输出目录已有实验记录，请使用新的输出目录。")
    start = time.perf_counter()
    cv2.setNumThreads(1)
    images, metadata, features = [], [], []
    for i, path in enumerate(paths):
        rgb, meta = load_image(path, config.max_size)
        ft = detect(rgb, config.feature, config.max_features)
        images.append(rgb)
        metadata.append(meta)
        features.append(ft)
        meta["sha256"] = digest(path)
        meta["features"] = len(ft.points)
        progress(0.2 * (i + 1) / len(paths), f"提取特征 {i + 1}/{len(paths)} · {path.name}")
    feature_seconds = time.perf_counter() - start
    edges, rows = [], []
    pairs = list(combinations(range(len(paths)), 2))
    for number, (i, j) in enumerate(pairs):
        indices, _ = match(features[i], features[j], config.ratio)
        src, dst = features[i].points[indices[:, 0]], features[j].points[indices[:, 1]]
        row = {
            "i": i,
            "j": j,
            "image_i": paths[i].name,
            "image_j": paths[j].name,
            "matches": len(indices),
            "inliers": 0,
            "accepted": False,
        }
        if len(indices) >= 12:
            try:
                estimate = ransac(
                    src, dst, threshold=config.threshold, seed=config.seed + i * 1009 + j
                )
                inliers = int(estimate.inliers.sum())
                fraction = inliers / len(indices)
                support = np.prod(np.ptp(src[estimate.inliers], axis=0)) / np.prod(
                    images[i].shape[:2]
                )
                accepted = inliers >= 12 and fraction >= 0.2 and support > 0.003
                row.update(
                    inliers=inliers,
                    inlier_ratio=fraction,
                    rmse_px=float(np.sqrt(np.mean(estimate.errors[estimate.inliers] ** 2))),
                    accepted=bool(accepted),
                    iterations=estimate.iterations,
                )
                if accepted:
                    edges.append(
                        Edge(
                            i,
                            j,
                            estimate.homography,
                            src[estimate.inliers],
                            dst[estimate.inliers],
                            row["rmse_px"],
                        )
                    )
            except (ValueError, np.linalg.LinAlgError) as error:
                row["reason"] = str(error)
        rows.append(row)
        progress(0.2 + 0.4 * (number + 1) / len(pairs), f"匹配重叠图像 {number + 1}/{len(pairs)}")
    write_json(output / "pairs.json", rows)
    matching_seconds = time.perf_counter() - start - feature_seconds
    root, tree = spanning_tree(len(images), edges)
    progress(0.61, "全局对齐与相机参数优化")
    if config.projection == "spherical":
        transforms, focal_ratio, details = align_spherical(
            images, edges, root, tree, config.optimize
        )
        canvas = spherical_canvas(images, transforms, focal_ratio, config.max_megapixels * 1e6)
    else:
        transforms = align_planar(images, tree)
        focal_ratio, details = None, {}
        canvas = planar_canvas(images, transforms, config.max_megapixels * 1e6)
    gains = exposure_gains(images, edges) if config.exposure else np.ones(len(images))
    alignment_seconds = time.perf_counter() - start - feature_seconds - matching_seconds
    progress(0.7, "生成全景画布并计算接缝")
    panorama, mask, coverage = render(
        images, transforms, canvas, focal_ratio, gains, config.blending, progress=progress
    )
    save_rgb(output / "panorama.jpg", panorama)
    save_rgb(output / "panorama.png", panorama)
    (output / "matches").mkdir(exist_ok=True)
    for e in sorted(edges, key=lambda e: e.weight, reverse=True)[:12]:
        preview = preview_matches(images[e.i], images[e.j], e.src, e.dst)
        save_rgb(output / "matches" / f"{e.i:02d}-{e.j:02d}.jpg", preview)
    total = time.perf_counter() - start
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(Path(folder).resolve()),
        "config": asdict(config),
        "code": code_state(),
        "images": metadata,
        "input_count": len(paths),
        "used_count": len(images),
        "root": root,
        "tree": [[i, j] for i, j, _ in tree],
        "overlap_edges": len(edges),
        "canvas": canvas,
        "transforms": transforms,
        "exposure_gains": gains,
        "alignment": details,
        "metrics": {
            "coverage_fraction": float(mask.mean()),
            "overlap_fraction": float((coverage[mask] > 1).mean()),
            "median_pair_inlier_rmse_px": float(np.median([e.rmse for e in edges])),
        },
        "timing": {
            "feature_seconds": feature_seconds,
            "matching_seconds": matching_seconds,
            "alignment_seconds": alignment_seconds,
            "render_and_export_seconds": total
            - feature_seconds
            - matching_seconds
            - alignment_seconds,
            "total_seconds": total,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "platform": platform.platform(),
        },
    }
    write_json(output / "manifest.json", manifest)
    progress(1.0, "拼接完成 · 所有输入图片均已接入")
    return manifest
