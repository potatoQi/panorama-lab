"""Named evaluation protocols; panorama inference remains independent of these rules."""

import time
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .features import detect, match
from .geometry import project, ransac

SUPERPOINT = "hpatches-superpoint2018-v1"
UDIS = "udis2-2023-v1"
LEGACY = "legacy-v1"
DEFAULT_PROTOCOL = {"HPatches": SUPERPOINT, "UDIS-D": UDIS, "APAP": LEGACY}
PROTOCOL_LABELS = {SUPERPOINT: "SuperPoint 2018", UDIS: "UDIS++ 2023", LEGACY: "自定义 v1"}
UDIS_COMMIT = "b4158950ab1411268c67a48169ba732a19d399d0"
PROTOCOLS = {
    SUPERPOINT: {
        "name": "SuperPoint 2018, section 7.3 and appendix A",
        "reference": "https://arxiv.org/abs/1712.07629",
        "preprocessing": "OpenCV grayscale decode and INTER_LINEAR resize to width=640, height=480; no crop.",
        "max_features": 1000,
        "matching": "One-way nearest neighbor, L2 for floating descriptors / Hamming for ORB; no ratio or mutual filter; no ground-truth feature filtering.",
        "estimator": "OpenCV findHomography RANSAC: 3 px, maxIters=2000, confidence=0.995, RNG seed=42.",
        "geometry": "Mean error of four corners in resized-image pixels; H-Acc <= 1/3/5; all failed estimates count as incorrect.",
        "failure_policy": "No extra inlier-count, inlier-ratio or photometric-overlap gate.",
        "comparability": "公开评测设置对齐；原论文评测代码未公开，具体实现存在差异。",
    },
    UDIS: {
        "name": "UDIS++ 2023, section 4.2.1 and official Warp/Codes/test.py",
        "reference": "https://arxiv.org/abs/2302.08207",
        "reference_code": f"https://github.com/nie-lang/UDIS2/blob/{UDIS_COMMIT}/Warp/Codes/test.py",
        "reference_commit": UDIS_COMMIT,
        "preprocessing": "OpenCV color decode, original 512x512 test images, float32 normalized to [-1,1].",
        "max_features": 1800,
        "matching": "Our feature pipeline: ratio=0.78 and mutual nearest neighbor.",
        "estimator": "Our normalized DLT/RANSAC, seed=42, threshold=3px; at least 12 inliers and 20% inlier ratio.",
        "photometry": "Warp input2 into input1 with bilinear interpolation; apply the warped all-ones mask to both color images; PSNR over the entire image; per-channel SSIM (7x7 uniform window, sample covariance), data_range=255; no mask erosion, cropping or blending.",
        "failure_policy": "Identity warp on estimation failure (paper section 4.2.1); every test pair contributes to PSNR/SSIM.",
        "comparability": "测试集与指标口径对齐；匹配和图像变形实现不同。",
    },
    LEGACY: {
        "name": "Project-defined adjacent-pair supplementary experiment",
        "reference": "https://cs.adelaide.edu.au/~tjchin/apap/",
        "max_features": 1800,
        "photometry": "Grayscale valid-overlap PSNR/SSIM, 4px erosion, successful estimates only.",
        "comparability": "相邻图像对补充实验，非 APAP 原论文 RMSE 协议。",
    },
}


def resized_homography(h, size_a, size_b, target=(640, 480)):
    """Change both coordinate systems, allowing different source image dimensions."""
    sa = np.diag([target[0] / size_a[0], target[1] / size_a[1], 1.0])
    sb = np.diag([target[0] / size_b[0], target[1] / size_b[1], 1.0])
    return sb @ np.asarray(h, float) @ np.linalg.inv(sa)


def corner_error(h, truth, width, height):
    corners = np.array([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]], float)
    error = np.linalg.norm(project(h, corners) - project(truth, corners), axis=1).mean()
    if not np.isfinite(error):
        raise ValueError("角点投影无效。")
    return float(error)


def paper_features(image, method):
    if method in {"harris", "xfeat"}:
        return detect(image, method, 1000)
    if method not in {"sift", "sift-plain", "orb"}:
        raise ValueError(f"未知特征算法：{method}")
    # OpenCV detectors with the 1000-feature limit from SuperPoint section 7.3.
    from .features import Features

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    detector = (
        cv2.SIFT_create(nfeatures=1000)
        if method in {"sift", "sift-plain"}
        else cv2.ORB_create(nfeatures=1000)
    )
    kp, desc = detector.detectAndCompute(gray, None)
    if desc is None:
        raise ValueError("未检测到可用特征。")
    ids = np.argsort([-k.response for k in kp], kind="stable")[:1000]
    points = np.array([kp[i].pt for i in ids], float)
    desc = desc[ids]
    if method == "sift":
        desc = np.sqrt(desc / (desc.sum(axis=1, keepdims=True) + 1e-12))
    return Features(points, desc, method == "orb")


def nearest_matches(a, b):
    if not len(a.points) or not len(b.points):
        raise ValueError("未检测到可用特征。")
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING if a.binary else cv2.NORM_L2, crossCheck=False)
    matches = matcher.match(a.descriptors, b.descriptors)
    return np.array([[m.queryIdx, m.trainIdx] for m in matches], int).reshape(-1, 2)


def udis_metrics(reference, target, h):
    """Official color/masked-full-image metric; h maps input2 into input1."""
    height, width = reference.shape[:2]
    # Match the official float32 dataset normalization and metric denormalization.
    ref = reference.astype(np.float32) / 127.5 - 1
    source = target.astype(np.float32) / 127.5 - 1
    warped = cv2.warpPerspective(source, h, (width, height), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(
        np.ones(target.shape[:2], np.float32), h, (width, height), flags=cv2.INTER_LINEAR
    )
    # An empty warp must not receive an infinite score for two all-black images.
    if not np.isfinite(warped).all() or not np.isfinite(mask).all() or not np.any(mask > 0):
        raise ValueError("变换未产生有效重叠。")
    a = ((ref + 1) * 127.5) * mask[..., None]
    b = ((warped + 1) * 127.5) * mask[..., None]
    psnr = float(peak_signal_noise_ratio(a, b, data_range=255))
    return {
        "psnr_db": psnr if np.isfinite(psnr) else None,
        "ssim": float(
            structural_similarity(
                a.astype(np.float64),
                b.astype(np.float64),
                data_range=255,
                channel_axis=-1,
                win_size=7,
                gaussian_weights=False,
                use_sample_covariance=True,
            )
        ),
        "overlap_fraction": float(mask.mean()),
    }


def read_cv(path, gray=False):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"无法读取图像：{path}")
    return image if gray else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def paper_case(job):
    item, method, _, base, protocol = job
    cv2.setNumThreads(1)
    cv2.setRNGSeed(42)
    start = time.perf_counter()
    record = {
        "dataset": item["dataset"],
        "case": item["id"],
        "method": method,
        "protocol": protocol,
        "status": "failed",
        "matches": 0,
        "inliers": 0,
        "has_ground_truth": "homography" in item,
    }
    # Missing/corrupt inputs must abort the run, not masquerade as algorithm failures.
    a, b = [read_cv(Path(base) / item[k], gray=protocol == SUPERPOINT) for k in ("a", "b")]
    if protocol == SUPERPOINT:
        truth = resized_homography(item["homography"], a.shape[::-1], b.shape[::-1])
        a, b = [
            cv2.cvtColor(
                cv2.resize(x, (640, 480), interpolation=cv2.INTER_LINEAR), cv2.COLOR_GRAY2RGB
            )
            for x in (a, b)
        ]
    elif a.shape != (512, 512, 3) or b.shape != (512, 512, 3):
        raise ValueError("UDIS-D 官方测试协议要求原始 512×512 图像。")
    h = np.eye(3)
    try:
        if protocol == SUPERPOINT:
            fa, fb = paper_features(a, method), paper_features(b, method)
            ids = nearest_matches(fa, fb)
            src, dst = fa.points[ids[:, 0]], fb.points[ids[:, 1]]
            record["matches"] = len(ids)
            if len(ids) < 4:
                raise ValueError("有效匹配少于四组。")
            h, inliers = cv2.findHomography(
                src, dst, cv2.RANSAC, 3.0, maxIters=2000, confidence=0.995
            )
            if h is None or not np.isfinite(h).all():
                raise ValueError("RANSAC 未找到有效单应矩阵。")
            record["inliers"] = int(inliers.sum())
            record["corner_error_px"] = corner_error(h, truth, 640, 480)
        else:
            fa, fb = detect(a, method, 1800), detect(b, method, 1800)
            ids, _ = match(fa, fb, 0.78)
            record["matches"] = len(ids)
            # Official metric evaluates the warped target in the reference frame.
            estimate = ransac(fb.points[ids[:, 1]], fa.points[ids[:, 0]], seed=42)
            record["inliers"] = int(estimate.inliers.sum())
            if record["inliers"] < 12 or estimate.inliers.mean() < 0.2:
                raise ValueError("少于 12 个内点或内点率低于 20%。")
            h = estimate.homography
            record.update(udis_metrics(a, b, h))
        record["homography"] = h.tolist()
        record["status"] = "ok"
    except (ValueError, np.linalg.LinAlgError, cv2.error) as error:
        record["error"] = str(error)
        if protocol == UDIS:
            record["fallback"] = "identity"
            record["homography"] = np.eye(3).tolist()
            record.update(udis_metrics(a, b, np.eye(3)))
    record["seconds"] = time.perf_counter() - start
    return record
