"""Normalized DLT and adaptive RANSAC, with no OpenCV geometry estimators."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np


def project(h, points):
    p = np.column_stack((points, np.ones(len(points)))) @ np.asarray(h).T
    z = p[:, 2:3]
    safe = np.where(np.abs(z) > 1e-12, z, np.nan)
    return p[:, :2] / safe


def normalize(points):
    center = points.mean(axis=0)
    distance = np.linalg.norm(points - center, axis=1).mean()
    if distance < 1e-9:
        raise ValueError("退化点集：位置重合。")
    s = np.sqrt(2) / distance
    t = np.array([[s, 0, -s * center[0]], [0, s, -s * center[1]], [0, 0, 1.0]])
    return (points - center) * s, t


def dlt(src, dst):
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if len(src) < 4 or src.shape != dst.shape:
        raise ValueError("单应矩阵至少需要四组对应点。")
    a, ta = normalize(src)
    b, tb = normalize(dst)
    x, y = a.T
    u, v = b.T
    z, o = np.zeros(len(a)), np.ones(len(a))
    matrix = np.stack(
        [
            np.column_stack((-x, -y, -o, z, z, z, u * x, u * y, u)),
            np.column_stack((z, z, z, -x, -y, -o, v * x, v * y, v)),
        ],
        axis=1,
    ).reshape(-1, 9)
    # Minimal 8x9 systems require the ninth right singular vector.
    _, singular, vt = np.linalg.svd(matrix, full_matrices=len(src) == 4)
    if singular[-2] < 1e-10 or (len(src) == 4 and singular[-1] < 1e-10):
        raise ValueError("退化点集：无法确定单应矩阵。")
    h = np.linalg.inv(tb) @ vt[-1].reshape(3, 3) @ ta
    if abs(h[2, 2]) < 1e-12 or not np.isfinite(h).all():
        raise ValueError("无效单应矩阵。")
    return h / h[2, 2]


def _degenerate(p):
    # No three of the four minimal points may be collinear.
    for i, j, k in combinations(range(4), 3):
        u, v = p[j] - p[i], p[k] - p[i]
        if abs(u[0] * v[1] - u[1] * v[0]) < 1e-4:
            return True
    return False


@dataclass
class Estimate:
    homography: np.ndarray
    inliers: np.ndarray
    errors: np.ndarray
    iterations: int


def ransac(src, dst, threshold=3.0, max_iterations=2000, confidence=0.999, seed=42):
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if len(src) < 4:
        raise ValueError("有效匹配少于四组。")
    rng = np.random.default_rng(seed)
    best, best_h, best_score = np.zeros(len(src), bool), None, (-1, -np.inf)
    limit, count = max_iterations, 0
    while count < max(32, limit) and count < max_iterations:
        count += 1
        ids = rng.choice(len(src), 4, replace=False)
        if _degenerate(src[ids]) or _degenerate(dst[ids]):
            continue
        try:
            h = dlt(src[ids], dst[ids])
            errors = np.linalg.norm(project(h, src) - dst, axis=1)
            mask = errors < threshold
            score = (
                int(mask.sum()),
                -float(np.minimum(np.nan_to_num(errors, nan=threshold), threshold).sum()),
            )
            if score > best_score and mask.sum() >= 4:
                best, best_h, best_score = mask, h, score
                probability = (mask.mean()) ** 4
                if probability >= 1:
                    limit = 32
                elif probability > 0:
                    limit = min(
                        limit, int(np.ceil(np.log(1 - confidence) / np.log1p(-probability)))
                    )
        except (ValueError, np.linalg.LinAlgError):
            continue
    if best_h is None:
        raise ValueError("RANSAC 未找到可靠的几何模型。")
    # Local refitting uses all consensus points; preserve the best scoring model.
    for _ in range(3):
        try:
            h = dlt(src[best], dst[best])
            errors = np.linalg.norm(project(h, src) - dst, axis=1)
            mask = errors < threshold
            score = (
                int(mask.sum()),
                -float(np.minimum(np.nan_to_num(errors, nan=threshold), threshold).sum()),
            )
            if score < best_score:
                break
            stable = np.array_equal(best, mask)
            best_h, best, best_score = h, mask, score
            if stable:
                break
        except (ValueError, np.linalg.LinAlgError):
            break
    errors = np.linalg.norm(project(best_h, src) - dst, axis=1)
    return Estimate(best_h, errors < threshold, errors, count)
