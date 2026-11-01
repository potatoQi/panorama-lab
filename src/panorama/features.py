"""Handwritten Harris + oriented gradient descriptor; explicit library baselines."""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter

# Keep the original `sift` key for old runs; it has always meant RootSIFT.
METHODS = {
    "harris": "Harris",
    "orb": "ORB",
    "sift-plain": "SIFT",
    "sift": "RootSIFT",
    "xfeat": "XFeat",
}


@dataclass
class Features:
    points: np.ndarray
    descriptors: np.ndarray
    binary: bool = False


def _describe(gx, gy, points):
    axis = np.arange(-8, 8, dtype=float) + 0.5
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    gaussian = np.exp(-(xx**2 + yy**2) / (2 * 6.0**2))
    descriptors = []
    for x, y in points:
        sample_x, sample_y = x + xx, y + yy
        dx = map_coordinates(gx, [sample_y, sample_x], order=1, mode="reflect")
        dy = map_coordinates(gy, [sample_y, sample_x], order=1, mode="reflect")
        angles = np.mod(np.arctan2(dy, dx), 2 * np.pi)
        hist = np.bincount(
            (angles.ravel() * (36 / (2 * np.pi))).astype(int) % 36,
            weights=(np.hypot(dx, dy) * gaussian).ravel(),
            minlength=36,
        )
        hist = (np.roll(hist, 1) + 2 * hist + np.roll(hist, -1)) / 4
        theta = (np.argmax(hist) + 0.5) * (2 * np.pi / 36)
        c, s = np.cos(theta), np.sin(theta)
        # A 32-pixel support sampled onto a 16x16 grid; primitives only are delegated.
        sx = x + 2 * (c * xx - s * yy)
        sy = y + 2 * (s * xx + c * yy)
        dx = map_coordinates(gx, [sy, sx], order=1, mode="reflect")
        dy = map_coordinates(gy, [sy, sx], order=1, mode="reflect")
        ob = np.mod(np.arctan2(dy, dx) - theta, 2 * np.pi) * (8 / (2 * np.pi))
        mag = np.hypot(dx, dy) * gaussian
        xb, yb = (xx + 8) / 4 - 0.5, (yy + 8) / 4 - 0.5
        hist3 = np.zeros((4, 4, 8), float)
        # Trilinear voting reduces sensitivity to small localization errors.
        for ox in (0, 1):
            xi = np.floor(xb).astype(int) + ox
            wx = 1 - np.abs(xb - xi)
            for oy in (0, 1):
                yi = np.floor(yb).astype(int) + oy
                wy = 1 - np.abs(yb - yi)
                valid = (xi >= 0) & (xi < 4) & (yi >= 0) & (yi < 4)
                for oo in (0, 1):
                    oi = np.floor(ob).astype(int) + oo
                    weight = mag * wx * wy * (1 - np.abs(ob - oi))
                    np.add.at(hist3, (yi[valid], xi[valid], oi[valid] % 8), weight[valid])
        desc = hist3.ravel()
        desc /= np.linalg.norm(desc) + 1e-12
        desc = np.minimum(desc, 0.2)
        # Root normalization is shared with the SIFT baseline.
        desc = np.sqrt(desc / (desc.sum() + 1e-12))
        descriptors.append(desc)
    return np.asarray(descriptors, dtype=np.float32).reshape(-1, 128)


def detect(rgb: np.ndarray, method="sift", max_features=1800) -> Features:
    if method == "xfeat":
        from .xfeat import extract

        points, descriptors = extract(rgb, max_features)
        return Features(points, descriptors)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if method in {"sift", "sift-plain", "orb"}:
        detector = (
            cv2.SIFT_create(nfeatures=max_features, contrastThreshold=0.025)
            if method in {"sift", "sift-plain"}
            else cv2.ORB_create(nfeatures=max_features, fastThreshold=12)
        )
        kp, desc = detector.detectAndCompute(gray, None)
        pts = np.array([k.pt for k in kp], dtype=np.float64).reshape(-1, 2)
        if desc is None:
            desc = np.empty(
                (0, 32 if method == "orb" else 128),
                dtype=np.uint8 if method == "orb" else np.float32,
            )
        if method == "sift":
            desc = np.sqrt(desc / (desc.sum(axis=1, keepdims=True) + 1e-12))
        return Features(pts, desc, method == "orb")
    if method != "harris":
        raise ValueError(f"未知特征算法：{method}")
    points, descriptors = [], []
    for scale in (1.0, 0.75, 0.5):
        level = cv2.resize(gray, None, fx=scale, fy=scale).astype(float) / 255
        if min(level.shape) < 48:
            continue
        smooth = gaussian_filter(level, 1)
        gy, gx = np.gradient(smooth)
        a, b, c = (gaussian_filter(v, 1.5) for v in (gx * gx, gy * gy, gx * gy))
        response = a * b - c * c - 0.04 * (a + b) ** 2
        maxima = (response == maximum_filter(response, size=9)) & (
            response > max(response.max() * 0.005, 1e-10)
        )
        maxima[:24] = maxima[-24:] = False
        maxima[:, :24] = maxima[:, -24:] = False
        ys, xs = np.nonzero(maxima)
        order = np.argsort(response[ys, xs])[::-1][: max_features // 3]
        p = np.column_stack((xs[order], ys[order])).astype(float)
        descriptors.append(_describe(gx, gy, p))
        # Actual dimensions, not the requested scale, define coordinates after rounding.
        points.append(p / [level.shape[1] / gray.shape[1], level.shape[0] / gray.shape[0]])
    if not points:
        return Features(np.empty((0, 2)), np.empty((0, 128), np.float32))
    return Features(np.concatenate(points), np.concatenate(descriptors))


def match(a: Features, b: Features, ratio=0.78) -> tuple[np.ndarray, np.ndarray]:
    """Exact brute-force 2-NN, ratio test and mutual nearest-neighbor filtering."""
    if len(a.points) < 2 or len(b.points) < 2:
        return np.empty((0, 2), int), np.empty(0)
    nearest, second, distances = [], [], []
    reverse_distance = np.full(len(b.points), np.inf)
    reverse_index = np.full(len(b.points), -1, dtype=int)
    # For bits, squared Euclidean distance equals Hamming distance exactly.
    # Matrix multiplication avoids a large (features x features x bytes) XOR tensor.
    da_all = np.unpackbits(a.descriptors, axis=1).astype(np.float32) if a.binary else a.descriptors
    db = np.unpackbits(b.descriptors, axis=1).astype(np.float32) if b.binary else b.descriptors
    block = 256
    for start in range(0, len(a.points), block):
        da = da_all[start : start + block]
        d = np.maximum((da * da).sum(1)[:, None] + (db * db).sum(1)[None] - 2 * da @ db.T, 0)
        best2 = np.argpartition(d, 1, axis=1)[:, :2]
        best2 = np.take_along_axis(best2, np.argsort(np.take_along_axis(d, best2, 1), 1), 1)
        rows = np.arange(len(da))
        nearest.extend(best2[:, 0])
        second.extend(d[rows, best2[:, 1]])
        distances.extend(d[rows, best2[:, 0]])
        ri = np.argmin(d, axis=0)
        rd = d[ri, np.arange(len(b.points))]
        better = rd < reverse_distance
        reverse_index[better] = start + ri[better]
        reverse_distance[better] = rd[better]
    nearest, distances, second = np.array(nearest), np.array(distances), np.array(second)
    threshold = ratio if a.binary else ratio**2
    ids = np.arange(len(a.points))
    valid = (distances < threshold * second) & (reverse_index[nearest] == ids)
    pairs = np.column_stack((ids[valid], nearest[valid]))
    quality = distances[valid] / np.maximum(second[valid], 1e-12)
    order = np.argsort(quality)
    return pairs[order], quality[order]
