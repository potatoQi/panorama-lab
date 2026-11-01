"""Overlap graph, maximum spanning tree and robust rotation bundle adjustment."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation


@dataclass
class Edge:
    i: int
    j: int
    h: np.ndarray  # Maps image i pixels to image j pixels.
    src: np.ndarray
    dst: np.ndarray
    rmse: float

    @property
    def weight(self):
        return len(self.src) / (1 + self.rmse)


def spanning_tree(n, edges):
    if not edges:
        raise ValueError("没有找到可靠的重叠图像。请尝试 SIFT 或提高处理分辨率。")
    strength = np.zeros(n)
    for e in edges:
        strength[e.i] += e.weight
        strength[e.j] += e.weight
    root = int(np.argmax(strength))
    visited, tree = {root}, []
    while len(visited) < n:
        candidates = [e for e in edges if (e.i in visited) != (e.j in visited)]
        if not candidates:
            missing = sorted(set(range(n)) - visited)
            raise ValueError(f"匹配图未连通，图像索引 {missing} 无法接入。")
        edge = max(candidates, key=lambda e: e.weight)
        parent, child = (edge.i, edge.j) if edge.i in visited else (edge.j, edge.i)
        tree.append((parent, child, edge))
        visited.add(child)
    return root, tree


def intrinsics(shape, ratio):
    h, w = shape[:2]
    f = max(h, w) * ratio
    return np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]])


def closest_rotation(matrix):
    # Homographies are defined only up to scale, including sign.
    if np.linalg.det(matrix) < 0:
        matrix = -matrix
    u, _, vt = np.linalg.svd(matrix)
    return u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt


def align_spherical(images, edges, root, tree, optimize=True):
    n = len(images)
    shapes = [im.shape for im in images]

    def focal_cost(log_ratio):
        ks = [intrinsics(s, np.exp(log_ratio)) for s in shapes]
        values = []
        for e in sorted(edges, key=lambda e: e.weight, reverse=True)[: max(20, n)]:
            m = np.linalg.inv(ks[e.j]) @ e.h @ ks[e.i]
            sv = np.linalg.svd(m, compute_uv=False)
            values.append(np.sum(np.log(np.maximum(sv / np.exp(np.log(sv).mean()), 1e-9)) ** 2))
        return np.median(values)

    estimate = minimize_scalar(focal_cost, bounds=(np.log(0.35), np.log(2.5)), method="bounded")
    ratio = float(np.exp(estimate.x))
    ks = [intrinsics(s, ratio) for s in shapes]
    rotations = [np.eye(3) for _ in images]  # Camera-to-world convention.
    for parent, child, e in tree:
        q = closest_rotation(np.linalg.inv(ks[e.j]) @ e.h @ ks[e.i])
        if parent == e.i:
            rotations[child] = rotations[parent] @ q.T
        else:
            rotations[child] = rotations[parent] @ q

    active = [i for i in range(n) if i != root]
    slot = {i: k for k, i in enumerate(active)}
    x0 = np.r_[
        np.concatenate([Rotation.from_matrix(rotations[i]).as_rotvec() for i in active]),
        np.log(ratio),
    ]
    samples = []
    for e in edges:
        ids = np.linspace(0, len(e.src) - 1, min(60, len(e.src))).astype(int)
        samples.append((e.i, e.j, e.src[ids], e.dst[ids]))
    pixels = float(np.median([max(s[:2]) for s in shapes]))

    def unpack(x):
        rs = [np.eye(3) for _ in images]
        for i in active:
            rs[i] = Rotation.from_rotvec(x[3 * slot[i] : 3 * slot[i] + 3]).as_matrix()
        return rs, np.exp(x[-1])

    def rays(points, shape, focal_ratio):
        k = intrinsics(shape, focal_ratio)
        r = np.column_stack(
            (
                (points[:, 0] - k[0, 2]) / k[0, 0],
                (points[:, 1] - k[1, 2]) / k[1, 1],
                np.ones(len(points)),
            )
        )
        return r / np.linalg.norm(r, axis=1, keepdims=True)

    def residual(x):
        rs, fr = unpack(x)
        return np.concatenate(
            [
                (
                    (rays(a, shapes[i], fr) @ rs[i].T - rays(b, shapes[j], fr) @ rs[j].T) * pixels
                ).ravel()
                for i, j, a, b in samples
            ]
        )

    initial = residual(x0)
    result = None
    if optimize:
        sparse = lil_matrix((len(initial), len(x0)), dtype=int)
        row = 0
        for i, j, a, _ in samples:
            rows = slice(row, row + len(a) * 3)
            for index in (i, j):
                if index != root:
                    sparse[rows, 3 * slot[index] : 3 * slot[index] + 3] = 1
            sparse[rows, -1] = 1
            row += len(a) * 3
        low, high = np.full(len(x0), -np.inf), np.full(len(x0), np.inf)
        low[-1], high[-1] = np.log(0.25), np.log(4)
        result = least_squares(
            residual,
            x0,
            jac_sparsity=sparse.tocsr(),
            loss="soft_l1",
            f_scale=2,
            max_nfev=100,
            bounds=(low, high),
            ftol=1e-5,
        )
        rotations, ratio = unpack(result.x)
    # Set the averaged camera vertical as world vertical (automatic leveling).
    up = np.mean([r[:, 1] for r in rotations], axis=0)
    up /= np.linalg.norm(up)
    forward = rotations[root][:, 2] - up * np.dot(up, rotations[root][:, 2])
    forward /= np.linalg.norm(forward)
    right = np.cross(up, forward)
    frame = np.column_stack((right, up, forward))
    rotations = [frame.T @ r for r in rotations]
    return (
        rotations,
        ratio,
        {
            "focal_ratio": ratio,
            "bundle_initial_ray_rmse_px": float(np.sqrt(np.mean(initial**2))),
            "bundle_final_ray_rmse_px": float(
                np.sqrt(np.mean(residual(result.x if result is not None else x0) ** 2))
            ),
            "bundle_converged": bool(result.success) if result is not None else None,
            "bundle_evaluations": int(result.nfev) if result is not None else 0,
        },
    )


def align_planar(images, tree):
    transforms = [np.eye(3) for _ in images]
    for parent, child, e in tree:
        h = np.linalg.inv(e.h) if parent == e.i else e.h
        transforms[child] = transforms[parent] @ h
        transforms[child] /= transforms[child][2, 2]
    return transforms


def exposure_gains(images, edges):
    rows, values = [], []
    for e in edges:
        p, q = np.rint(e.src).astype(int), np.rint(e.dst).astype(int)
        a, b = images[e.i], images[e.j]
        valid = (
            (p[:, 0] >= 0)
            & (p[:, 0] < a.shape[1])
            & (p[:, 1] >= 0)
            & (p[:, 1] < a.shape[0])
            & (q[:, 0] >= 0)
            & (q[:, 0] < b.shape[1])
            & (q[:, 1] >= 0)
            & (q[:, 1] < b.shape[0])
        )
        av = a[p[valid, 1], p[valid, 0]].mean(axis=1)
        bv = b[q[valid, 1], q[valid, 0]].mean(axis=1)
        valid = (av > 15) & (av < 240) & (bv > 15) & (bv < 240)
        if valid.sum() < 8:
            continue
        row = np.zeros(len(images))
        row[e.i] = 1
        row[e.j] = -1
        rows.append(row)
        values.append(np.median(np.log(bv[valid] / av[valid])))
    if not rows:
        return np.ones(len(images))
    rows.append(np.ones(len(images)) / len(images))
    values.append(0.0)
    gains = np.exp(np.linalg.lstsq(np.array(rows), np.array(values), rcond=None)[0])
    return np.clip(gains, 0.6, 1.7)
