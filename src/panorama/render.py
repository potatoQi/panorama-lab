"""Inverse spherical/planar sampling and handwritten multiband blending."""

import cv2
import numpy as np

from .alignment import intrinsics
from .geometry import project


def _corners(shape):
    h, w = shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float)


def spherical_canvas(images, rotations, ratio, max_pixels):
    angles, elevations = [], []
    for im, r in zip(images, rotations):
        h, w = im.shape[:2]
        y, x = np.meshgrid(np.linspace(0, h - 1, 30), np.linspace(0, w - 1, 40), indexing="ij")
        points = np.column_stack((x.ravel(), y.ravel(), np.ones(x.size)))
        rays = points @ np.linalg.inv(intrinsics(im.shape, ratio)).T @ r.T
        angles.extend(np.arctan2(rays[:, 0], rays[:, 2]))
        elevations.extend(np.arctan2(rays[:, 1], np.hypot(rays[:, 0], rays[:, 2])))
    angles = np.sort(np.mod(angles, 2 * np.pi))
    gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
    gap = int(np.argmax(gaps))
    start = angles[(gap + 1) % len(angles)]
    span = 2 * np.pi - gaps[gap]
    full_circle = span > 5.9
    if full_circle:
        start, span = -np.pi, 2 * np.pi
    low, high = float(min(elevations)), float(max(elevations))
    scale = np.median([max(im.shape[:2]) * ratio for im in images])
    scale = min(scale, np.sqrt(max_pixels / (span * (high - low))))
    width, height = int(np.ceil(span * scale)) + 1, int(np.ceil((high - low) * scale)) + 1
    return {
        "projection": "spherical",
        "width": width,
        "height": height,
        "theta_start": float(start),
        "phi_start": low,
        "scale": float(scale),
        "full_circle": bool(full_circle),
    }


def planar_canvas(images, transforms, max_pixels):
    coords = []
    for im, h in zip(images, transforms):
        corners = _corners(im.shape)
        denominator = np.column_stack((corners, np.ones(4))) @ h[2]
        if denominator.min() * denominator.max() <= 0:
            raise ValueError("平面投影跨越无穷远，请切换球面投影。")
        coords.append(project(h, corners))
    coords = np.concatenate(coords)
    if not np.isfinite(coords).all():
        raise ValueError("投影范围无效。")
    low, high = np.floor(coords.min(0)), np.ceil(coords.max(0))
    size = high - low + 1
    if min(size) < 1 or max(size) > 100000:
        raise ValueError("平面投影畸变过大，请切换球面投影。")
    scale = min(1.0, np.sqrt(max_pixels / np.prod(size)))
    t = np.array([[scale, 0, -low[0] * scale], [0, scale, -low[1] * scale], [0, 0, 1]])
    return {
        "projection": "planar",
        "width": int(np.ceil(size[0] * scale)),
        "height": int(np.ceil(size[1] * scale)),
        "canvas_transform": t.tolist(),
        "scale": float(scale),
        "full_circle": False,
    }


def warp_image(image, transform, canvas, ratio=None):
    h, w = canvas["height"], canvas["width"]
    output = np.zeros((h, w, 3), np.float32)
    weights = np.zeros((h, w), np.float32)
    ih, iw = image.shape[:2]
    # Work in strips so a large panorama does not allocate an HxWx3 float64 ray tensor.
    for row in range(0, h, 128):
        y, x = np.mgrid[row : min(row + 128, h), :w].astype(np.float32)
        if canvas["projection"] == "spherical":
            theta = x / canvas["scale"] + canvas["theta_start"]
            phi = y / canvas["scale"] + canvas["phi_start"]
            rays = np.stack(
                (np.sin(theta) * np.cos(phi), np.sin(phi), np.cos(theta) * np.cos(phi)), -1
            )
            camera = rays @ transform
            k = intrinsics(image.shape, ratio)
            z = camera[..., 2]
            safe = np.where(z > 0.01, z, 1)
            mx = camera[..., 0] / safe * k[0, 0] + k[0, 2]
            my = camera[..., 1] / safe * k[1, 1] + k[1, 2]
            visible = z > 0.01
        else:
            inverse = np.linalg.inv(np.array(canvas["canvas_transform"]) @ transform)
            src = np.stack((x, y, np.ones_like(x)), -1) @ inverse.T
            z = src[..., 2]
            safe = np.where(np.abs(z) > 1e-9, z, 1)
            mx, my = src[..., 0] / safe, src[..., 1] / safe
            visible = np.abs(z) > 1e-9
        valid = visible & (mx >= 0) & (mx <= iw - 1) & (my >= 0) & (my <= ih - 1)
        sampled = cv2.remap(
            image,
            mx.astype(np.float32),
            my.astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        output[row : row + len(y)] = sampled * valid[..., None]
        distance = np.minimum.reduce([mx + 1, iw - mx, my + 1, ih - my])
        weights[row : row + len(y)] = np.where(valid, np.maximum(distance, 1) / max(ih, iw), 0)
    return output, weights


def _pyramid(image, levels):
    pyramid = [image.astype(np.float32)]
    for _ in range(levels):
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid


def _expand(image, shape):
    return cv2.pyrUp(image, dstsize=(shape[1], shape[0]))


def render(
    images,
    transforms,
    canvas,
    ratio=None,
    gains=None,
    blending="multiband",
    levels=5,
    progress=None,
):
    height, width = canvas["height"], canvas["width"]
    levels = max(0, min(levels, int(np.log2(min(height, width))) - 2))
    gains = np.ones(len(images)) if gains is None else gains
    owner = np.full((height, width), -1, np.int16)
    strongest = np.zeros((height, width), np.float32)
    coverage = np.zeros((height, width), np.uint16)
    # Determine seam ownership before blending; high-frequency details come from one image.
    for i, (im, transform) in enumerate(zip(images, transforms)):
        _, weight = warp_image(im, transform, canvas, ratio)
        better = weight > strongest
        owner[better] = i
        strongest[better] = weight[better]
        coverage += (weight > 0).astype(np.uint16)
    valid = owner >= 0
    if blending == "feather":
        total = np.zeros((height, width, 3), np.float32)
        norm = np.zeros((height, width), np.float32)
    else:
        shape_pyramid = _pyramid(np.zeros((height, width), np.float32), levels)
        total = [np.zeros((*p.shape, 3), np.float32) for p in shape_pyramid]
        norm = [np.zeros(p.shape, np.float32) for p in shape_pyramid]
    for i, (im, transform, gain) in enumerate(zip(images, transforms, gains)):
        warped, weight = warp_image(im, transform, canvas, ratio)
        warped *= gain
        if blending == "feather":
            total += warped * weight[..., None]
            norm += weight
        else:
            # Extend borders into unsupported pixels before building Laplacian bands.
            # This prevents black rims from leaking into the panorama at coarse scales.
            mask = (weight > 0).astype(np.uint8)
            if mask.any():
                _, labels = cv2.distanceTransformWithLabels(
                    1 - mask, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
                )
                yy, xx = np.nonzero(mask)
                lookup = warped[yy, xx]
                warped = lookup[np.maximum(labels - 1, 0)]
            gp = _pyramid(warped, levels)
            lap = [gp[k] - _expand(gp[k + 1], gp[k].shape) for k in range(levels)] + [gp[-1]]
            wp = _pyramid((owner == i).astype(np.float32), levels)
            support = _pyramid((weight > 0).astype(np.float32), levels)
            for k in range(levels + 1):
                wk = wp[k] * (support[k] > 0.001)
                total[k] += lap[k] * wk[..., None]
                norm[k] += wk
        if progress:
            progress(0.72 + 0.26 * (i + 1) / len(images), f"融合图像 {i + 1}/{len(images)}")
    if blending == "feather":
        result = total / np.maximum(norm[..., None], 1e-8)
    else:
        normalized = [a / np.maximum(b[..., None], 1e-8) for a, b in zip(total, norm)]
        result = normalized[-1]
        for k in range(levels - 1, -1, -1):
            result = _expand(result, normalized[k].shape) + normalized[k]
    result[~valid] = 0
    return np.clip(result, 0, 255).astype(np.uint8), valid, coverage
