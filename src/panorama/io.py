"""Image loading, orientation, provenance and JSON output."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()
EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".bmp", ".ppm"}


def image_paths(folder: str | Path) -> list[Path]:
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError(f"图片目录不存在：{folder}")
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS and p.is_file())
    if len(paths) < 2:
        raise ValueError("目录中至少需要两张受支持的图片。")
    return paths


def load_image(path: str | Path, max_size: int = 1200) -> tuple[np.ndarray, dict]:
    with Image.open(path) as source:
        im = ImageOps.exif_transpose(source).convert("RGB")
        original = im.size
        scale = min(1.0, max_size / max(original)) if max_size else 1.0
        if scale < 1:
            im = im.resize(
                tuple(max(1, round(v * scale)) for v in original), Image.Resampling.LANCZOS
            )
        rgb = np.asarray(im).copy()
    return rgb, {
        "name": Path(path).name,
        "original_size": list(original),
        "size": list(im.size),
        "scale_xy": [im.width / original[0], im.height / original[1]],
    }


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: str | Path, value) -> None:
    def default(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)

    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=default, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def save_rgb(path: str | Path, rgb: np.ndarray) -> None:
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(path)


def preview_matches(a, b, pa, pb, limit=90):
    height = max(a.shape[0], b.shape[0])
    canvas = np.zeros((height, a.shape[1] + b.shape[1], 3), np.uint8)
    canvas[: a.shape[0], : a.shape[1]] = a
    canvas[: b.shape[0], a.shape[1] :] = b
    ids = np.arange(len(pa))
    if len(ids) > limit:
        ids = ids[np.linspace(0, len(ids) - 1, limit).astype(int)]
    for k in ids:
        p = tuple(np.rint(pa[k]).astype(int))
        q = tuple(np.rint(pb[k] + [a.shape[1], 0]).astype(int))
        cv2.line(canvas, p, q, (64, 226, 163), 1, cv2.LINE_AA)
        cv2.circle(canvas, p, 3, (255, 196, 80), -1)
        cv2.circle(canvas, q, 3, (255, 196, 80), -1)
    return canvas


def code_state():
    """Record the actual source content as well as the repository revision."""
    import subprocess

    package = Path(__file__).resolve().parent
    source = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        source.update(path.relative_to(package).as_posix().encode())
        source.update(path.read_bytes())
    result = {"source_sha256": source.hexdigest()}
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=package, text=True, stderr=subprocess.DEVNULL
        ).strip()
        result["git_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"], cwd=package, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        result["git_commit"] = None
    return result
