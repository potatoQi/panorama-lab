"""Pinned official XFeat sparse extractor, CPU inference and verified weights."""

import hashlib
import json
import os
import urllib.request
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

SOURCE = json.loads((Path(__file__).parent / "_vendor/xfeat/SOURCE.json").read_text())
CACHE = Path(os.environ.get("PANORAMA_CACHE", Path(__file__).resolve().parents[2] / ".cache"))


def prepare_weights():
    path = CACHE / "xfeat/xfeat.pt"
    expected = SOURCE["files"]["weights/xfeat.pt"]
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"XFeat 权重校验失败：{path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://raw.githubusercontent.com/verlab/accelerated_features/"
        + SOURCE["commit"]
        + "/weights/xfeat.pt"
    )
    # Only fixed, verified model weights are downloaded; inference code is vendored.
    data = urllib.request.urlopen(url, timeout=120).read()
    if hashlib.sha256(data).hexdigest() != expected:
        raise RuntimeError("XFeat 下载权重的 SHA-256 不符。")
    with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(data)
    os.replace(tmp.name, path)
    return path


@lru_cache(maxsize=1)
def model():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("XFeat 需要 PyTorch，请按 README 安装依赖。") from error
    from ._vendor.xfeat.xfeat import XFeat

    torch.set_num_threads(1)
    net = XFeat()
    net.net.load_state_dict(torch.load(prepare_weights(), map_location="cpu", weights_only=True))
    return net


def extract(rgb, max_features):
    import torch

    # Official sparse inference; keep its NMS, score threshold and 64-D descriptors.
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].float() / 255.0
    result = model().detectAndCompute(tensor, top_k=max_features, detection_threshold=0.05)[0]
    return (
        result["keypoints"].cpu().numpy().astype(np.float64),
        result["descriptors"].cpu().numpy().astype(np.float32),
    )


def provenance():
    import torch

    prepare_weights()
    return {
        "repository": SOURCE["repository"],
        "commit": SOURCE["commit"],
        "weights_sha256": SOURCE["files"]["weights/xfeat.pt"],
        "torch": torch.__version__,
        "device": "cpu",
        "threads": 1,
        "variant": "sparse detectAndCompute; no XFeat* or LighterGlue",
        "detection_threshold": 0.05,
        "timing": "Model initialization and worker warmup excluded; pair I/O and inference included.",
    }


if __name__ == "__main__":
    print(prepare_weights())
