"""Download public archives and make explicit, reproducible pair manifests."""

import argparse
import itertools
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from panorama.io import digest, write_json

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "udis-d": "https://drive.usercontent.google.com/download?id=1unnBdPg9Vi12-1xXoUdzMub_XUq7yZz0&export=download&confirm=t",
    "apap": "https://cs.adelaide.edu.au/~tjchin/apap/files/images.zip",
    "apap2": "https://cs.adelaide.edu.au/~tjchin/apap/files/images2.zip",
    "apap3": "https://cs.adelaide.edu.au/~tjchin/apap/files/images3.zip",
    "hpatches": "https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/hpatches-sequences-release.zip",
}

UDIS_PATCH_URL = "https://drive.usercontent.google.com/download?id=1M_-lqNdICT4tM3CWLK2WOcPCIzeQdc4d&export=download&confirm=t"
UDIS_PATCH_SHA256 = "bbf99e1817f20f706e3fa2b68f2adaca0ae49dbb8f9a2ccf2410ebbc878f0d18"


def restore_udis_image(destination):
    """Restore the image missing from the author's ZIP using the UniStitch release."""
    patch = ROOT / "data/archives/udis-d-input1-000001.jpg"
    if not patch.exists():
        response = requests.get(UDIS_PATCH_URL, timeout=(20, 90))
        response.raise_for_status()
        temporary = patch.with_suffix(".partial")
        temporary.write_bytes(response.content)
        if digest(temporary) != UDIS_PATCH_SHA256:
            raise ValueError("UDIS-D 补发图片 SHA-256 不一致。")
        temporary.replace(patch)
    if digest(patch) != UDIS_PATCH_SHA256:
        raise ValueError("UDIS-D 补发图片 SHA-256 不一致。")
    shutil.copy2(patch, destination / "testing/input1/000001.jpg")
    return patch


def extract(archive, destination):
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            target = (destination / item.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("压缩包包含越界路径。")
            if "__MACOSX" not in item.filename and not Path(item.filename).name.startswith("."):
                z.extract(item, destination)


def download(name, mirror=False):
    archive = ROOT / "data/archives" / (name + ".zip")
    if name == "hpatches" and (ROOT / "data/archives/hpatches-mirror.zip").exists():
        archive = ROOT / "data/archives/hpatches-mirror.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists() or not zipfile.is_zipfile(archive):
        url = SOURCES[name]
        if mirror and name == "hpatches":
            url = url.replace("huggingface.co", "hf-mirror.com")
        temporary = archive.with_suffix(".partial")
        with requests.get(url, stream=True, timeout=(20, 90)) as response:
            response.raise_for_status()
            with temporary.open("wb") as f:
                for block in response.iter_content(1 << 20):
                    f.write(block)
        if not zipfile.is_zipfile(temporary):
            raise ValueError("下载内容不是完整 ZIP。")
        temporary.replace(archive)
    if name == "hpatches":
        expected = "99cf7e1ca167896eb4ca2fe3d903beff6566243a3c0c8e07a13a2596648ad670"
        if digest(archive) != expected:
            raise ValueError("HPatches SHA-256 与作者数据仓库 LFS 对象不一致。")
    if (
        name == "udis-d"
        and digest(archive) != "7e7b6c0b930a18e177b83409680f57077f7d92a84d3f66a26c81a03d4eea3478"
    ):
        raise ValueError("UDIS-D 测试包 SHA-256 与记录版本不一致。")
    destination = ROOT / "data/benchmarks" / name
    extract(archive, destination)
    if name == "udis-d":
        restore_udis_image(destination)
    return archive, destination


def make_apap():
    folders = ROOT / "data/benchmarks"
    groups = []
    first = folders / "apap/images"
    for prefix in ("DSC029", "DSC030", "P101"):
        groups.append(sorted(first.glob(prefix + "*.JPG")))
    for p in sorted((folders / "apap2").glob("case*")):
        if p.is_dir():
            groups.append(sorted(p.glob("*.JPG")))
    groups.append(sorted((folders / "apap3").glob("*.JPG")))
    pairs = []
    for group in groups:
        for a, b in itertools.pairwise(group):
            pairs.append(
                {
                    "dataset": "APAP",
                    "id": a.parent.name + "-" + a.stem + "-" + b.stem,
                    "a": str(a.relative_to(folders)),
                    "b": str(b.relative_to(folders)),
                }
            )
    return {
        "root": "benchmarks",
        "source": "https://cs.adelaide.edu.au/~tjchin/apap/",
        "protocol": "All adjacent image pairs in the 8 sequences from the three official archives; own pair protocol, not an official leaderboard split.",
        "pairs": pairs,
    }


def make_hpatches():
    folders = ROOT / "data/benchmarks"
    pairs = []
    for path in sorted((folders / "hpatches").rglob("1.ppm")):
        for i in range(2, 7):
            pairs.append(
                {
                    "dataset": "HPatches",
                    "id": path.parent.name + f"-1-{i}",
                    "a": str(path.relative_to(folders)),
                    "b": str((path.parent / f"{i}.ppm").relative_to(folders)),
                    "homography": np.loadtxt(path.parent / f"H_1_{i}").tolist(),
                }
            )
    return {
        "root": "benchmarks",
        "source": "https://github.com/hpatches/hpatches-dataset",
        "protocol": "All 116 sequences, reference image 1 against 2..6 (580 pairs), no sequence exclusion. Homography estimation protocol, not the original descriptor-patch benchmark.",
        "pairs": pairs,
    }


def make_udis():
    folders = ROOT / "data/benchmarks"
    testing = folders / "udis-d/testing"
    expected = {f"{i:06d}.jpg" for i in range(1, 1107)}
    for side in ("input1", "input2"):
        names = {p.name for p in (testing / side).glob("*.jpg")}
        if names != expected:
            raise ValueError(
                f"UDIS-D {side} 必须包含完整的 1106 张测试图片，缺少 {sorted(expected - names)[:5]}。"
            )
    pairs = []
    for name in sorted(expected):
        a, b = testing / "input1" / name, testing / "input2" / name
        for path in (a, b):
            with Image.open(path) as im:
                im.verify()
        pairs.append(
            {
                "dataset": "UDIS-D",
                "id": Path(name).stem,
                "a": str(a.relative_to(folders)),
                "b": str(b.relative_to(folders)),
                "sha256_a": digest(a),
                "sha256_b": digest(b),
            }
        )
    return {
        "root": "benchmarks",
        "source": "https://github.com/nie-lang/UnsupervisedDeepImageStitching",
        "supplement_source": "https://github.com/MmelodYy/UniStitch#-dataset",
        "split": "testing",
        "expected_pairs": 1106,
        "protocol": "Complete official testing split: 1106 filename-matched pairs. Missing input1/000001.jpg restored from UniStitch authors' supplement. No training images or test exclusions. Evaluation protocol is chosen by the benchmark runner.",
        "pairs": pairs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["apap", "hpatches", "udis-d", "all"])
    parser.add_argument("--mirror", action="store_true", help="仅 HPatches 使用 hf-mirror 传输镜像")
    args = parser.parse_args()
    names = ["apap", "apap2", "apap3"] if args.dataset in {"apap", "all"} else []
    if args.dataset in {"hpatches", "all"}:
        names.append("hpatches")
    if args.dataset in {"udis-d", "all"}:
        names.append("udis-d")
    checksums = []
    for name in names:
        archive, _ = download(name, args.mirror)
        checksums.append(
            {
                "name": name,
                "source": SOURCES[name],
                "sha256": digest(archive),
                "bytes": archive.stat().st_size,
            }
        )
    if "udis-d" in names:
        patch = ROOT / "data/archives/udis-d-input1-000001.jpg"
        checksums.append(
            {
                "name": "udis-d-input1-000001",
                "source": UDIS_PATCH_URL,
                "sha256": digest(patch),
                "bytes": patch.stat().st_size,
                "provenance": "https://github.com/MmelodYy/UniStitch#-dataset",
            }
        )
    data = ROOT / "data"
    makers = {"apap": make_apap, "hpatches": make_hpatches, "udis-d": make_udis}
    for name, make_manifest in makers.items():
        if args.dataset in {name, "all"}:
            manifest = make_manifest()
            if not manifest["pairs"]:
                raise ValueError("没有找到数据集图像。")
            write_json(data / f"{name}.json", manifest)
            print(name, len(manifest["pairs"]), "pairs", flush=True)
    path = ROOT / "results/datasets.json"
    old = json.loads(path.read_text()) if path.exists() else []
    new = {row["name"]: row for row in old + checksums}
    write_json(path, list(new.values()))


if __name__ == "__main__":
    main()
