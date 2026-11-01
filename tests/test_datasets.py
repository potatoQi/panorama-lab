"""Guard against incomplete or offset pairing in the upstream UDIS-D release."""

import importlib.util
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location(
    "dataset_script", Path(__file__).resolve().parents[1] / "scripts/datasets.py"
)
datasets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(datasets)


def test_udis_requires_complete_filename_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets, "ROOT", tmp_path)
    blob = BytesIO()
    Image.new("RGB", (8, 8)).save(blob, format="JPEG")
    for side in ("input1", "input2"):
        folder = tmp_path / "data/benchmarks/udis-d/testing" / side
        folder.mkdir(parents=True)
        for i in range(1, 1107):
            (folder / f"{i:06d}.jpg").write_bytes(blob.getvalue())
    manifest = datasets.make_udis()
    assert manifest["split"] == "testing"
    assert len(manifest["pairs"]) == 1106
    assert len({p["id"] for p in manifest["pairs"]}) == 1106
    assert all(Path(p["a"]).name == Path(p["b"]).name for p in manifest["pairs"])
    # The published archive lacks this image; do not silently shift the pairing.
    (tmp_path / "data/benchmarks/udis-d/testing/input1/000001.jpg").unlink()
    with pytest.raises(ValueError, match="000001.jpg"):
        datasets.make_udis()
