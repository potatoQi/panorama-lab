"""Publish benchmark results and update README tables."""

import argparse
import csv
import json
import shutil
from pathlib import Path

from PIL import Image

from panorama.benchmark import summarize
from panorama.features import METHODS
from panorama.protocols import LEGACY, PROTOCOLS, SUPERPOINT, UDIS
from panorama.references import PAPERS, comparison_rows

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apap", default="runs/reproduce-apap")
    parser.add_argument("--hpatches", default="runs/reproduce-hpatches")
    parser.add_argument("--udis-d", default="runs/reproduce-udis-d")
    parser.add_argument("--course-prefix", default="reproduce")
    args = parser.parse_args()
    summaries = []
    for dataset, path in [
        ("apap", args.apap),
        ("hpatches", args.hpatches),
        ("udis-d", args.udis_d),
    ]:
        source = ROOT / path
        protocol = read(source / "protocol.json")
        required = {"apap": LEGACY, "hpatches": SUPERPOINT, "udis-d": UDIS}[dataset]
        if protocol.get("protocol_id") != required:
            raise ValueError(f"{dataset}: 评测协议不匹配。")
        if protocol["is_subset"]:
            raise ValueError("评测结果不完整。")
        rows = read(source / "summary.json")
        records = [json.loads(line) for line in (source / "records.jsonl").read_text().splitlines()]
        expected = {"apap": 27, "hpatches": 580, "udis-d": 1106}[dataset]
        methods = set(METHODS)
        if (
            protocol["pairs"] != expected
            or protocol["available_pairs"] != expected
            or set(protocol["methods"]) != methods
            or len(rows) != len(methods)
            or {r["method"] for r in rows} != methods
            or any(r["pairs"] != expected for r in rows)
            or len(records) != expected * len(methods)
            or any(
                len({r["case"] for r in records if r["method"] == m}) != expected for m in methods
            )
        ):
            raise ValueError(f"{dataset}: 缺少算法或评测样本。")
        destination = ROOT / "results" / dataset
        destination.mkdir(parents=True, exist_ok=True)
        for filename in ("summary.csv", "summary.json", "records.jsonl"):
            shutil.copy2(source / filename, destination / filename)
        # Keep reproducibility metadata; machine paths and run timestamps stay in runs/.
        published = {
            k: v
            for k, v in protocol.items()
            if k not in {"created_utc", "manifest", "seconds", "warning", "geometry", "photometry"}
        }
        published["evaluation"] = dict(protocol["evaluation"])
        published["evaluation"]["comparability"] = PROTOCOLS[required]["comparability"]
        (destination / "protocol.json").write_text(
            json.dumps(published, ensure_ascii=False, indent=2) + "\n"
        )
        summaries.extend(read(source / "summary.json"))
        if dataset == "hpatches":
            subsets = summarize(
                [
                    dict(r, dataset="illumination" if r["case"].startswith("i_") else "viewpoint")
                    for r in records
                ]
            )
            with (destination / "subsets.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, list(subsets[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(subsets)
    destination = ROOT / "results/course"
    destination.mkdir(parents=True, exist_ok=True)
    course = []
    for variant in METHODS:
        source = ROOT / "runs" / (args.course_prefix + "-" + variant)
        m = read(source / "manifest.json")
        if variant == "sift":
            with Image.open(source / "panorama.jpg") as image:
                image.thumbnail((2200, 1000))
                image.save(destination / "sift.jpg", quality=88)
        course.append(
            {
                "method": variant,
                "used": m["used_count"],
                "total": m["input_count"],
                "width": m["canvas"]["width"],
                "height": m["canvas"]["height"],
                "seconds": m["timing"]["total_seconds"],
            }
        )
    with (ROOT / "results/course.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, list(course[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(course)
    names = dict(METHODS, harris="Harris（手写）")

    def comparison(dataset, directory, metrics):
        measured = [r for r in summaries if r["dataset"] == dataset]
        protocol = read(ROOT / "results" / directory / "protocol.json")
        rows = comparison_rows(measured, protocol)
        headers = ["算法", "来源", *[label for _, label, _ in metrics], "估计成功", "秒/对 ↓"]
        formatted = []
        for row in rows:
            formatted.append(
                [
                    row["method"],
                    row["source"],
                    *[
                        format(row[key], fmt) if row.get(key) is not None else "—"
                        for key, _, fmt in metrics
                    ],
                    f"{row['succeeded']}/{row['pairs']}" if "succeeded" in row else "—",
                    f"{row['mean_seconds']:.2f}" if "mean_seconds" in row else "—",
                ]
            )
        return table(headers, formatted)

    notes = [
        f"论文报告来自 [{paper['title']}]({paper['url']})。{paper['note']}"
        for paper in (PAPERS[SUPERPOINT], PAPERS[UDIS])
    ]
    settings = "<details>\n<summary>实验设置</summary>\n\n" + "\n\n".join(notes) + "\n\n</details>"
    chunks = [
        settings,
        "### HPatches（完整 580 对，SuperPoint 2018 协议）",
        comparison(
            "HPatches",
            "hpatches",
            [
                ("h_accuracy_1px", "H-Acc@1px ↑", ".1%"),
                ("h_accuracy_3px", "H-Acc@3px ↑", ".1%"),
                ("h_accuracy_5px", "H-Acc@5px ↑", ".1%"),
            ],
        ),
        "### UDIS-D（完整测试集 1106 对，UDIS++ 2023 协议，失败用单位矩阵计分）",
        comparison(
            "UDIS-D", "udis-d", [("psnr_db", "PSNR (dB) ↑", ".2f"), ("ssim", "SSIM ↑", ".3f")]
        ),
        "### APAP（补充实验：完整 27 对相邻图像，非原论文 RMSE 协议）",
        comparison(
            "APAP",
            "apap",
            [
                ("overlap_psnr_db", "Overlap PSNR (dB) ↑", ".2f"),
                ("overlap_ssim", "Overlap SSIM ↑", ".3f"),
            ],
        ),
        "### 课程全景（28 张）",
        table(
            ["特征", "拼接图片数", "尺寸 (px)", "耗时 (s) ↓"],
            [
                [
                    names[r["method"]],
                    f"{r['used']}/{r['total']}",
                    f"{r['width']} × {r['height']}",
                    f"{r['seconds']:.1f}",
                ]
                for r in course
                if r["method"] in METHODS
            ],
        ),
        "![RootSIFT 课程全景](results/course/sift.jpg)",
    ]
    path = ROOT / "README.md"
    s = path.read_text()
    left = s.index("<!-- RESULTS_START -->") + len("<!-- RESULTS_START -->")
    right = s.index("<!-- RESULTS_END -->")
    path.write_text(s[:left] + "\n\n" + "\n\n".join(chunks) + "\n\n" + s[right:])
    print("Updated README and results/")


if __name__ == "__main__":
    main()
