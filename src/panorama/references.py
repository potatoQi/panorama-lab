"""Paper results for benchmark comparison."""

from .features import METHODS
from .protocols import PROTOCOLS, SUPERPOINT, UDIS

PAPERS = {
    SUPERPOINT: {
        "title": "SuperPoint 2018，Table 4",
        "url": "https://arxiv.org/pdf/1712.07629",
        "pairs": 580,
        "dataset": "HPatches",
        "note": PROTOCOLS[SUPERPOINT]["comparability"],
        "rows": [
            {
                "method": "ORB",
                "h_accuracy_1px": 0.150,
                "h_accuracy_3px": 0.395,
                "h_accuracy_5px": 0.538,
            },
            {
                "method": "SIFT",
                "h_accuracy_1px": 0.424,
                "h_accuracy_3px": 0.676,
                "h_accuracy_5px": 0.759,
            },
            {
                "method": "SuperPoint",
                "h_accuracy_1px": 0.310,
                "h_accuracy_3px": 0.684,
                "h_accuracy_5px": 0.829,
            },
        ],
    },
    UDIS: {
        "title": "UDIS++ 2023，Table 1",
        "url": "https://arxiv.org/pdf/2302.08207",
        "pairs": 1106,
        "dataset": "UDIS-D",
        "note": PROTOCOLS[UDIS]["comparability"],
        "rows": [
            {"method": "SIFT + RANSAC", "psnr_db": 23.27, "ssim": 0.779},
            {"method": "APAP", "psnr_db": 23.79, "ssim": 0.794},
            {"method": "UDIS++", "psnr_db": 25.43, "ssim": 0.838},
        ],
    },
}


def comparison_rows(rows, protocol):
    order = {method: index for index, method in enumerate(METHODS)}
    rows = sorted(rows, key=lambda row: order.get(row["method"], len(order)))
    result = [
        dict(r, method=METHODS.get(r["method"], r["method"]), source="本地实测") for r in rows
    ]
    paper = PAPERS.get(protocol.get("protocol_id"))
    if (
        paper
        and not protocol.get("is_subset", True)
        and protocol.get("pairs") == paper["pairs"]
        and all(r["dataset"] == paper["dataset"] and r["pairs"] == paper["pairs"] for r in rows)
    ):
        result.extend(dict(r, source="论文报告", reference=paper["url"]) for r in paper["rows"])
    return result
