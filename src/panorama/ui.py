"""Compact research workspace for stitching and experiment inspection."""

import json
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from panorama.features import METHODS
from panorama.io import EXTENSIONS, image_paths
from panorama.pipeline import Config, stitch
from panorama.protocols import DEFAULT_PROTOCOL, LEGACY, PROTOCOL_LABELS
from panorama.references import PAPERS, comparison_rows
from panorama.viewer import viewer_html

ROOT = Path(__file__).resolve().parents[2]
FEATURES = {METHODS[key]: key for key in ("sift", "sift-plain", "xfeat", "harris", "orb")}
DATASETS = {"apap": "APAP", "hpatches": "HPatches", "udis-d": "UDIS-D"}
METRICS = {
    "h_accuracy_1px": "H-Acc@1px",
    "h_accuracy_3px": "H-Acc@3px",
    "h_accuracy_5px": "H-Acc@5px",
    "psnr_db": "PSNR (dB)",
    "ssim": "SSIM",
    "overlap_psnr_db": "重叠 PSNR (dB)",
    "overlap_ssim": "重叠 SSIM",
    "mean_seconds": "耗时 (s/对)",
}

st.set_page_config(page_title="图像拼接", layout="wide", initial_sidebar_state="expanded")
# Keep the interface dense and functional: flat controls, restrained purple, no slogans.
st.markdown(
    """<style>
.block-container{padding:3rem 1.25rem 1.5rem;max-width:1600px}
[data-testid="stHeader"]{height:32px}
[data-testid="stSidebar"]{border-right:1px solid #ded9e3;box-shadow:none;position:relative!important;
 min-width:260px!important;max-width:260px!important;width:260px!important;transform:none!important}
[data-testid="stSidebarCollapseButton"],[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarResizeHandle"]{display:none!important}
[data-testid="stSidebarUserContent"]{padding:.5rem .9rem}
[data-testid="stSidebarHeader"]{height:32px;min-height:32px;padding:.25rem .5rem}
[data-testid="stVerticalBlock"]{gap:12px}
h3{color:#5C307D!important;font-size:1.15rem!important;font-weight:600!important;padding:0 0 .35rem!important}
h4{font-size:1rem!important;font-weight:600!important;line-height:1.4!important;padding:0!important}
/* Streamlit offsets heading markdown by -1rem; neutralize it for compact headings. */
[data-testid="stMarkdownContainer"]:has(> h3),
[data-testid="stMarkdownContainer"]:has(> h4){margin-bottom:0!important}
[data-testid="stSelectbox"] [data-testid="stWidgetLabel"],
[data-testid="stMultiSelect"] [data-testid="stWidgetLabel"],
[data-testid="stTextInput"] [data-testid="stWidgetLabel"],
[data-testid="stNumberInput"] [data-testid="stWidgetLabel"],
[data-testid="stFileUploader"] [data-testid="stWidgetLabel"]{margin-bottom:6px}
[data-testid="stWidgetLabel"] p{font-size:.88rem}
[data-testid="stTabs"] [role="tablist"]{gap:1.5rem;border-bottom:1px solid #ded9e3}
[data-testid="stTabs"] [role="tab"]{height:32px;padding:0 2px;font-size:.92rem}
[data-testid="stExpander"]{border-radius:0!important}
[data-testid="stButton"] button,[data-testid="stDownloadButton"] button{min-height:30px;padding:3px 10px}
[data-testid="stElementToolbar"]{border-radius:0}
.stAppDeployButton{display:none}
.st-key-panorama-flat{position:relative}
.st-key-panorama-flat [data-testid="stElementContainer"]:has([data-testid="stDownloadButton"]){
 position:absolute;top:8px;right:8px;width:auto;z-index:10}
.st-key-panorama-flat [data-testid="stDownloadButton"] button{
 background:#5c307ddd;color:white;border:1px solid #ffffff44;border-radius:0;padding:5px 9px}
.st-key-panorama-flat [data-testid="stElementToolbar"]{right:72px}
.run-stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));margin:0 0 .2rem;
 border-top:1px solid #ded9e3;border-bottom:1px solid #ded9e3;padding:9px 0;gap:12px}
.run-stats div{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}
.run-stats dt{font-size:.85rem;color:#39343d}.run-stats dd{margin:0;font-size:1rem;font-weight:600;font-variant-numeric:tabular-nums}
[data-testid="stAppViewContainer"]:has([data-testid="stSidebar"]) [data-testid="stMain"]{margin-left:0!important;min-width:0;width:100%!important}
@media(max-width:1000px){.run-stats{grid-template-columns:repeat(2,minmax(0,1fr));gap:7px}}
</style>""",
    unsafe_allow_html=True,
)


def stitching_panel():
    with st.sidebar:
        st.markdown("#### 输入与参数")
        source = st.radio(
            "输入",
            ["本地目录", "上传图片"],
            horizontal=True,
            label_visibility="collapsed",
            key="stitch_source",
        )
        uploaded = None
        if source == "本地目录":
            folder = st.text_input("本地目录", key="stitch_folder")
        else:
            uploaded = st.file_uploader(
                "图片",
                type=sorted(extension.lstrip(".") for extension in EXTENSIONS),
                accept_multiple_files=True,
            )
            folder = None
        feature_label = st.selectbox("特征算法", list(FEATURES), key="stitch_feature")
        projection_label = st.selectbox("投影方式", ["球面", "平面"], key="stitch_projection")
        size = st.selectbox("输入长边上限 (px)", [600, 800, 1200, 1600, 2000], key="stitch_size")
        with st.expander("高级设置"):
            blend_label = st.selectbox("融合方式", ["多频段", "羽化"], key="stitch_blend")
            count = st.number_input(
                "每张图特征点上限", 100, 10000, value=None, step=100, key="stitch_count"
            )
            exposure = st.checkbox("曝光补偿", key="stitch_exposure")
            optimize = st.checkbox(
                "全局优化", disabled=projection_label == "平面", key="stitch_optimize"
            )
        run = st.button("开始拼接", type="primary", width="stretch")

    if run:
        try:
            if source == "上传图片":
                if not uploaded or len(uploaded) < 2:
                    raise ValueError("请至少上传两张图片。")
                destination = ROOT / "data/uploads" / uuid.uuid4().hex[:12]
                destination.mkdir(parents=True)
                for i, upload in enumerate(uploaded):
                    (destination / f"{i:03d}_{Path(upload.name).name}").write_bytes(
                        upload.getbuffer()
                    )
                folder = str(destination)
            image_paths(folder)
            output = (
                ROOT
                / "runs"
                / (
                    datetime.now().strftime("%Y%m%d-%H%M%S")
                    + "-"
                    + FEATURES[feature_label]
                    + "-"
                    + uuid.uuid4().hex[:4]
                )
            )
            config = Config(
                feature=FEATURES[feature_label],
                projection="spherical" if projection_label == "球面" else "planar",
                blending="multiband" if blend_label == "多频段" else "feather",
                max_size=size,
                max_features=count,
                exposure=exposure,
                optimize=optimize,
            )
            bar = st.progress(0, text="读取图片")
            stitch(
                folder,
                output,
                config,
                lambda fraction, text: bar.progress(fraction, text=text.replace(" · ", "，")),
            )
            bar.empty()
            st.session_state["selected_run"] = str(output)
            st.success("拼接完成")
        except (ValueError, OSError, RuntimeError) as error:
            st.error(str(error))

    available = sorted(
        (ROOT / "runs").glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if available:
        records = {str(p.parent): json.loads(p.read_text()) for p in available}
        options = list(records)

        def run_label(value):
            record = records[value]
            timestamp = datetime.fromisoformat(record["created_utc"]).astimezone()
            config = record["config"]
            method = METHODS[config["feature"]]
            projection = "球面" if config["projection"] == "spherical" else "平面"
            suffix = " / 关闭全局优化" if projection == "球面" and not config["optimize"] else ""
            return (
                f"{timestamp:%m-%d %H:%M:%S %Z} / {method} / {projection}"
                f" / {record['input_count']} 张{suffix}"
            )

        current = st.session_state.get("selected_run", options[0])
        selected = st.selectbox(
            "历史运行",
            options,
            index=options.index(current) if current in options else 0,
            format_func=run_label,
        )
        st.session_state["selected_run"] = selected
        output = Path(selected)
        manifest = records[selected]
        stats = [
            ("已拼接图片", f"{manifest['used_count']}/{manifest['input_count']}"),
            ("输出尺寸", f"{manifest['canvas']['width']} × {manifest['canvas']['height']} px"),
            ("总耗时", f"{manifest['timing']['total_seconds']:.1f} s"),
        ]
        st.markdown(
            '<dl class="run-stats">'
            + "".join(f"<div><dt>{label}</dt><dd>{value}</dd></div>" for label, value in stats)
            + "</dl>",
            unsafe_allow_html=True,
        )
    else:
        output = manifest = None
        st.info("暂无结果")

    if output:
        spherical = manifest["canvas"]["projection"] == "spherical"
        mode = (
            st.radio(
                "查看方式", ["交互全景", "展开图"], horizontal=True, label_visibility="collapsed"
            )
            if spherical
            else "展开图"
        )
        if mode == "交互全景":
            st.iframe(viewer_html(output / "panorama.jpg", manifest["canvas"]), height=425)
        else:
            with st.container(key="panorama-flat"):
                st.image(str(output / "panorama.jpg"), width="stretch")
                st.download_button(
                    "下载", (output / "panorama.png").read_bytes(), "panorama.png", "image/png"
                )
        previews = sorted((output / "matches").glob("*.jpg"))
        if previews:
            with st.expander("特征匹配示例"):
                names = [image["name"] for image in manifest["images"]]

                def pair_label(path):
                    i, j = map(int, path.stem.split("-"))
                    return f"{names[i]} / {names[j]}"

                chosen = st.selectbox("图像对", previews, format_func=pair_label)
                st.image(str(chosen), width="stretch")


def benchmark_panel():
    process = st.session_state.get("benchmark_process")
    running = process is not None and process.poll() is None
    manifests = [
        ROOT / "data" / f"{name}.json"
        for name in DATASETS
        if (ROOT / "data" / f"{name}.json").exists()
    ]
    with st.sidebar:
        st.markdown("#### 评测设置")
        if manifests:
            dataset = st.selectbox(
                "数据集",
                manifests,
                format_func=lambda p: DATASETS[p.stem] + ("（补充）" if p.stem == "apap" else ""),
                key="bench_dataset",
            )
            methods = st.multiselect("特征算法", list(FEATURES), key="bench_methods")
            profile = DEFAULT_PROTOCOL[DATASETS[dataset.stem]]
            st.selectbox("评测协议", [PROTOCOL_LABELS[profile]], disabled=True)
            if profile == LEGACY:
                size = st.selectbox(
                    "输入长边上限 (px)", [600, 800, 1200, 1600, 2000], key="bench_size"
                )
            run = st.button(
                "开始评测", type="primary", width="stretch", disabled=running or not methods
            )
            if run:
                target = (
                    ROOT
                    / "runs"
                    / (
                        f"benchmark-ui-{dataset.stem}-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
                    )
                )
                target.mkdir(parents=True)
                with (target / "process.log").open("w") as log:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "panorama",
                            "benchmark",
                            str(dataset),
                            "--output",
                            str(target),
                            "--methods",
                            *[FEATURES[m] for m in methods],
                            "--protocol",
                            profile,
                            *(["--max-size", str(size)] if profile == LEGACY else []),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"),
                        cwd=ROOT,
                    )
                st.session_state["benchmark_process"] = process
                st.session_state["benchmark_output"] = str(target)
                st.rerun()
        else:
            st.info("请先下载评测数据集")

    if process is not None:
        status = process.poll()
        if status is None:
            st.write("评测运行中")
            if st.button("刷新进度"):
                st.rerun()
        elif status != 0:
            st.error("评测失败")
            log = Path(st.session_state["benchmark_output"]) / "process.log"
            st.code("\n".join(log.read_text().splitlines()[-8:]), language=None)
        elif st.session_state.get("benchmark_completed") != st.session_state["benchmark_output"]:
            st.session_state["benchmark_completed"] = st.session_state["benchmark_output"]
            st.session_state["benchmark_result"] = str(
                Path(st.session_state["benchmark_output"]) / "summary.csv"
            )

    baselines = sorted((ROOT / "results").glob("*/summary.csv"))
    history = sorted(
        (ROOT / "runs").glob("*/summary.csv"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    paths = baselines + history
    if not paths:
        st.info("暂无评测结果")
        return

    def result_label(value):
        path = Path(value)
        protocol = json.loads((path.parent / "protocol.json").read_text())
        label = PROTOCOL_LABELS.get(protocol.get("protocol_id", LEGACY), "自定义 v1")
        if path.parent.parent == ROOT / "results":
            if path.parent.name == "apap":
                label = "补充实验"
            return DATASETS.get(path.parent.name, path.parent.name) + " / " + label
        timestamp = datetime.fromisoformat(protocol["created_utc"]).astimezone()
        dataset_name = pd.read_csv(path, nrows=1)["dataset"].iloc[0]
        return f"{timestamp:%m-%d %H:%M:%S %Z} / {dataset_name} / {label}"

    options = list(map(str, paths))
    current = st.session_state.get(
        "benchmark_result",
        str(ROOT / "results" / (dataset.stem if manifests else "hpatches") / "summary.csv"),
    )
    st.session_state["benchmark_result"] = current if current in options else options[0]
    selected = st.selectbox("历史评测", options, format_func=result_label, key="benchmark_result")
    frame = pd.read_csv(selected)
    protocol = json.loads((Path(selected).parent / "protocol.json").read_text())
    frame = pd.DataFrame(comparison_rows(frame.to_dict("records"), protocol))
    frame["completed"] = frame.apply(
        lambda r: f"{int(r.succeeded)}/{int(r.pairs)}" if pd.notna(r.succeeded) else "—", axis=1
    )
    metrics = (
        ["h_accuracy_1px", "h_accuracy_3px", "h_accuracy_5px"]
        if "h_accuracy_3px" in frame and frame["h_accuracy_3px"].notna().any()
        else (["psnr_db", "ssim"] if "psnr_db" in frame else ["overlap_psnr_db", "overlap_ssim"])
    )
    columns = ["method", "source", "completed", *metrics, "mean_seconds"]
    labels = {"method": "算法", "source": "来源", "completed": "成功对数", **METRICS}
    formats = {labels[m]: "{:.1%}" if m.startswith("h_accuracy_") else "{:.3f}" for m in metrics}
    display = frame[columns].rename(columns=labels)
    # Streamlit renders numeric nulls as "None" even with Styler.na_rep.
    display[labels["mean_seconds"]] = frame["mean_seconds"].map(
        lambda value: "—" if pd.isna(value) else f"{value:.3f}"
    )
    st.dataframe(
        display.style.format(formats, na_rep="—"),
        hide_index=True,
        width="stretch",
    )
    st.download_button(
        "下载评测表",
        frame[columns + (["reference"] if "reference" in frame else [])]
        .rename(columns=labels)
        .to_csv(index=False)
        .encode("utf-8-sig"),
        "benchmark.csv",
        "text/csv",
    )
    if "论文报告" in frame["source"].values:
        paper = PAPERS[protocol["protocol_id"]]
        with st.expander("论文对照说明"):
            st.markdown(f"[{paper['title']}]({paper['url']})。{paper['note']}")


# Defaults live in session state so switching tabs preserves each task's settings.
defaults = Config()
for key, value in {
    "stitch_folder": str(ROOT / "data/course/panorama"),
    "stitch_size": defaults.max_size,
    "stitch_count": defaults.max_features,
    "stitch_exposure": defaults.exposure,
    "stitch_optimize": defaults.optimize,
    "bench_size": 800,
    "bench_methods": list(FEATURES),
}.items():
    st.session_state.setdefault(key, value)

# Keep each task's settings when its widgets are temporarily hidden by the other tab.
for key in list(st.session_state):
    if key.startswith(("stitch_", "bench_")) or key == "benchmark_result":
        st.session_state[key] = st.session_state[key]

stitch_tab, benchmark_tab = st.tabs(["图像拼接", "数据集评测"], key="workspace", on_change="rerun")
if stitch_tab.open:
    with stitch_tab:
        stitching_panel()
if benchmark_tab.open:
    with benchmark_tab:
        benchmark_panel()
