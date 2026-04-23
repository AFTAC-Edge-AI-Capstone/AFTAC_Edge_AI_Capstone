import hashlib
import json
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:
    ort = None


PROJECT_DIR = Path(__file__).resolve().parent
VISION_DIR = PROJECT_DIR / "EfficientNet_Lite2"
UNQUANTIZED_MODEL = VISION_DIR / "efficientnet_lite2_unquantized.onnx"
QUANTIZED_MODEL = VISION_DIR / "efficientnet_lite2_locally_quantized.onnx"
BOARD_BENCHMARK = VISION_DIR / "results" / "developer_cloud_benchmark_stm32n6570dk.csv"
MANIFEST = VISION_DIR / "imagenet_val_manifest.csv"
CLASS_INDEX = VISION_DIR / "imagenet_class_index.json"
IMAGE_SIZE = 260
MODEL_OPTIONS = {
    "Quantized INT8/QDQ ONNX": QUANTIZED_MODEL,
    "FP32 ONNX": UNQUANTIZED_MODEL,
}

MODEL_COLORS = {
    "FP32 ONNX": "#536878",
    "INT8/QDQ ONNX": "#b4533c",
    "CSWin-Tiny Reference": "#2f855a",
    "EfficientNet-Lite2 INT8": "#b4533c",
}


def bytes_to_mb(value):
    return value / 1_000_000


def format_bytes(value):
    if value is None:
        return "Not found"

    return f"{value:,} B ({bytes_to_mb(value):.2f} MB)"


def model_size(path, fallback_bytes):
    if path.exists():
        return path.stat().st_size

    return fallback_bytes


@st.cache_data
def load_class_index():
    with CLASS_INDEX.open("r") as file:
        class_index = json.load(file)

    labels = []
    for model_idx in range(len(class_index)):
        wnid, class_name = class_index[str(model_idx)]
        labels.append(
            {
                "index": model_idx,
                "wnid": wnid,
                "class": class_name.replace("_", " "),
            }
        )

    return labels


@st.cache_resource
def load_onnx_session(model_path):
    if ort is None:
        raise ImportError("onnxruntime is not installed.")

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]

    return {
        "session": session,
        "input_name": input_meta.name,
        "output_name": output_meta.name,
        "input_shape": input_meta.shape,
        "input_type": input_meta.type,
    }


@st.cache_data
def load_board_benchmark():
    if not BOARD_BENCHMARK.exists():
        return {}

    benchmark = pd.read_csv(BOARD_BENCHMARK)
    if benchmark.empty:
        return {}

    row = benchmark.iloc[0]
    latency_ms = pd.to_numeric(row.get("duration_ms"), errors="coerce")
    cycles = pd.to_numeric(row.get("cycles"), errors="coerce")
    total_ram = pd.to_numeric(row.get("total_ram_bytes"), errors="coerce")
    internal_ram = pd.to_numeric(row.get("internal_ram"), errors="coerce")
    external_ram = pd.to_numeric(row.get("external_ram"), errors="coerce")

    runtime_memory = total_ram
    if pd.isna(runtime_memory) or runtime_memory == 0:
        runtime_memory = 0
        for value in (internal_ram, external_ram):
            if not pd.isna(value):
                runtime_memory += value

    return {
        "board": row.get("board", "STM32N6570-DK"),
        "created_at": row.get("createdAt", ""),
        "latency_ms": float(latency_ms) if not pd.isna(latency_ms) else 49.08,
        "cycles": int(cycles) if not pd.isna(cycles) else None,
        "runtime_memory_bytes": int(runtime_memory) if not pd.isna(runtime_memory) else None,
        "state": row.get("state", ""),
    }


@st.cache_data
def load_manifest_summary():
    if not MANIFEST.exists():
        return None, None

    manifest = pd.read_csv(MANIFEST)
    summary = {
        "images": len(manifest),
        "synsets": manifest["synset"].nunique(),
    }
    return summary, manifest.head(10)


def build_host_results():
    fp32_size = model_size(UNQUANTIZED_MODEL, 24_353_308)
    int8_size = model_size(QUANTIZED_MODEL, 6_880_048)

    return pd.DataFrame(
        [
            {
                "Model": "FP32 ONNX",
                "Format": "Full precision",
                "Top-1 Accuracy (%)": 76.09,
                "Top-5 Accuracy (%)": 92.93,
                "Throughput (img/s)": 47.10,
                "ONNX Size (B)": fp32_size,
                "ONNX Size (MB)": bytes_to_mb(fp32_size),
                "STM32N6570-DK Deployable": "No",
            },
            {
                "Model": "INT8/QDQ ONNX",
                "Format": "Static PTQ, QDQ graph",
                "Top-1 Accuracy (%)": 76.01,
                "Top-5 Accuracy (%)": 92.96,
                "Throughput (img/s)": 134.65,
                "ONNX Size (B)": int8_size,
                "ONNX Size (MB)": bytes_to_mb(int8_size),
                "STM32N6570-DK Deployable": "Yes",
            },
        ]
    )


def build_reference_results():
    return pd.DataFrame(
        [
            {
                "Model": "CSWin-Tiny Reference",
                "Top-1 Accuracy (%)": 82.82,
                "Top-5 Accuracy (%)": 96.30,
                "Host Throughput (img/s)": 6.13,
                "Size / Footprint (MB)": 87.461,
            },
            {
                "Model": "EfficientNet-Lite2 INT8",
                "Top-1 Accuracy (%)": 76.01,
                "Top-5 Accuracy (%)": 92.96,
                "Host Throughput (img/s)": 134.65,
                "Size / Footprint (MB)": bytes_to_mb(model_size(QUANTIZED_MODEL, 6_880_048)),
            },
        ]
    )


def input_size_from_shape(input_shape):
    if len(input_shape) == 4:
        if isinstance(input_shape[1], int) and isinstance(input_shape[2], int):
            return input_shape[1], input_shape[2]
        if isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
            return input_shape[2], input_shape[3]

    return IMAGE_SIZE, IMAGE_SIZE


def preprocess_image(image, input_shape):
    height, width = input_size_from_shape(input_shape)
    resized = image.convert("RGB").resize((width, height))
    image_array = np.asarray(resized).astype(np.float32)
    image_array /= 255.0
    image_array = (image_array - 0.5) / 0.5
    batch = np.expand_dims(image_array, axis=0)

    if len(input_shape) == 4 and input_shape[1] == 3:
        batch = np.transpose(batch, (0, 3, 1, 2))

    return batch, resized


def normalize_scores(raw_scores):
    scores = np.asarray(raw_scores).reshape(-1).astype(np.float32)

    if scores.min() < 0 or scores.max() > 1 or not np.isclose(scores.sum(), 1.0, rtol=0.05, atol=0.05):
        shifted = scores - np.max(scores)
        exp_scores = np.exp(shifted)
        scores = exp_scores / np.sum(exp_scores)

    return scores


def run_image_inference(image, model_path, top_k=10):
    model = load_onnx_session(str(model_path))
    batch, resized = preprocess_image(image, model["input_shape"])

    start = time.perf_counter()
    outputs = model["session"].run([model["output_name"]], {model["input_name"]: batch})
    elapsed_ms = (time.perf_counter() - start) * 1000

    scores = normalize_scores(outputs[0])
    labels = load_class_index()
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, class_index in enumerate(top_indices, start=1):
        label = labels[int(class_index)]
        results.append(
            {
                "Rank": rank,
                "Class": label["class"],
                "WordNet ID": label["wnid"],
                "Confidence": float(scores[class_index] * 100),
            }
        )

    return pd.DataFrame(results), elapsed_ms, batch.shape, resized


def render_inference_tab():
    if ort is None:
        st.error("ONNX Runtime is required for image inference. Install `onnxruntime` in the app environment.")
        return

    if not CLASS_INDEX.exists():
        st.error(f"ImageNet class index was not found at `{CLASS_INDEX}`.")
        return

    model_choice = st.selectbox("ONNX model", options=list(MODEL_OPTIONS.keys()), index=0)
    model_path = MODEL_OPTIONS[model_choice]
    top_k = st.slider("Top predictions", min_value=1, max_value=10, value=5)

    if not model_path.exists():
        st.error(f"Model file was not found at `{model_path}`.")
        return

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=False,
    )

    if uploaded is None:
        st.info("Upload an image to run EfficientNet-Lite2 inference.")
        return

    image_bytes = uploaded.getvalue()
    image_key = f"{model_choice}:{uploaded.name}:{hashlib.sha256(image_bytes).hexdigest()}"

    try:
        image = Image.open(BytesIO(image_bytes)).copy()
    except Exception as exc:
        st.error(f"Could not read the uploaded image: {exc}")
        return

    image_col, result_col = st.columns([1, 1.25])
    with image_col:
        st.image(image, caption=f"{uploaded.name} - {image.width} x {image.height}", width="stretch")
        st.caption(f"Selected model: `{model_path.name}`")

    with result_col:
        if st.button("Run inference", type="primary"):
            with st.spinner("Running EfficientNet-Lite2 ONNX inference..."):
                try:
                    results, elapsed_ms, input_shape, _ = run_image_inference(image, model_path)
                except Exception as exc:
                    st.error(f"Inference failed: {exc}")
                    return
            st.session_state.vision_inference = {
                "key": image_key,
                "results": results,
                "elapsed_ms": elapsed_ms,
                "input_shape": input_shape,
            }

        prediction = st.session_state.get("vision_inference")
        if prediction is None or prediction.get("key") != image_key:
            st.caption("Inference results will appear here after the model runs.")
            return

        results = prediction["results"].head(top_k)
        top_prediction = results.iloc[0]

        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Top Prediction", top_prediction["Class"])
        metric_col2.metric("Inference Time", f"{prediction['elapsed_ms']:.1f} ms")
        st.caption(f"Input tensor shape: `{tuple(prediction['input_shape'])}`")

        chart = px.bar(
            results.sort_values("Confidence"),
            x="Confidence",
            y="Class",
            orientation="h",
            text="Confidence",
            color="Confidence",
            color_continuous_scale=["#536878", "#b4533c"],
            title="Top ImageNet Predictions",
        )
        chart.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        chart.update_layout(showlegend=False, coloraxis_showscale=False, margin=dict(l=10, r=25, t=55, b=10))
        chart.update_xaxes(range=[0, max(100, results["Confidence"].max() * 1.15)], ticksuffix="%")
        st.plotly_chart(chart, width="stretch")

        st.dataframe(
            results,
            width="stretch",
            hide_index=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    format="%.2f%%",
                    min_value=0,
                    max_value=100,
                )
            },
        )


def render_metric_strip(host_results, board_metrics):
    fp32 = host_results.iloc[0]
    int8 = host_results.iloc[1]
    size_reduction = (1 - int8["ONNX Size (B)"] / fp32["ONNX Size (B)"]) * 100
    throughput_speedup = int8["Throughput (img/s)"] / fp32["Throughput (img/s)"]
    latency = board_metrics.get("latency_ms", 49.08)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Quantized Top-1", f"{int8['Top-1 Accuracy (%)']:.2f}%", "-0.08 pp")
    col2.metric("Quantized Top-5", f"{int8['Top-5 Accuracy (%)']:.2f}%", "+0.03 pp")
    col3.metric("ONNX Size", f"{int8['ONNX Size (MB)']:.2f} MB", f"-{size_reduction:.2f}%", delta_color="inverse")
    col4.metric("Board Latency", f"{latency:.2f} ms", f"{throughput_speedup:.2f}x host speedup")


def render_summary_tab(host_results, board_metrics):
    render_metric_strip(host_results, board_metrics)

    st.divider()

    left, right = st.columns([1.35, 1])

    with left:
        accuracy = host_results.melt(
            id_vars="Model",
            value_vars=["Top-1 Accuracy (%)", "Top-5 Accuracy (%)"],
            var_name="Metric",
            value_name="Accuracy (%)",
        )
        fig = px.bar(
            accuracy,
            x="Metric",
            y="Accuracy (%)",
            color="Model",
            barmode="group",
            text_auto=".2f",
            color_discrete_map=MODEL_COLORS,
            title="Host Accuracy Before and After Quantization",
        )
        fig.update_yaxes(range=[70, 100])
        fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, width="stretch")

    with right:
        size_fig = px.bar(
            host_results,
            x="Model",
            y="ONNX Size (MB)",
            color="Model",
            text_auto=".2f",
            color_discrete_map=MODEL_COLORS,
            title="Model File Size",
        )
        size_fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=55, b=10))
        size_fig.update_xaxes(title=None)
        st.plotly_chart(size_fig, width="stretch")

    st.caption(
        "The quantized EfficientNet-Lite2 model preserved ImageNet accuracy while reducing ONNX size enough "
        "to support STM32N6570-DK deployment."
    )


def render_host_tab(host_results):
    display = host_results.copy()
    display["ONNX Size"] = display["ONNX Size (B)"].apply(format_bytes)
    display = display[
        [
            "Model",
            "Format",
            "Top-1 Accuracy (%)",
            "Top-5 Accuracy (%)",
            "Throughput (img/s)",
            "ONNX Size",
            "STM32N6570-DK Deployable",
        ]
    ]

    st.dataframe(display, width="stretch", hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        throughput_fig = px.bar(
            host_results,
            x="Model",
            y="Throughput (img/s)",
            color="Model",
            text_auto=".2f",
            color_discrete_map=MODEL_COLORS,
            title="Host Throughput",
        )
        throughput_fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=55, b=10))
        throughput_fig.update_xaxes(title=None)
        st.plotly_chart(throughput_fig, width="stretch")

    with col2:
        tradeoff = host_results.assign(
            Accuracy_Label=host_results["Top-1 Accuracy (%)"].map(lambda value: f"{value:.2f}%")
        )
        scatter_fig = px.scatter(
            tradeoff,
            x="ONNX Size (MB)",
            y="Throughput (img/s)",
            color="Model",
            size="Top-1 Accuracy (%)",
            text="Accuracy_Label",
            color_discrete_map=MODEL_COLORS,
            title="Size, Speed, and Accuracy Tradeoff",
        )
        scatter_fig.update_traces(textposition="top center")
        scatter_fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(scatter_fig, width="stretch")


def render_edge_tab(board_metrics):
    latency = board_metrics.get("latency_ms", 49.08)
    throughput = 1000 / latency if latency else 20.37
    runtime_memory = board_metrics.get("runtime_memory_bytes")

    col1, col2, col3 = st.columns(3)
    col1.metric("Board", board_metrics.get("board", "STM32N6570-DK"))
    col2.metric("Latency / Inference", f"{latency:.2f} ms")
    col3.metric("Throughput", f"{throughput:.2f} inf/s")

    st.subheader("Runtime Footprint")
    memory = pd.DataFrame(
        [
            {"Component": "Weights", "Memory (MB)": 6.892},
            {"Component": "Activations", "Memory (MB)": 2.305},
            {"Component": "Runtime Total", "Memory (MB)": bytes_to_mb(runtime_memory) if runtime_memory else 9.197},
        ]
    )
    memory_fig = px.bar(
        memory,
        x="Component",
        y="Memory (MB)",
        color="Component",
        text_auto=".2f",
        color_discrete_sequence=["#b4533c", "#2f855a", "#536878"],
        title="STM32N6570-DK Memory Use",
    )
    memory_fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(memory_fig, width="stretch")

    details = pd.DataFrame(
        [
            {"Metric": "Deployable on STM32N6570-DK", "Value": "Yes"},
            {"Metric": "Board benchmark state", "Value": board_metrics.get("state", "done")},
            {"Metric": "Board benchmark created", "Value": board_metrics.get("created_at", "Apr 15 2026")},
            {"Metric": "Cycles", "Value": f"{board_metrics['cycles']:,}" if board_metrics.get("cycles") else "39263962"},
            {"Metric": "Quantized weight blob", "Value": "7,226,801 B"},
            {"Metric": "Runtime memory from benchmark CSV", "Value": format_bytes(runtime_memory)},
        ]
    )
    st.dataframe(details, width="stretch", hide_index=True)

    st.caption("The FP32 ONNX baseline was not board-benchmarked because it was not deployable within the target memory constraints.")


def render_workflow_tab():
    summary, sample_rows = load_manifest_summary()

    stages = pd.DataFrame(
        [
            {
                "Stage": "Baseline",
                "Artifact": "efficientnet_lite2_unquantized.onnx",
                "Purpose": "Evaluate the original FP32 ONNX model on ImageNet-1k validation data.",
            },
            {
                "Stage": "Calibration",
                "Artifact": "calibration_data.npz",
                "Purpose": "Capture representative activation ranges from ImageNet samples.",
            },
            {
                "Stage": "Static PTQ",
                "Artifact": "efficientnet_lite2_locally_quantized.onnx",
                "Purpose": "Apply ONNX Runtime QDQ quantization with QInt8 activations and weights.",
            },
            {
                "Stage": "Target Benchmark",
                "Artifact": "developer_cloud_benchmark_stm32n6570dk.csv",
                "Purpose": "Record STM32N6570-DK latency, cycles, and deployment status.",
            },
        ]
    )
    st.dataframe(stages, width="stretch", hide_index=True)

    if summary:
        col1, col2 = st.columns(2)
        col1.metric("Validation Images", f"{summary['images']:,}")
        col2.metric("ImageNet Synsets", f"{summary['synsets']:,}")

    if sample_rows is not None:
        with st.expander("Validation manifest sample"):
            st.dataframe(sample_rows, width="stretch", hide_index=True)

    with st.expander("Quantization settings"):
        st.markdown(
            """
            - Quantization format: `QuantFormat.QDQ`
            - Activation type: `QuantType.QInt8`
            - Weight type: `QuantType.QInt8`
            - Calibration method: `CalibrationMethod.MinMax`
            - Per-channel weight quantization: `True`
            - External interface: float input and float output
            """
        )


def render_reference_tab():
    reference = build_reference_results()
    st.dataframe(reference, width="stretch", hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        ref_accuracy = reference.melt(
            id_vars="Model",
            value_vars=["Top-1 Accuracy (%)", "Top-5 Accuracy (%)"],
            var_name="Metric",
            value_name="Accuracy (%)",
        )
        ref_accuracy_fig = px.bar(
            ref_accuracy,
            x="Metric",
            y="Accuracy (%)",
            color="Model",
            barmode="group",
            text_auto=".2f",
            color_discrete_map=MODEL_COLORS,
            title="Accuracy Reference",
        )
        ref_accuracy_fig.update_yaxes(range=[70, 100])
        ref_accuracy_fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(ref_accuracy_fig, width="stretch")

    with col2:
        footprint_fig = px.bar(
            reference,
            x="Model",
            y="Size / Footprint (MB)",
            color="Model",
            text_auto=".2f",
            color_discrete_map=MODEL_COLORS,
            title="Footprint Reference",
        )
        footprint_fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=55, b=10))
        footprint_fig.update_xaxes(title=None)
        st.plotly_chart(footprint_fig, width="stretch")

    st.caption("CSWin-Tiny is a host-side reference point, not a confirmed STM32N6570-DK deployment result.")


@st.fragment
def render():
    st.title("The Computer Vision Model")
    st.caption("EfficientNet-Lite2 ImageNet classifier optimized for STM32N6570-DK edge deployment.")

    host_results = build_host_results()
    board_metrics = load_board_benchmark()

    inference_tab, summary_tab, host_tab, edge_tab, workflow_tab, reference_tab = st.tabs(
        ["Inference", "Summary", "Host Validation", "Edge Deployment", "Workflow", "Reference"]
    )

    with inference_tab:
        render_inference_tab()

    with summary_tab:
        render_summary_tab(host_results, board_metrics)

    with host_tab:
        render_host_tab(host_results)

    with edge_tab:
        render_edge_tab(board_metrics)

    with workflow_tab:
        render_workflow_tab()

    with reference_tab:
        render_reference_tab()
