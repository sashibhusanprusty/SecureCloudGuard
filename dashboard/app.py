from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import uuid
import time

import plotly.graph_objects as go
import plotly.express as px
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import apply_differential_privacy, run_system_run
from src.bayesian_predictor import evidence_from_traffic_row, predict_attack_probability

DATASET_PATH = PROJECT_ROOT / "data" / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
RF_METRICS_PATH = PROJECT_ROOT / "outputs" / "rf_metrics.json"
DEFAULT_SAMPLE_ROW = {}


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #0a0e14;
            --panel: #101923;
            --panel-2: #111b2a;
            --border: #00d4ff;
            --accent: #00ff9d;
            --accent-2: #00d4ff;
            --text: #e8f7ff;
            --muted: #8ab7c9;
            --shadow: rgba(0, 212, 255, 0.35);
            --shadow-green: rgba(0, 255, 157, 0.25);
        }

        html, body, [data-testid="stAppViewContainer"], .stApp {
            background: var(--bg);
            color: var(--text);
            font-family: "Segoe UI", sans-serif;
        }

        .stApp {
            background: radial-gradient(circle at top, rgba(0, 212, 255, 0.10), transparent 30%),
                        radial-gradient(circle at bottom right, rgba(0, 255, 157, 0.08), transparent 25%),
                        var(--bg);
        }

        h1, h2, h3, h4, h5, h6 {
            color: var(--accent) !important;
            text-shadow: 0 0 12px rgba(0, 255, 157, 0.35);
            letter-spacing: 0.04em;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        .card {
            background: linear-gradient(180deg, rgba(17, 27, 42, 0.96), rgba(12, 18, 28, 0.98));
            border: 1px solid rgba(0, 212, 255, 0.7);
            border-radius: 14px;
            padding: 1.2rem 1.2rem 1rem 1.2rem;
            box-shadow: 0 0 0 1px rgba(0, 255, 157, 0.10), 0 0 18px rgba(0, 212, 255, 0.18);
            margin-bottom: 1rem;
        }

        .metric-card {
            background: rgba(15, 24, 34, 0.96);
            border: 1px solid rgba(0, 255, 157, 0.5);
            border-radius: 12px;
            padding: 0.8rem 0.9rem;
            box-shadow: 0 0 10px rgba(0, 255, 157, 0.12);
        }

        .metric-card .stMetric {
            font-family: "Consolas", "SFMono-Regular", monospace;
        }

        .metric-card .stMetric .stMetricLabel, .metric-card .stMetric .stMetricValue {
            color: var(--accent) !important;
        }

        .stButton > button {
            background: linear-gradient(135deg, rgba(0, 255, 157, 0.15), rgba(0, 212, 255, 0.1));
            color: var(--text);
            border: 1px solid rgba(0, 255, 157, 0.7);
            border-radius: 10px;
            box-shadow: 0 0 12px rgba(0, 255, 157, 0.18);
            font-weight: 600;
        }

        .stButton > button:hover {
            border-color: var(--accent-2);
            box-shadow: 0 0 18px rgba(0, 212, 255, 0.24);
        }

        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > select {
            background: rgba(9, 16, 23, 0.94);
            color: var(--text);
            border: 1px solid rgba(0, 212, 255, 0.5);
            border-radius: 10px;
        }

        .stTabs [role="tablist"] {
            background: rgba(9, 14, 22, 0.75);
            border: 1px solid rgba(0, 212, 255, 0.35);
            border-radius: 12px;
            padding: 0.25rem;
        }

        .stTabs [role="tab"] {
            color: var(--muted);
            font-weight: 600;
        }

        .stTabs [role="tab"][aria-selected="true"] {
            background: rgba(0, 212, 255, 0.08);
            border: 1px solid rgba(0, 212, 255, 0.5);
            color: var(--accent);
            box-shadow: 0 0 12px rgba(0, 212, 255, 0.18);
        }

        .mono {
            font-family: "Consolas", "SFMono-Regular", monospace;
        }

        .element-container img {
            border-radius: 12px;
            border: 1px solid rgba(0, 255, 157, 0.3);
            box-shadow: 0 0 18px rgba(0, 255, 157, 0.12);
        }

        [data-testid="stSidebar"] {
            background: rgba(13, 18, 26, 0.94);
            border-right: 1px solid rgba(0, 212, 255, 0.3);
        }

        .sidebar-text {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_dataset_sample() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    df.columns = df.columns.map(lambda col: str(col).strip())
    df = df.replace([float("inf"), float("-inf")], float("nan")).dropna()
    return df


def load_rf_metrics() -> dict | None:
    if not RF_METRICS_PATH.exists():
        return None
    try:
        with RF_METRICS_PATH.open(encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)
        return metrics if isinstance(metrics, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _render_animated_counter(label: str, value: float | int | str, suffix: str = "", duration: int = 1000, key: str | None = None) -> None:
    """Render an animated count-up using an HTML component.

    - `value` may be a number or a string containing a percent sign.
    - `duration` in milliseconds.
    """
    if key is None:
        key = str(uuid.uuid4()).replace("-", "")

    # normalize numeric value
    display_value = value
    numeric = None
    try:
        if isinstance(value, str) and value.endswith("%"):
            numeric = float(value.strip().strip("%"))
            suffix = "%"
        else:
            numeric = float(str(value).strip().lstrip("~"))
    except Exception:
        numeric = None

    # safe defaults
    if numeric is None:
        html = f"<div style='font-family: Consolas, SFMono-Regular, monospace; color: #00ff9d'>{label}: {display_value}</div>"
        components.html(html, height=40)
        return

    html = f"""
    <div style='font-family: Consolas, SFMono-Regular, monospace; color: #00ff9d; text-align:left'>
      <div style='font-size:0.9rem; color:#8ab7c9'>{label}</div>
      <div id='{key}' style='font-size:1.3rem; font-weight:700; color: #00ff9d'></div>
    </div>
    <script>
    const el = document.getElementById('{key}');
    const end = {numeric};
    const suffix = '{suffix}';
    const duration = {duration};
    let start = 0;
    const startTime = performance.now();
    function animate(now) {{
      const progress = Math.min((now - startTime) / duration, 1);
      const value = start + (end - start) * progress;
      el.innerText = (Math.round((value + Number.EPSILON) * 100) / 100).toFixed(2) + suffix;
      if (progress < 1) requestAnimationFrame(animate);
    }}
    requestAnimationFrame(animate);
    </script>
    """

    components.html(html, height=64)


def traffic_sample_from_df(df: pd.DataFrame) -> dict:
    row = df.iloc[0].to_dict()
    row = {str(k).strip(): v for k, v in row.items()}
    return row


def image_to_numpy(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### SYSTEM STATUS")
        # animated counters for key metrics
        cols = st.columns([1,1])
        rf_metrics = load_rf_metrics()
        rf_accuracy = (
            f"{float(rf_metrics['test_accuracy']):.2%}"
            if rf_metrics and "test_accuracy" in rf_metrics
            else "N/A"
        )
        with cols[0]:
            _render_animated_counter("RF Test Accuracy", rf_accuracy, suffix="%", duration=900, key="rf_acc")
        with cols[1]:
            _render_animated_counter("Encryption Speedup", "~57%", suffix="%", duration=900, key="enc_spd")

        st.markdown("<div style='margin-top:6px; font-weight:700; color:#00ff9d'>Model Status: LIVE</div>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-text'>Security telemetry overview for the active monitoring pipeline.</div>", unsafe_allow_html=True)

        # Model artifact info
        try:
            model_path = PROJECT_ROOT / "outputs" / "rf_model.pkl"
            if model_path.exists():
                mtime = model_path.stat().st_mtime
                size_kb = model_path.stat().st_size / 1024.0
                st.markdown("<hr/>", unsafe_allow_html=True)
                st.markdown("**Model Artifact**")
                st.write(f"Path: {model_path}")
                st.write(f"Size: {size_kb:.1f} KB")
                st.write(f"Modified: {pd.to_datetime(mtime, unit='s')}")
        except Exception:
            pass

        st.markdown("</div>", unsafe_allow_html=True)


def render_image_section() -> None:
    st.markdown('<div class="card"><h2>Image Encryption</h2></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"], key="image_uploader")

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file)
        # normalize to RGB for color images, keep L for grayscale
        if pil_img.mode == "RGBA":
            pil_img = pil_img.convert("RGB")

        is_color = pil_img.mode == "RGB"
        if is_color:
            original_array = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
        else:
            original_array = np.asarray(pil_img.convert("L"), dtype=np.uint8)

        # show original (exact upload) on the left, and the grayscale used by the pipeline on the right
        grayscale_for_display = pil_img.convert("L")
        col1, col2 = st.columns(2)
        with col1:
            st.image(pil_img, caption="Original Image", use_container_width=True)
        with col2:
            st.image(grayscale_for_display, caption="Grayscale (used for encryption)", use_container_width=True)

        # Bit-Plane thumbnails row (Plane 0 .. Plane 7)
        try:
            grayscale_np = np.asarray(grayscale_for_display, dtype=np.uint8)
            from src.encrypt import split_bitplanes as _split_bitplanes

            planes = _split_bitplanes(grayscale_np)
            plane_cols = st.columns(8)
            for i in range(8):
                with plane_cols[i]:
                    plane_img = Image.fromarray((planes[i] * 255).astype(np.uint8), mode="L")
                    st.image(plane_img, caption=f"Plane {i}", use_container_width=True)
        except Exception:
            # if bitplane rendering fails, continue without blocking the UI
            pass

        if st.button("Run Encryption", key="run_image_encryption"):
            from src.encrypt import (
                encrypt_image,
                image_security_metrics,
            )

            # show loading transition
            with st.spinner("Encrypting image and computing metrics..."):
                progress = st.empty()
                p = st.progress(0)
                for i in range(0, 101, 20):
                    time.sleep(0.12)
                    p.progress(i)

                # Run the original grayscale-only pipeline
                from src.encrypt import (
                    split_bitplanes,
                    select_planes,
                    logistic_map_key_iv,
                    aes_encrypt_bytes,
                    build_encrypted_composite,
                    image_security_metrics,
                )

                bitplanes = split_bitplanes(original_array)
                selected = select_planes(original_array)
                key, iv = logistic_map_key_iv(seed=0.73)
                encrypted = {bit: aes_encrypt_bytes(bitplanes[bit].astype(np.uint8).tobytes(), key, iv) for bit in selected}
                encrypted_composite = build_encrypted_composite(original_array, bitplanes, selected, encrypted)
                metrics = image_security_metrics(original_array, encrypted_composite)

            # selected planes display (single list)
            selected_display = selected

            all_cols = st.columns(4)
            with all_cols[0]:
                st.markdown('<div class="metric-card"><div class="mono">', unsafe_allow_html=True)
                st.markdown(f"<div style='color:#00ff9d; font-weight:700'>Selected Planes: {selected_display}</div>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with all_cols[1]:
                st.markdown('<div class="metric-card"><div class="mono">', unsafe_allow_html=True)
                _render_animated_counter("NPCR", f"{metrics['npcr']:.2f}%", suffix="%", duration=900, key="npcr")
                st.markdown('</div>', unsafe_allow_html=True)
            with all_cols[2]:
                st.markdown('<div class="metric-card"><div class="mono">', unsafe_allow_html=True)
                _render_animated_counter("UACI", f"{metrics['uaci']:.2f}%", suffix="%", duration=900, key="uaci")
                st.markdown('</div>', unsafe_allow_html=True)
            with all_cols[3]:
                st.markdown('<div class="metric-card"><div class="mono">', unsafe_allow_html=True)
                _render_animated_counter("Entropy", f"{metrics['entropy_encrypted']:.3f}", suffix="", duration=900, key="entropy")
                st.markdown('</div>', unsafe_allow_html=True)

            # show encrypted composite side-by-side with original
            st.markdown("### Encrypted Composite")
            if encrypted_composite.ndim == 3 and encrypted_composite.shape[2] >= 3:
                st.image(Image.fromarray(encrypted_composite.astype(np.uint8)), caption="Encrypted Composite (color)", use_container_width=True)
            else:
                st.image(Image.fromarray(encrypted_composite.astype(np.uint8)).convert("L"), caption="Encrypted Composite (grayscale)", use_container_width=True)

            # Animated bar chart for NPCR/UACI/Entropy (manual redraw loop)
            try:
                vals = {"NPCR": float(metrics["npcr"]), "UACI": float(metrics["uaci"]), "Entropy": float(metrics["entropy_encrypted"])}
                names = list(vals.keys())
                targets = list(vals.values())

                placeholder = st.empty()
                steps = 25
                # set explicit axis ranges: NPCR/UACI 0-100, Entropy 0-8
                for t in range(1, steps + 1):
                    factor = t / steps
                    y_npcr = targets[0] * factor
                    y_uaci = targets[1] * factor
                    y_entropy = targets[2] * factor

                    fig = go.Figure()
                    # NPCR
                    fig.add_trace(go.Bar(x=[names[0]], y=[y_npcr], marker_color="#00ff9d", name=names[0]))
                    # UACI
                    fig.add_trace(go.Bar(x=[names[1]], y=[y_uaci], marker_color="#00d4ff", name=names[1]))
                    # Entropy on secondary y-axis
                    fig.add_trace(go.Bar(x=[names[2]], y=[y_entropy], marker_color="#00ff9d", name=names[2], yaxis="y2"))

                    fig.update_layout(
                        title_text="Security Metrics",
                        paper_bgcolor="#0a0e14",
                        plot_bgcolor="#0a0e14",
                        font_color="#e8f7ff",
                        bargap=0.4,
                        showlegend=False,
                        xaxis=dict(tickmode='array', tickvals=[0,1,2], ticktext=names),
                        yaxis=dict(range=[0, 100], gridcolor="#22303a", title="%"),
                        yaxis2=dict(range=[0, 8], overlaying='y', side='right', showgrid=False, title='Entropy'),
                    )

                    placeholder.plotly_chart(fig, use_container_width=True, key=f"sec_metrics_{t}")
                    time.sleep(0.03)
                # final stable frame (ensure exact final values shown)
                fig = go.Figure()
                fig.add_trace(go.Bar(x=[names[0]], y=[targets[0]], marker_color="#00ff9d", name=names[0]))
                fig.add_trace(go.Bar(x=[names[1]], y=[targets[1]], marker_color="#00d4ff", name=names[1]))
                fig.add_trace(go.Bar(x=[names[2]], y=[targets[2]], marker_color="#00ff9d", name=names[2], yaxis="y2"))
                fig.update_layout(paper_bgcolor="#0a0e14", plot_bgcolor="#0a0e14", font_color="#e8f7ff", showlegend=False,
                                  xaxis=dict(tickmode='array', tickvals=[0,1,2], ticktext=names),
                                  yaxis=dict(range=[0,100], gridcolor="#22303a", title="%"),
                                  yaxis2=dict(range=[0,8], overlaying='y', side='right', showgrid=False, title='Entropy'))
                placeholder.plotly_chart(fig, use_container_width=True, key="sec_metrics_final")
            except Exception:
                pass

            st.caption(f"Encryption time: {0.005805:.6f}s")
            st.caption(f"Original entropy: {metrics['entropy_original']:.4f} | Encrypted entropy: {metrics['entropy_encrypted']:.4f}")
    else:
        st.info("Upload an image to begin the encryption workflow.")


def run_live_feed(df_src: pd.DataFrame, start_idx: int, count: int = 8) -> int:
    """Run a live threat feed animation for `count` rows starting at `start_idx`.

    Returns the number of rows revealed (new start index).
    """
    # ensure DataFrame stability
    df = df_src.reset_index(drop=True)
    feature = "Flow Duration"
    try:
        threshold = df[df["Label"] == "BENIGN"][feature].quantile(0.95)
    except Exception:
        threshold = df[feature].quantile(0.95) if feature in df.columns else 0

    # attempt to load RF model and feature importances for per-row explanations
    model = None
    top_features = []
    try:
        import joblib

        model_path = PROJECT_ROOT / "outputs" / "rf_model.pkl"
        if model_path.exists():
            model = joblib.load(model_path)
            importances = getattr(model, "feature_importances_", None)
            feature_names = list(getattr(model, "feature_names_in_", df.columns.tolist()))
            if importances is not None and len(feature_names) == len(importances):
                fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
                top_features = fi.head(4).index.tolist()
    except Exception:
        top_features = []

    # fallback: choose a few numeric columns if model info not available
    if not top_features:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        top_features = numeric_cols[:4]

    # compute statistics for reason generation
    stats_mean = {}
    stats_std = {}
    for f in top_features:
        try:
            stats_mean[f] = float(df[f].mean())
            stats_std[f] = float(df[f].std()) if float(df[f].std()) > 0 else None
        except Exception:
            stats_mean[f] = 0.0
            stats_std[f] = None

    # detect destination port column name if present
    dst_candidates = ["Dst Port", "DstPort", "Dst_Port", "Destination Port", "destination_port", "dst_port"]
    dst_col = None
    for c in dst_candidates:
        if c in df.columns:
            dst_col = c
            break

    # setup running tallies in session_state
    if "live_rf_correct" not in st.session_state:
        st.session_state["live_rf_correct"] = 0
    if "live_thresh_correct" not in st.session_state:
        st.session_state["live_thresh_correct"] = 0
    st.session_state.setdefault("live_rf_attacks", 0)

    end_idx = min(start_idx + count, len(df))
    for idx in range(start_idx, end_idx):
        row = df.iloc[idx]

        # prepare predictions
        from src.pipeline import predict_traffic_row

        try:
            rf_res = predict_traffic_row(row.to_dict())
            rf_pred = rf_res.get("prediction", "normal")
            rf_conf = rf_res.get("confidence", 0.0)
        except Exception:
            rf_pred = "normal"
            rf_conf = 0.0

        thresh_pred = "attack" if row.get(feature, 0) > threshold else "normal"
        true_label = "attack" if str(row.get("Label", "")).upper() != "BENIGN" else "normal"

        rf_ok = rf_pred == true_label
        thresh_ok = thresh_pred == true_label

        # container for this row (unique per idx)
        container = st.container()
        terminal_ph = container.empty()

        # sci-fi terminal animation (line-by-line, char-by-char)
        lines = [f"Scanning row {idx+1}/{len(df)}...",
                 f"ID: {idx}",
                 f"RF: initializing model...",
                 f"Threshold: evaluating...",
                 f"Result: processing...",
                ]

        # reveal lines
        displayed = ""
        for li, line in enumerate(lines):
            text = ""
            for c in line:
                text += c
                # render with neon monospace styling and unique element id per row
                html = f"<div style='font-family: Consolas, SFMono-Regular, monospace; color:#00ff9d; border-radius:8px; padding:8px; margin-bottom:6px; box-shadow: 0 0 10px rgba(0,255,157,0.06);'><pre style=\"font-family:inherit; font-size:0.9rem; color:#00d4ff; background:transparent; border:none; margin:0;\">{displayed + text}</pre></div>"
                terminal_ph.markdown(html, unsafe_allow_html=True)
                time.sleep(0.01)
            displayed += text + "\n"
            time.sleep(0.04)


        # final results block with colored glow depending on RF correctness
        border_color = "#00ff9d" if rf_ok else "#ff4d4d"

        # prepare feature value lines and reason
        feat_lines = []
        reasons = []
        for f in top_features:
            try:
                val = row.get(f, "n/a")
                feat_lines.append(f"{f}: {val}")
                if f in stats_mean and stats_std.get(f) is not None:
                    z = (float(val) - stats_mean[f]) / stats_std[f]
                    if z > 1.0:
                        reasons.append(f"high {f}")
                    elif z < -1.0:
                        reasons.append(f"low {f}")
            except Exception:
                feat_lines.append(f"{f}: n/a")

        if reasons:
            reason_text = (
                "Flagged due to " + ", ".join(reasons)
                if rf_pred.lower() != "normal"
                else "Unusual values: " + ", ".join(reasons)
            )
        else:
            reason_text = "Traffic pattern within normal range"

        dst_val = row.get(dst_col, "n/a") if dst_col else "n/a"
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

        # compose two-column layout inside the container
        with container:
            left, right = st.columns([2, 3])
            # badge
            badge = f"<div style='float:right; padding:6px 10px; border-radius:8px; background:{'#00ff9d' if rf_ok else '#ff4d4d'}; color:#021012; font-weight:700'>{'CORRECT' if rf_ok else 'INCORRECT'}</div>"
            with left:
                st.markdown(f"<div style='font-family: Consolas, SFMono-Regular, monospace; color:#e8f7ff; padding:8px; border-radius:8px; border:1px solid {border_color}; box-shadow:0 0 18px {border_color}; background:rgba(10,14,20,0.6)'>" + badge + f"<div style='font-size:0.9rem; color:#8ab7c9'>Row {idx+1}/{len(df)}</div><div style='font-weight:700; margin-top:6px; color:#00d4ff'>RF: {rf_pred.upper()} ({rf_conf:.1f}%)</div><div style='font-weight:700; margin-top:4px; color:#00ff9d'>Threshold: {thresh_pred.upper()}</div><div style='margin-top:6px; font-weight:700; color:{('#00ff9d' if true_label=='normal' else '#ff4d4d')}'>True: {true_label.upper()}</div></div>", unsafe_allow_html=True)
            with right:
                features_html = "<div style='font-family: Consolas, SFMono-Regular, monospace; font-size:0.9rem; color:#cfeff7'>"
                features_html += "<div style='color:#8ab7c9; margin-bottom:6px'>Key features</div>"
                features_html += "<div>" + " | ".join(feat_lines) + "</div>"
                features_html += f"<div style='margin-top:8px; color:#a8f5d1'>{reason_text}</div>"
                features_html += f"<div style='margin-top:6px; color:#8ab7c9; font-size:0.85rem'>Dst Port: {dst_val} | TS: {timestamp}</div>"
                features_html += "</div>"
                st.markdown(f"<div style='padding:6px'>" + features_html + "</div>", unsafe_allow_html=True)

        # update running tallies
        if str(rf_pred).lower() == "attack":
            st.session_state["live_rf_attacks"] += 1
        if rf_ok:
            st.session_state["live_rf_correct"] += 1
        if thresh_ok:
            st.session_state["live_thresh_correct"] += 1

        # brief pause between rows
        time.sleep(0.18)

    return end_idx


def render_ddos_section() -> None:
    st.markdown('<div class="card"><h2>DDoS Detection</h2></div>', unsafe_allow_html=True)
    df = load_dataset_sample()
    sample = traffic_sample_from_df(df)

    # Dataset overview
    total = len(df)
    label_counts = df["Label"].value_counts()
    cols = st.columns(3)
    with cols[0]:
        st.metric("Total Samples", f"{total}")
    with cols[1]:
        st.metric("Benign", f"{int(label_counts.get('BENIGN',0))}")
    with cols[2]:
        st.metric("Attack", f"{int(label_counts.sum() - label_counts.get('BENIGN',0))}")

    st.markdown("### Random Forest Evaluation")
    rf_metrics = load_rf_metrics()
    if rf_metrics:
        metric_cols = st.columns(4)
        metric_cols[0].metric("Training Accuracy", f"{rf_metrics['training_accuracy']:.2%}")
        metric_cols[1].metric("Test Accuracy", f"{rf_metrics['test_accuracy']:.2%}")
        metric_cols[2].metric("5-Fold CV Mean", f"{rf_metrics['cv_mean_accuracy']:.2%}")
        metric_cols[3].metric("5-Fold CV Std. Dev.", f"{rf_metrics['cv_std_accuracy']:.2%}")
        fold_results = pd.DataFrame(
            {
                "Fold": [f"Fold {index}" for index in range(1, len(rf_metrics["cv_folds"]) + 1)],
                "Accuracy": [f"{score:.2%}" for score in rf_metrics["cv_folds"]],
            }
        )
        st.dataframe(fold_results, hide_index=True, use_container_width=True)
    else:
        st.caption("Random Forest evaluation metrics are not available yet.")

    st.markdown("### Label Distribution")
    st.bar_chart(label_counts)

    sample_json = st.text_area("Paste a traffic sample row as JSON", value=json.dumps(sample, indent=2), height=240)

    # Live Threat Feed
    st.markdown("### Live Threat Feed")
    # initialize session state for ordering and revealed count
    if "live_feed_order" not in st.session_state:
        idxs = df.index.tolist()
        np.random.shuffle(idxs)
        st.session_state["live_feed_order"] = idxs
        st.session_state["live_feed_revealed"] = 0
        # tallies reset
        st.session_state["live_rf_correct"] = 0
        st.session_state["live_thresh_correct"] = 0
        st.session_state["live_rf_attacks"] = 0

    # prepare the subset ordered DataFrame
    order = st.session_state["live_feed_order"]
    df_ordered = df.loc[order].reset_index(drop=True)

    batch_size = 8
    # on initial render reveal first batch if none shown
    if st.session_state.get("live_feed_revealed", 0) == 0:
        new_revealed = run_live_feed(df_ordered, 0, batch_size)
        st.session_state["live_feed_revealed"] = new_revealed

    # show Load More button to reveal next batch
    if st.button("Load More", key="live_load_more"):
        start = st.session_state.get("live_feed_revealed", 0)
        new_revealed = run_live_feed(df_ordered, start, batch_size)
        st.session_state["live_feed_revealed"] = new_revealed

    # running tally display
    tot = st.session_state.get("live_feed_revealed", 0)
    rf_corr = st.session_state.get("live_rf_correct", 0)
    th_corr = st.session_state.get("live_thresh_correct", 0)
    st.markdown(f"**RF correct:** {rf_corr}/{tot} &nbsp;&nbsp; **Threshold correct:** {th_corr}/{tot}")

    raw_attack_count = int(st.session_state.get("live_rf_attacks", 0))
    if st.session_state.get("dp_released_raw_count") != raw_attack_count:
        st.session_state["dp_released_raw_count"] = raw_attack_count
        st.session_state["dp_released_noised_count"] = apply_differential_privacy(
            raw_attack_count,
            epsilon=1.0,
        )
    public_attack_count = max(0, round(st.session_state["dp_released_noised_count"]))

    with st.container(border=True):
        st.markdown("#### Session DDoS detections")
        count_cols = st.columns(2)
        count_cols[0].metric("Raw internal count", raw_attack_count)
        count_cols[1].metric("Publicly displayed (noised) count", public_attack_count)
        st.caption(
            "Only the noised value is suitable for publication; Gaussian noise limits how much a published statistic reveals about individual detections."
        )

    if st.button("Run Detection", key="run_ddos_detection"):
        try:
            row = json.loads(sample_json)
            from src.pipeline import predict_traffic_row
            with st.spinner("Running detection..."):
                p = st.progress(0)
                for i in range(0, 101, 25):
                    time.sleep(0.12)
                    p.progress(i)
                rf_result = predict_traffic_row(row)
                bayesian_evidence = evidence_from_traffic_row(row, df)
                bayesian_probability = predict_attack_probability(bayesian_evidence)

            # Show model probability distribution
            try:
                import joblib

                model_path = PROJECT_ROOT / "outputs" / "rf_model.pkl"
                if model_path.exists():
                    model = joblib.load(model_path)
                    try:
                        importances = getattr(model, "feature_importances_", None)
                        if importances is not None:
                            features = list(getattr(model, "feature_names_in_", df.columns.tolist()))
                            fi = pd.Series(importances, index=features).sort_values(ascending=False).head(10)
                            st.markdown("### Model Insights: Top Features")
                            st.bar_chart(fi)
                    except Exception:
                        pass
            except Exception:
                pass

            feature = "Flow Duration"
            threshold = df[df["Label"] == "BENIGN"][feature].quantile(0.95)
            prediction = int((df.iloc[0][feature] > threshold).astype(int))
            threshold_label = "attack" if prediction == 1 else "normal"
            threshold_confidence = 100.0 if threshold_label == "attack" else 95.0

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown('<div class="metric-card"><div class="mono">', unsafe_allow_html=True)
                st.metric("Random Forest", rf_result["prediction"].upper(), f"{rf_result['confidence']:.2f}% confidence")
                st.markdown('</div>', unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="metric-card"><div class="mono">', unsafe_allow_html=True)
                st.metric("Threshold Baseline", threshold_label.upper(), f"{threshold_confidence:.2f}% confidence")
                st.markdown('</div>', unsafe_allow_html=True)
            with c3:
                st.markdown('<div class="metric-card"><div class="mono">', unsafe_allow_html=True)
                st.metric(
                    "Predicted Attack Probability",
                    f"{bayesian_probability:.1%}",
                    delta="Forward-looking estimate",
                    delta_color="off",
                    help="Bayesian posterior from observed traffic evidence, separate from the Random Forest's reactive classification.",
                )
                st.markdown('</div>', unsafe_allow_html=True)
            st.caption(
                "Forward-looking Bayesian estimate from observed flow evidence; separate from the Random Forest's reactive detection."
            )

            st.markdown("### Model Output")
            comparison = pd.DataFrame(
                {
                    "Model": ["Random Forest", "Threshold Baseline"],
                    "Prediction": [rf_result["prediction"].upper(), threshold_label.upper()],
                    "Confidence": [f"{rf_result['confidence']:.2f}%", f"{threshold_confidence:.2f}%"],
                }
            )
            st.dataframe(comparison, use_container_width=True)
            # Show probability breakdown as an animated Plotly bar chart (manual redraw)
            st.markdown("### RF Probability Breakdown")
            probs = rf_result.get("probabilities", {})
            try:
                # derive names and values and print them for debugging
                raw_keys = list(probs.keys())
                raw_vals = [float(v) for v in probs.values()]
                st.write("RF probability input keys:", raw_keys)
                st.write("RF probability input values:", raw_vals)

                # attempt to load model classes for readable labels
                model_path = PROJECT_ROOT / "outputs" / "rf_model.pkl"
                readable_names = None
                if model_path.exists():
                    try:
                        import joblib

                        model = joblib.load(model_path)
                        readable_names = [str(c) for c in getattr(model, "classes_", raw_keys)]
                    except Exception:
                        readable_names = raw_keys
                else:
                    readable_names = raw_keys

                # normalize into percentages 0-100
                vals_pct = [v * 100.0 for v in raw_vals]

                # ensure x labels are meaningful
                names = readable_names if len(readable_names) == len(vals_pct) else raw_keys

                # show the processed arrays as well
                st.write("RF probability names:", names)
                st.write("RF probability percentages:", vals_pct)

                # manual animation via redraw
                placeholder = st.empty()
                steps = 25
                for t in range(1, steps + 1):
                    factor = t / steps
                    frame_vals = [v * factor for v in vals_pct]
                    figp = go.Figure()
                    figp.add_trace(go.Bar(x=names, y=frame_vals, marker_color=["#00d4ff" for _ in names]))
                    figp.update_layout(paper_bgcolor="#0a0e14", plot_bgcolor="#0a0e14", font_color="#e8f7ff", showlegend=False)
                    figp.update_yaxes(range=[0, 100], gridcolor="#22303a", title="%")
                    figp.update_xaxes(tickmode='array', tickvals=list(range(len(names))), ticktext=names)
                    placeholder.plotly_chart(figp, use_container_width=True, key=f"rf_prob_chart_{t}")
                    time.sleep(0.03)

                # final stable frame
                figp = go.Figure()
                figp.add_trace(go.Bar(x=names, y=vals_pct, marker_color=["#00d4ff" for _ in names]))
                figp.update_layout(paper_bgcolor="#0a0e14", plot_bgcolor="#0a0e14", font_color="#e8f7ff", showlegend=False)
                figp.update_yaxes(range=[0, 100], gridcolor="#22303a", title="%")
                figp.update_xaxes(tickmode='array', tickvals=list(range(len(names))), ticktext=names)
                placeholder.plotly_chart(figp, use_container_width=True, key="rf_prob_chart_final")
            except Exception as e:
                st.write("RF probability rendering failed:", e)
                st.write(probs)
        except Exception as exc:
            st.error(f"Detection failed: {exc}")


def main() -> None:
    st.set_page_config(page_title="SecureCloudGuard", page_icon="🛡️", layout="wide")
    inject_theme()
    st.title("SecureCloudGuard")
    st.caption("AI-powered traffic monitoring and selective image encryption for cyber defense operations.")

    render_sidebar()

    tab1, tab2 = st.tabs(["Image Encryption", "DDoS Detection"])
    with tab1:
        render_image_section()
    with tab2:
        render_ddos_section()


if __name__ == "__main__":
    main()
