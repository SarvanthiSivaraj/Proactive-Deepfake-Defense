from __future__ import annotations

import os
import sys
import tempfile
import json
from pathlib import Path

# Add project root directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from benchmark_suite import generate_dataset, run_benchmark
from src.preprocessing.loader import load_audio
from src.preprocessing.stft import compute_stft, split_mag_phase
from src.verification.service import DeepfakeDefenseService


st.set_page_config(page_title="Proactive Deepfake Defense", page_icon="🛡", layout="wide")
SERVICE = DeepfakeDefenseService()


def _persist_uploaded(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded_file.getbuffer())
        return handle.name


def _style_console():
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(135deg, #08111a 0%, #111827 45%, #0b1320 100%); color: #e5eef7; }
        .status-chip { display: inline-block; padding: 10px 14px; border-radius: 999px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
        .title-line { font-size: 2.4rem; font-weight: 800; letter-spacing: -0.04em; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _gauge_figure(ber: float):
    return go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(ber),
            number={"valueformat": ".3f"},
            title={"text": "BER"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "#f59e0b"},
                "steps": [
                    {"range": [0, 0.1], "color": "#064e3b"},
                    {"range": [0.1, 0.3], "color": "#854d0e"},
                    {"range": [0.3, 1.0], "color": "#7f1d1d"},
                ],
            },
        )
    )


def _spectral_metrics_figure(metrics: dict[str, float]):
    keys = ["centroid", "hf_ratio", "lf_ratio", "flatness", "edge_ratio", "column_shift_pressure"]
    return px.bar(
        x=keys,
        y=[metrics.get(key, 0.0) for key in keys],
        labels={"x": "Metric", "y": "Value"},
        title="Spectral Metrics",
        color_discrete_sequence=["#f59e0b"],
    )


def _feature_importance_figure(importance_rows: list[dict[str, float]]):
    if not importance_rows:
        return go.Figure()
    labels = [row["feature"] for row in importance_rows[:8]]
    values = [row["importance"] for row in importance_rows[:8]]
    return px.bar(
        x=values,
        y=labels,
        orientation="h",
        title="Top Influential Features",
        color=values,
        color_continuous_scale=["#082f49", "#0ea5e9", "#f59e0b"],
    )


def _shap_explanation_figure(shap_rows: list[dict[str, float]], predicted_label: str):
    if not shap_rows:
        return go.Figure()
    
    sorted_rows = sorted(shap_rows, key=lambda x: abs(x["shap_value"]), reverse=True)[:8]
    sorted_rows = sorted_rows[::-1]
    
    labels = [row["feature"] for row in sorted_rows]
    values = [row["shap_value"] for row in sorted_rows]
    
    colors = ["#ef4444" if val >= 0 else "#3b82f6" for val in values]
    
    min_val = min(values) if min(values) < 0 else 0
    max_val = max(values) if max(values) > 0 else 0
    padding = (max_val - min_val) * 0.18 or 0.05
    
    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{val:+.4f}" for val in values],
        textposition="outside"
    ))
    
    fig.update_layout(
        title=f"SHAP Local Explanations (Predicted: {predicted_label})",
        xaxis=dict(
            title="SHAP Value (Contribution)",
            zeroline=True,
            zerolinecolor="#94a3b8",
            zerolinewidth=1.5,
            range=[min_val - padding, max_val + padding]
        ),
        yaxis=dict(
            title="Feature",
            anchor="free",
            position=0.0,
            side="left"
        ),
        margin=dict(l=150, r=20, t=50, b=50),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def _rule_heatmap_figure(scores: dict[str, float]):
    labels = list(scores.keys())
    values = [scores[label] for label in labels]
    figure = go.Figure(data=go.Heatmap(z=[values], x=labels, y=["Rule Score"], colorscale=[[0, "#082f49"], [0.5, "#0ea5e9"], [1, "#f59e0b"]]))
    figure.update_layout(title="Rule Score Heatmap")
    return figure


def _audio_profile_figures(audio, sr):
    times = np.arange(len(audio)) / float(sr)
    waveform = go.Figure()
    waveform.add_trace(go.Scatter(x=times, y=audio, mode="lines", line=dict(color="#38bdf8", width=1.2)))
    waveform.update_layout(title="Waveform", xaxis_title="Seconds", yaxis_title="Amplitude")

    stft = compute_stft(audio)
    magnitude, _ = split_mag_phase(stft)
    spectrogram = go.Figure(data=go.Heatmap(z=20 * np.log10(magnitude + 1e-12), colorscale="Viridis"))
    spectrogram.update_layout(title="Spectrogram", xaxis_title="Frame", yaxis_title="Frequency Bin")

    freq_profile = magnitude.mean(axis=1)
    freq_axis = np.linspace(0.0, sr / 2.0, len(freq_profile))
    frequency = go.Figure()
    frequency.add_trace(go.Scatter(x=freq_axis, y=freq_profile, mode="lines", line=dict(color="#f59e0b", width=1.4)))
    frequency.update_layout(title="Frequency Profile", xaxis_title="Hz", yaxis_title="Mean Magnitude")
    return waveform, spectrogram, frequency


def _generate_pdf_report(result, uploaded_file, metrics, explainability) -> bytes:
    from fpdf import FPDF
    
    # Create PDF object
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Page 1: Verification & Metadata
    pdf.add_page()
    
    # Header Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Proactive Deepfake Defense", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 8, "Forensic Verification Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)
    
    # Horizontal line separator
    pdf.line(10, 33, 200, 33)
    pdf.ln(10)
    
    # Verification Summary
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Verification Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    
    # Status highlight
    status = result.auth_card_status
    if status == "AUTHENTIC":
        pdf.set_text_color(15, 118, 110) # Teal/Green
    else:
        pdf.set_text_color(153, 27, 27) # Red
        
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, f"STATUS: {status}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0) # Reset to black
    pdf.set_font("Helvetica", "I", 10)
    pdf.multi_cell(0, 6, f"Detail: {result.final_authentication}")
    pdf.ln(5)
    
    # Provenance Metadata
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Provenance Details", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    
    if result.signature_valid:
        for key in ["creator", "timestamp", "generator", "model_version", "source_hash", "verified_hash"]:
            val = result.provenance.get(key, "N/A")
            pdf.set_font("Helvetica", "B", 10)
            pdf.write(6, f"{key.replace('_', ' ').capitalize()}: ")
            pdf.set_font("Helvetica", "", 10)
            pdf.write(6, f"{val}\n")
    else:
        pdf.write(6, "Watermark Status: NOT FOUND\n")
        pdf.write(6, "Authentication: UNVERIFIED AUDIO\n")
        pdf.write(6, "Provenance: No embedded provenance metadata detected.\n")
        pdf.write(6, "Reason: This audio was never protected using the Proactive Deepfake Defense system.\n")
        
    pdf.ln(5)
    
    # Attack Analysis
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Forensic Attack Analysis", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    
    pdf.write(6, f"ML Prediction: {result.attack_analysis.get('likely_manipulation', 'UNKNOWN')}\n")
    pdf.write(6, f"Confidence: {result.attack_analysis.get('confidence', 'LOW')}\n")
    if result.attack_analysis.get("rule_label"):
        pdf.write(6, f"Rule Prediction: {result.attack_analysis.get('rule_label')}\n")
        
    rule_evidence = result.attack_analysis.get("rule_evidence", [])
    if rule_evidence:
        pdf.write(6, "Heuristic Evidence:\n")
        for item in rule_evidence:
            pdf.write(6, f"  - {item}\n")
            
    # Page 2: Detailed Forensic Feature Metrics Table
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Forensic Feature Metrics", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Computed Audio Signal and Watermark Metrics:", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    
    # Render table header
    col_w_name = 100
    col_w_val = 80
    row_h = 7
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(col_w_name, row_h, " Forensic Feature / Metric", border=1, fill=True)
    pdf.cell(col_w_val, row_h, " Computed Value", border=1, fill=True)
    pdf.ln()
    
    pdf.set_font("Helvetica", "", 10)
    
    # First row: Watermark BER
    pdf.cell(col_w_name, row_h, " Watermark Bit Error Rate (BER)", border=1)
    pdf.cell(col_w_val, row_h, f" {result.ber:.6f}", border=1)
    pdf.ln()
    
    # Other rows: Metrics
    for key, val in metrics.items():
        name = key.replace("_", " ").title()
        formatted_val = f" {val:.6f}" if isinstance(val, (int, float)) else f" {str(val)}"
        pdf.cell(col_w_name, row_h, f" {name}", border=1)
        pdf.cell(col_w_val, row_h, formatted_val, border=1)
        pdf.ln()
        
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(0, 6, "* Note: To save a PDF containing full visual charts, use browser print (Ctrl + P) with 'Background graphics' enabled.", new_x="LMARGIN", new_y="NEXT")
                
    return bytes(pdf.output())


def _generate_pdf_report_with_graphs(result, uploaded_file, metrics, explainability) -> bytes:
    from fpdf import FPDF
    import tempfile
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    # Create PDF object
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Page 1: Verification & Metadata
    pdf.add_page()
    
    # Header Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Proactive Deepfake Defense", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 8, "Forensic Verification Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)
    
    # Horizontal line separator
    pdf.line(10, 33, 200, 33)
    pdf.ln(10)
    
    # Verification Summary
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Verification Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    
    # Status highlight
    status = result.auth_card_status
    if status == "AUTHENTIC":
        pdf.set_text_color(15, 118, 110) # Teal/Green
    else:
        pdf.set_text_color(153, 27, 27) # Red
        
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, f"STATUS: {status}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0) # Reset to black
    pdf.set_font("Helvetica", "I", 10)
    pdf.multi_cell(0, 6, f"Detail: {result.final_authentication}")
    pdf.ln(5)
    
    # Provenance Metadata
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Provenance Details", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    
    if result.signature_valid:
        for key in ["creator", "timestamp", "generator", "model_version", "source_hash", "verified_hash"]:
            val = result.provenance.get(key, "N/A")
            pdf.set_font("Helvetica", "B", 10)
            pdf.write(6, f"{key.replace('_', ' ').capitalize()}: ")
            pdf.set_font("Helvetica", "", 10)
            pdf.write(6, f"{val}\n")
    else:
        pdf.write(6, "Watermark Status: NOT FOUND\n")
        pdf.write(6, "Authentication: UNVERIFIED AUDIO\n")
        pdf.write(6, "Provenance: No embedded provenance metadata detected.\n")
        pdf.write(6, "Reason: This audio was never protected using the Proactive Deepfake Defense system.\n")
        
    pdf.ln(5)
    
    # Attack Analysis
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Forensic Attack Analysis", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    
    pdf.write(6, f"ML Prediction: {result.attack_analysis.get('likely_manipulation', 'UNKNOWN')}\n")
    pdf.write(6, f"Confidence: {result.attack_analysis.get('confidence', 'LOW')}\n")
    if result.attack_analysis.get("rule_label"):
        pdf.write(6, f"Rule Prediction: {result.attack_analysis.get('rule_label')}\n")
        
    rule_evidence = result.attack_analysis.get("rule_evidence", [])
    if rule_evidence:
        pdf.write(6, "Heuristic Evidence:\n")
        for item in rule_evidence:
            pdf.write(6, f"  - {item}\n")
            
    # Paths for temporary PNG files
    temp_dir = tempfile.gettempdir()
    gauge_path = os.path.join(temp_dir, "pdf_gauge.png")
    spectral_path = os.path.join(temp_dir, "pdf_spectral.png")
    explain_path = os.path.join(temp_dir, "pdf_explain.png")
    heatmap_path = os.path.join(temp_dir, "pdf_heatmap.png")
    waveform_path = os.path.join(temp_dir, "pdf_waveform.png")
    spectrogram_path = os.path.join(temp_dir, "pdf_spectrogram.png")
    frequency_path = os.path.join(temp_dir, "pdf_frequency.png")
    
    try:
        # Load audio data to generate audio plots
        audio, sr = load_audio(uploaded_file if isinstance(uploaded_file, str) else _persist_uploaded(uploaded_file))
        
        # 1. BER Gauge plot
        fig, ax = plt.subplots(figsize=(4, 1.5))
        ber_val = float(result.ber)
        if ber_val <= 0.1:
            color_bar = "#064e3b" # Green
        elif ber_val <= 0.3:
            color_bar = "#854d0e" # Yellow
        else:
            color_bar = "#7f1d1d" # Red
            
        ax.barh([0], [ber_val], color=color_bar, height=0.4)
        ax.set_xlim(0, 1.0)
        ax.set_yticks([])
        ax.set_xlabel("Watermark Bit Error Rate (BER)")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.tight_layout()
        plt.savefig(gauge_path, dpi=150)
        plt.close()
        
        # 2. Spectral Metrics plot
        keys = ["centroid", "hf_ratio", "lf_ratio", "flatness", "edge_ratio", "column_shift_pressure"]
        values = [metrics.get(key, 0.0) for key in keys]
        display_keys = [k.replace("_", " ").title() for k in keys]
        
        fig, ax = plt.subplots(figsize=(5.5, 3))
        ax.bar(display_keys, values, color="#f59e0b")
        ax.set_ylabel("Normalized Value")
        ax.set_title("Spectral & Synchronization Metrics")
        plt.xticks(rotation=25, ha="right", fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(spectral_path, dpi=150)
        plt.close()
        
        # 3. Explainability plot
        use_shap = explainability.get("use_shap", False)
        if use_shap:
            shap_rows = explainability.get("shap_values", [])
            sorted_rows = sorted(shap_rows, key=lambda x: abs(x["shap_value"]), reverse=True)[:8]
            sorted_rows = sorted_rows[::-1]
            labels = [row["feature"].replace("_", " ").title() for row in sorted_rows]
            values = [row["shap_value"] for row in sorted_rows]
            colors = ["#ef4444" if val >= 0 else "#3b82f6" for val in values]
            title = f"SHAP Local Explanations (Predicted: {result.attack_analysis.get('likely_manipulation', 'UNKNOWN')})"
            xlabel = "SHAP Value (Contribution)"
        else:
            importance_rows = explainability.get("feature_importance", [])
            sorted_rows = importance_rows[:8][::-1]
            labels = [row["feature"].replace("_", " ").title() for row in sorted_rows]
            values = [row["importance"] for row in sorted_rows]
            colors = ["#1e3b8b"] * len(values)
            title = "Top Influential Forensic Features"
            xlabel = "Feature Importance"
            
        fig, ax = plt.subplots(figsize=(5.5, 3))
        ax.barh(labels, values, color=colors)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.savefig(explain_path, dpi=150)
        plt.close()

        # 4. Rule Heatmap / Scores
        scores = result.attack_analysis.get("rule_scores", result.attack_analysis.get("scores", {}))
        fig, ax = plt.subplots(figsize=(5.5, 2))
        labels_scores = [k.replace("_", " ").title() for k in scores.keys()]
        values_scores = list(scores.values())
        ax.barh(labels_scores, values_scores, color="#0ea5e9")
        ax.set_xlim(0, max(1.0, max(values_scores) * 1.1) if values_scores else 1.0)
        ax.set_title("Rule Score Classifier Metrics", fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=150)
        plt.close()

        # 5. Audio Waveform
        fig, ax = plt.subplots(figsize=(5.5, 2.2))
        times = np.arange(len(audio)) / float(sr)
        downsample_factor = max(1, len(audio) // 2000)
        ax.plot(times[::downsample_factor], audio[::downsample_factor], color="#3b82f6", linewidth=0.5)
        ax.set_title("Audio Waveform Profile", fontsize=10)
        ax.set_xlabel("Seconds", fontsize=8)
        ax.set_ylabel("Amplitude", fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(waveform_path, dpi=150)
        plt.close()

        # 6. Audio Spectrogram
        stft = compute_stft(audio)
        magnitude, _ = split_mag_phase(stft)
        fig, ax = plt.subplots(figsize=(5.5, 2.2))
        db_mag = 20 * np.log10(magnitude + 1e-12)
        if db_mag.shape[1] > 500:
            db_mag = db_mag[:, ::(db_mag.shape[1] // 500)]
        if db_mag.shape[0] > 128:
            db_mag = db_mag[::(db_mag.shape[0] // 128), :]
            
        im = ax.imshow(db_mag, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title("Audio Spectrogram Plot", fontsize=10)
        ax.set_xlabel("Time Frame", fontsize=8)
        ax.set_ylabel("Frequency Bin", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(spectrogram_path, dpi=150)
        plt.close()

        # 7. Frequency Profile
        fig, ax = plt.subplots(figsize=(5.5, 2.2))
        freq_profile = magnitude.mean(axis=1)
        freq_axis = np.linspace(0.0, sr / 2.0, len(freq_profile))
        if len(freq_profile) > 1000:
            freq_profile = freq_profile[::(len(freq_profile) // 1000)]
            freq_axis = freq_axis[::(len(freq_axis) // 1000)]
        ax.plot(freq_axis, freq_profile, color="#f59e0b", linewidth=1.0)
        ax.set_title("Mean Frequency Magnitude Profile", fontsize=10)
        ax.set_xlabel("Hz", fontsize=8)
        ax.set_ylabel("Mean Magnitude", fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(frequency_path, dpi=150)
        plt.close()
        
        # Embed BER gauge on first page
        pdf.ln(5)
        pdf.image(gauge_path, x=10, y=pdf.get_y(), w=90)
        
        # Page 2: Visual Explanations & Plots
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Forensic Visualizations", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(5)
        
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, "Spectral & Synchronization Metrics", new_x="LMARGIN", new_y="NEXT")
        pdf.image(spectral_path, x=15, y=pdf.get_y() + 2, w=170)
        pdf.ln(97)
        
        pdf.cell(0, 6, "Local Explainability Model Contributions", new_x="LMARGIN", new_y="NEXT")
        pdf.image(explain_path, x=15, y=pdf.get_y() + 2, w=170)

        # Page 3: Rule Classifier Heatmap & Waveform
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Classification & Signal Waveform", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(5)

        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, "Rule-based Classification Metrics", new_x="LMARGIN", new_y="NEXT")
        pdf.image(heatmap_path, x=15, y=pdf.get_y() + 2, w=170)
        pdf.ln(66)

        pdf.cell(0, 6, "Audio Waveform Amplitude Profile", new_x="LMARGIN", new_y="NEXT")
        pdf.image(waveform_path, x=15, y=pdf.get_y() + 2, w=170)

        # Page 4: Spectrogram & Frequency Profile
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Audio Spectral Inspection", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(5)

        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, "Spectrogram Time-Frequency Distribution", new_x="LMARGIN", new_y="NEXT")
        pdf.image(spectrogram_path, x=15, y=pdf.get_y() + 2, w=170)
        pdf.ln(72)

        pdf.cell(0, 6, "Mean Frequency Magnitude Profile", new_x="LMARGIN", new_y="NEXT")
        pdf.image(frequency_path, x=15, y=pdf.get_y() + 2, w=170)
        
    except Exception as e:
        pdf.ln(5)
        pdf.set_font("Helvetica", "I", 8)
        pdf.cell(0, 8, f"(Could not export visual plots to PDF: {str(e)})", new_x="LMARGIN", new_y="NEXT")
        
    # Clean up temp files
    for p in [gauge_path, spectral_path, explain_path, heatmap_path, waveform_path, spectrogram_path, frequency_path]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass
                
    return bytes(pdf.output())


def main():
    _style_console()

    global col_header_left, col_header_right
    col_header_left, col_header_right = st.columns([3, 1])
    with col_header_left:
        st.markdown('<div class="title-line">Proactive Deepfake Defense</div>', unsafe_allow_html=True)
        st.caption("Hybrid forensic verification with provenance, ML explainability, and rule evidence.")

    st.sidebar.markdown('<div style="font-size:1.3rem;font-weight:700;margin-top:10px;margin-bottom:15px;color:#f59e0b;">Navigation</div>', unsafe_allow_html=True)
    page = st.sidebar.radio("Select View Mode", ["Single-File Verification", "System Benchmark Study"])
    st.sidebar.markdown("---")

    if page == "Single-File Verification":
        with st.sidebar:
            st.markdown('<div style="font-size:1.1rem;font-weight:600;margin-bottom:10px;">Upload & Verify</div>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Audio file", type=["wav", "mp3", "flac"])
            reference_file = st.file_uploader("Optional reference/original audio", type=["wav", "mp3", "flac"], key="reference")
            mode = st.selectbox("Attack analysis mode", ["hybrid", "rules", "ml"], index=0)
            run_button = st.button("Run Forensic Verification", type="primary")

        if not uploaded_file:
            st.session_state.pop("verification_result", None)
            st.session_state.pop("last_verified_file", None)
            st.info("Upload an audio file to inspect provenance, attacks, and explainability.")
            return

        # Detect file change to clear cache if a new file is uploaded
        current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("last_verified_file") != current_file_id:
            st.session_state.pop("verification_result", None)

        temp_path = _persist_uploaded(uploaded_file)
        SERVICE.attack_mode = mode

        # Run verification and store in session state
        if run_button:
            with st.spinner("Analyzing audio..."):
                result = SERVICE.verify_file(temp_path)
                st.session_state["verification_result"] = result
                st.session_state["last_verified_file"] = current_file_id
                # Clear cached PDF report (so we generate it on-demand for the new file)
                st.session_state.pop("pdf_report_bytes_with_graphs", None)

        result = st.session_state.get("verification_result")

        if not result:
            st.info("Click 'Run Forensic Verification' in the sidebar to analyze the uploaded file.")
            return

        audio, sr = load_audio(temp_path)
        explainability = result.attack_analysis.get("explainability", {})
        metrics = result.metrics

        with col_header_right:
            format_option = st.selectbox("Export Format", ["Text Report", "PDF Report"], label_visibility="collapsed")
            if format_option == "Text Report":
                report_text = "".join(result.report_lines)
                st.download_button(
                    label="Download Report",
                    data=report_text,
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_verification_report.txt",
                    mime="text/plain",
                    type="primary",
                    use_container_width=True
                )
            elif format_option == "PDF Report":
                pdf_data = st.session_state.get("pdf_report_bytes_with_graphs")
                if pdf_data is None:
                    with st.spinner("Generating PDF..."):
                        try:
                            pdf_data = _generate_pdf_report_with_graphs(result, uploaded_file, metrics, explainability)
                            st.session_state["pdf_report_bytes_with_graphs"] = pdf_data
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                if pdf_data:
                    st.download_button(
                        label="Download Report",
                        data=pdf_data,
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_forensic_report.pdf",
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True
                    )

        # Clean status card banner
        color = result.auth_card_color
        status = result.auth_card_status
        details = result.final_authentication

        status_html = f"""
        <div style="
            border: 1px solid {color}40;
            border-left: 6px solid {color};
            border-radius: 12px;
            padding: 20px;
            background-color: {color}10;
            margin-bottom: 25px;
        ">
            <span style="
                background-color: {color};
                color: #ffffff;
                padding: 4px 12px;
                border-radius: 4px;
                font-weight: 700;
                font-size: 0.85rem;
                letter-spacing: 0.05em;
                text-transform: uppercase;
                display: inline-block;
                margin-bottom: 8px;
            ">
                {status}
            </span>
            <div style="font-size: 1.25rem; font-weight: 600; color: #ffffff;">
                {details}
            </div>
        </div>
        """
        st.markdown(status_html, unsafe_allow_html=True)

        # 3-column side-by-side layout (original design restored)
        col_auth, col_meta, col_attack = st.columns(3)
        with col_auth:
            st.subheader("Authentication Card")
            st.metric("Status", result.auth_card_status)
            st.metric("Final", result.final_authentication)
            

        with col_meta:
            st.subheader("Provenance Panel")
            if result.signature_valid:
                for key in ["creator", "timestamp", "generator", "model_version", "source_hash", "verified_hash"]:
                    st.write(f"**{key}**: {result.provenance.get(key, 'N/A')}")
            else:
                st.write("**Watermark Status:**")
                st.write("NOT FOUND")
                st.write("**Authentication:**")
                st.markdown("<span style='color:#ef4444;font-weight:bold;'>UNVERIFIED AUDIO</span>", unsafe_allow_html=True)
                st.write("**Provenance:**")
                st.write("No embedded provenance metadata detected.")
                st.write("**Reason:**")
                st.write("This audio was never protected using the Proactive Deepfake Defense system.")
        with col_attack:
            st.subheader("Hybrid Attack Panel")
            st.write(f"**ML Prediction:** {result.attack_analysis.get('likely_manipulation', 'UNKNOWN')}")
            st.write(f"**Confidence:** {result.attack_analysis.get('confidence', 'LOW')}")
            if result.attack_analysis.get("rule_label"):
                st.write(f"**Rule Prediction:** {result.attack_analysis.get('rule_label')}")
                st.write("**Rule Evidence:**")
                for item in result.attack_analysis.get("rule_evidence", []):
                    st.write(f"- {item}")

        st.subheader("Explainability Visualization")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.plotly_chart(_gauge_figure(result.ber), use_container_width=True)
        with c2:
            st.plotly_chart(_spectral_metrics_figure(metrics), use_container_width=True)

        exp_cols = st.columns(2)
        with exp_cols[0]:
            if explainability.get("use_shap"):
                st.plotly_chart(
                    _shap_explanation_figure(
                        explainability.get("shap_values", []),
                        explainability.get("likely_manipulation", "UNKNOWN")
                    ),
                    use_container_width=True
                )
                st.caption(
                    f"**SHAP Base Value:** {explainability.get('base_value', 0.0):.4f} | "
                    f"**Red bars** increase prediction likelihood; **Blue bars** decrease it."
                )
            else:
                st.plotly_chart(_feature_importance_figure(explainability.get("feature_importance", [])), use_container_width=True)
        with exp_cols[1]:
            st.plotly_chart(_rule_heatmap_figure(result.attack_analysis.get("rule_scores", result.attack_analysis.get("scores", {}))), use_container_width=True)

        st.subheader("Audio Analysis Visualization")
        waveform, spectrogram, frequency = _audio_profile_figures(audio, sr)
        audio_cols = st.columns(3)
        with audio_cols[0]:
            st.plotly_chart(waveform, use_container_width=True)
        with audio_cols[1]:
            st.plotly_chart(spectrogram, use_container_width=True)
        with audio_cols[2]:
            st.plotly_chart(frequency, use_container_width=True)

        st.subheader("Side-by-Side Comparison")
        compare_cols = st.columns(2)
        with compare_cols[0]:
            st.write("Original / reference")
            if reference_file is not None:
                st.audio(reference_file.getvalue(), format=reference_file.type or "audio/wav")
            else:
                st.info("No reference audio uploaded.")
        with compare_cols[1]:
            st.write("Uploaded / attacked audio")
            st.audio(uploaded_file.getvalue(), format=uploaded_file.type or "audio/wav")

        st.subheader("Verification Data")
        st.json(
            {
                "authentication": result.final_authentication,
                "attack_analysis": result.attack_analysis,
                "provenance": result.provenance,
                "metrics": result.metrics,
            }
        )

    else:
        # Benchmark study tab
        BENCHMARK_DIR = "output/benchmarks"
        dataset_dir = "dataset/generated"
        summary_path = os.path.join(BENCHMARK_DIR, "benchmark_summary.json")

        with st.sidebar:
            st.markdown('<div style="font-size:1.1rem;font-weight:600;margin-bottom:10px;">Benchmark Config</div>', unsafe_allow_html=True)
            repeats = st.slider("Augmentation Repeats", min_value=1, max_value=15, value=9, help="Number of augmented variants per source audio. 9 repeats gives 1000+ total evaluation samples.")
            run_benchmark_btn = st.button("Run Benchmark Study", type="primary")

        if run_benchmark_btn:
            st.info("Starting Large-Scale Benchmark Study...")
            
            # 1. Dataset Generation Progress
            gen_bar = st.progress(0)
            gen_status = st.empty()
            
            def gen_callback(curr, total):
                gen_bar.progress(curr / total)
                gen_status.text(f"Generating augmented dataset: {curr} / {total} files...")
            
            generate_dataset(dataset_dir, repeats=repeats, progress_callback=gen_callback)
            gen_status.text("Dataset generation completed! (100%)")
            
            # 2. Evaluation Progress
            verify_bar = st.progress(0)
            verify_status = st.empty()
            
            def verify_callback(curr, total):
                verify_bar.progress(curr / total)
                verify_status.text(f"Evaluating samples: {curr} / {total} files...")
            
            run_benchmark(dataset_dir, BENCHMARK_DIR, progress_callback=verify_callback)
            verify_status.text("Benchmark evaluation completed successfully! (100%)")
            
            st.success("Benchmark completed! Loading results...")
            st.rerun()

        # Display results if summary exists
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)

            st.header("Large-Scale Benchmark Results")
            st.write(f"Showing evaluation results for **{summary.get('samples', 0)}** total audio samples.")

            # Metrics row 1
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Total Evaluation Samples", summary.get("samples"))
            with c2:
                st.metric("ML Classifier Accuracy", f"{summary.get('accuracy_ml', 0.0)*100:.1f}%")
            with c3:
                st.metric("Rule Heuristics Accuracy", f"{summary.get('accuracy_rule', 0.0)*100:.1f}%")
            with c4:
                st.metric("Weighted F1 Score", f"{summary.get('f1_weighted', 0.0):.3f}")

            # Metrics row 2
            c5, c6, c7, c8 = st.columns(4)
            with c5:
                st.metric("Mean Watermark BER", f"{summary.get('ber_mean', 0.0):.4f}")
            with c6:
                st.metric("ECC Recovery Rate", f"{summary.get('ecc_recovery_rate', 0.0)*100:.1f}%")
            with c7:
                st.metric("Mean Verification Latency", f"{summary.get('latency_ms_mean', 0.0):.1f} ms")
            with c8:
                st.metric("Mean Peak Memory", f"{summary.get('memory_kb_mean', 0.0)/1024.0:.1f} MB")

            st.markdown("---")

            # Confusion Matrix & ROC Curves side by side
            col_cm, col_roc = st.columns(2)
            with col_cm:
                st.subheader("Confusion Matrix")
                cm_img_path = os.path.join(BENCHMARK_DIR, "confusion_matrix.png")
                if os.path.exists(cm_img_path):
                    st.image(cm_img_path, use_container_width=True)
                else:
                    st.warning("Confusion matrix plot not found.")
            with col_roc:
                st.subheader("Receiver Operating Characteristic (ROC)")
                roc_img_path = os.path.join(BENCHMARK_DIR, "roc_curves.png")
                if os.path.exists(roc_img_path):
                    st.image(roc_img_path, use_container_width=True)
                else:
                    st.warning("ROC curves plot not found.")

            st.markdown("---")

            # Classification report table
            st.subheader("Detailed Class Performance Report")
            report = summary.get("classification_report", {})
            if report:
                df = pd.DataFrame(report).transpose()
                metrics_indices = [idx for idx in df.index if idx not in ["accuracy", "macro avg", "weighted avg"]]
                avg_indices = [idx for idx in df.index if idx in ["accuracy", "macro avg", "weighted avg"]]
                df_sorted = df.loc[metrics_indices + avg_indices]
                st.dataframe(df_sorted.style.format(precision=3), use_container_width=True)
            else:
                st.info("Classification report details not available.")
        else:
            st.warning("No benchmark results found. Set configuration in sidebar and click 'Run Benchmark Study' to evaluate the system.")


if __name__ == "__main__":
    main()
