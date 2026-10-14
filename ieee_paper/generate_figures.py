#!/usr/bin/env python3
"""
Refined IEEE Publication Figure Generator - Compact & Crisp Edition
Gunshot Edge Detection Paper

Calibrated strictly for IEEE Conference page-budget optimization:
  - Double column: textwidth = 7.16 inches (compact vertical ratios ~ 3:1)
  - Single column: columnwidth = 3.5 inches (compact vertical ratios ~ 1.8:1)
Engineered with bold publication typography, TrueType vector fonts (type 42),
and 600 DPI rasterization for zero blurriness and zero text collisions.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, Arc
import numpy as np

# TrueType vector font embeddings (prevents bitmap blurriness in PDF)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial', 'Helvetica']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.edgecolor'] = '#0f172a'
matplotlib.rcParams['axes.linewidth'] = 1.3

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
ROOT_DIR = Path("/home/jeevant/Desktop/GunShot")
OVERLEAF_DIR = ROOT_DIR / "overleaf_upload"

# Curated high-contrast publication palette
NAVY   = "#0a2540"
BLUE   = "#1d63a8"
STEEL  = "#336699"
TEAL   = "#087f5b"
GREEN  = "#15803d"
AMBER  = "#b45309"
RUST   = "#991b1b"
PURPLE = "#6b21a8"
DARK   = "#0f172a"
SLATE  = "#334155"
MUTED  = "#475569"
BORDER = "#1e293b"


def save(fig, name, do_tight=True):
    if do_tight:
        fig.tight_layout()
    base = Path(name).stem
    targets = [OUT]
    if OVERLEAF_DIR.exists():
        targets.append(OVERLEAF_DIR)
        if (OVERLEAF_DIR / "figures").exists():
            targets.append(OVERLEAF_DIR / "figures")

    for p in targets:
        if p.exists():
            fig.savefig(p / f"{base}.pdf", bbox_inches="tight", facecolor="white")
            fig.savefig(p / f"{base}.png", bbox_inches="tight", facecolor="white", dpi=600)
    plt.close(fig)
    print(f"Generated: {base}.pdf & {base}.png (600 DPI, TrueType Vector)")


# ==============================================================================
# ==============================================================================
# FIGURE 1: STREAMLINED EDGE PROCESSING PIPELINE (PUBLICATION-GRADE)
# ==============================================================================
def fig_edge_pipeline():
    fig, ax = plt.subplots(figsize=(14.2, 3.4), dpi=300)
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 3.4)
    ax.axis("off")

    # Boundary box: Raspberry Pi 4 on-device boundary
    rpi_box = FancyBboxPatch((3.50, 0.28), 10.50, 2.95,
                             boxstyle="round,pad=0.04,rounding_size=0.08",
                             facecolor="#f8fafc", edgecolor="#475569",
                             linestyle="--", linewidth=1.5, zorder=0)
    ax.add_patch(rpi_box)
    ax.text(8.75, 3.10, "ON-DEVICE EMBEDDED COMPUTING BOUNDARY (RASPBERRY PI 4) — ZERO AUDIO STREAMING",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color="#334155", zorder=2)

    bw = 2.45
    bh = 0.95
    col_x = [0.15, 3.65, 7.15, 10.65]
    top_y = 1.88
    bot_y = 0.45

    def draw_block(x, y, num, title, subtitle, border_color=BORDER, fill_color="#ffffff", text_color=DARK):
        box = FancyBboxPatch((x, y), bw, bh,
                             boxstyle="round,pad=0.02,rounding_size=0.06",
                             facecolor=fill_color, edgecolor=border_color,
                             linewidth=1.4, zorder=3)
        ax.add_patch(box)
        # 2-3 word title
        if num:
            ax.text(x + bw/2, y + bh - 0.30, f"{num}. {title}",
                    ha="center", va="center", fontsize=10.2, fontweight="bold", color=text_color, zorder=4)
        else:
            ax.text(x + bw/2, y + bh - 0.30, title,
                    ha="center", va="center", fontsize=10.2, fontweight="bold", color=text_color, zorder=4)
        # 1-line parameter subtitle
        ax.text(x + bw/2, y + 0.28, subtitle,
                ha="center", va="center", fontsize=8.2, fontweight="bold", color="#475569", zorder=4)

    # Top row (Stages 1-4)
    draw_block(col_x[0], top_y, "1", "Acoustic Input", "22,050 Hz Continuous Stream")
    draw_block(col_x[1], top_y, "2", "Sliding Buffer", "750 ms Frame · 187.5 ms Hop")
    draw_block(col_x[2], top_y, "3", "Conditioning", "DC Stripping & RMS Gate")
    draw_block(col_x[3], top_y, "4", "Feature Map", "64-Band Log-Mel / PCEN")

    # Bottom row (Stages 5-7 + Gateway)
    draw_block(col_x[3], bot_y, "5", "INT8 Inference", "ARM Cortex TFLite Micro")
    draw_block(col_x[2], bot_y, "6", "Decision Gate", "Gunshot Prob Threshold τ ≥ 0.50", border_color=AMBER, fill_color="#fffbeb")
    draw_block(col_x[1], bot_y, "7", "Alert Package", "1-Bit Flag + Timestamp (≤ 8 B)", border_color=RUST, fill_color="#fef2f2")
    draw_block(col_x[0], bot_y, "", "Central Gateway", "Spatial Triangulation (K ≥ 3)", border_color=TEAL, fill_color="#f0fdfa")

    # Clean Connecting Arrows
    def arrow(x1, y1, x2, y2, col=BORDER):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6, mutation_scale=11),
                    zorder=5)

    # Top row forward
    arrow(col_x[0] + bw, top_y + bh/2, col_x[1], top_y + bh/2)
    arrow(col_x[1] + bw, top_y + bh/2, col_x[2], top_y + bh/2)
    arrow(col_x[2] + bw, top_y + bh/2, col_x[3], top_y + bh/2)

    # Vertical 4 -> 5
    arrow(col_x[3] + bw/2, top_y, col_x[3] + bw/2, bot_y + bh)

    # Bottom row backward: 5 -> 6
    arrow(col_x[3], bot_y + bh/2, col_x[2] + bw, bot_y + bh/2)

    # 6 -> 7 (YES)
    arrow(col_x[2], bot_y + bh/2, col_x[1] + bw, bot_y + bh/2, col=AMBER)
    ax.text((col_x[2] + col_x[1] + bw)/2, bot_y + bh/2 + 0.16, "YES (Threat)",
            ha="center", va="center", fontsize=7.8, fontweight="bold", color=AMBER, zorder=6)

    # 7 -> Gateway (RADIO)
    arrow(col_x[1], bot_y + bh/2, col_x[0] + bw, bot_y + bh/2, col=RUST)
    ax.text((col_x[1] + col_x[0] + bw)/2, bot_y + bh/2 + 0.16, "Low-Power RF",
            ha="center", va="center", fontsize=7.8, fontweight="bold", color=RUST, zorder=6)

    # Clean Feedback Loop: 6 -> 2 (NO: Advance Hop)
    fb_start_x = col_x[2] + bw/2
    fb_mid_y = (top_y + bot_y + bh)/2
    fb_end_x = col_x[1] + bw/2

    ax.plot([fb_start_x, fb_start_x], [bot_y + bh, fb_mid_y], color="#64748b", lw=1.4, ls="--", zorder=4)
    ax.plot([fb_start_x, fb_end_x], [fb_mid_y, fb_mid_y], color="#64748b", lw=1.4, ls="--", zorder=4)
    ax.annotate("", xy=(fb_end_x, top_y), xytext=(fb_end_x, fb_mid_y),
                arrowprops=dict(arrowstyle="-|>", color="#64748b", lw=1.4, ls="--", mutation_scale=10),
                zorder=4)

    # Clean pill badge along feedback line
    fb_badge_x = (fb_start_x + fb_end_x)/2
    ax.text(fb_badge_x, fb_mid_y, " NO: Advance Ring Buffer (187.5 ms) ",
            ha="center", va="center", fontsize=8.0, fontweight="bold", color="#334155",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#ffffff", edgecolor="#94a3b8", lw=0.9),
            zorder=6)

    save(fig, "fig1_edge_pipeline.png")


# ==============================================================================
# FIGURE 2: MODULAR WORKING NEURAL ARCHITECTURES (TOPOLOGY MATRIX)
# ==============================================================================
def fig_architectures():
    fig, ax = plt.subplots(figsize=(14.2, 4.4), dpi=300)
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    # Top Banner: Shared Front-End
    banner = FancyBboxPatch((0.15, 3.82), 13.90, 0.44,
                            boxstyle="round,pad=0.02,rounding_size=0.06",
                            facecolor="#0f172a", edgecolor="#38bdf8",
                            linewidth=1.4, zorder=2)
    ax.add_patch(banner)
    ax.text(7.10, 4.04,
            "SHARED EMBEDDED FRONT-END: 750 ms Frame (16,537 smp @ 22.05 kHz)  →  DC Bias Removal (Eq. 1)  →  RMS Energy Gate (Eq. 2)",
            ha="center", va="center", fontsize=9.8, fontweight="bold", color="#f8fafc", zorder=3)

    # Column Category Headers
    col_x = [0.15, 3.15, 6.75, 10.75]
    col_w = [2.70, 3.30, 3.70, 3.30]

    ax.text(col_x[0] + col_w[0]/2, 3.56, "Neural Topology", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="#334155")
    ax.text(col_x[1] + col_w[1]/2, 3.56, "Input Representation", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="#334155")
    ax.text(col_x[2] + col_w[2]/2, 3.56, "Feature Extraction Backbone", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="#334155")
    ax.text(col_x[3] + col_w[3]/2, 3.56, "Classification & Decision Head", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="#334155")

    # 4 Model Rows
    models = [
        ("Baseline 1D CNN", "End-to-End Raw Waveform", "#f8fafc", BORDER,
         ("Raw Audio Waveform", "(16,537, 1) Float32 Samples"),
         ("Multi-Scale Conv1D Stack", "K=80...3 · MaxPool1D · BatchNorm"),
         ("Global Pool Classifier", "Global AvgPool 1D → Dense → P(Shot)")),

        ("Baseline 2D CNN", "Standard Time-Frequency", "#f8fafc", BORDER,
         ("Log-Mel Spectrogram", "64 Mel Bins × 130 Time Frames"),
         ("4-Stage Conv2D Stack", "32 → 128 Filters (3×3) + 2×2 Pooling"),
         ("Global Pool Classifier", "Global AvgPool 2D → Dense → P(Shot)")),

        ("Enhanced 2D CNN", "Primary Proposed Model", "#f0fdf4", GREEN,
         ("Normalized Mel Map", "SpecAugment & MixUp Augmentations"),
         ("Time-Preserving Backbone", "4-Stage Conv2D (16→128) · Early Pool"),
         ("Dual Classifier Heads", "Head 1: P(Shot) · Head 2: Anomaly Veto")),

        ("Robust CRNN-PCEN", "Acoustic Transducer Invariant", "#f8fafc", BORDER,
         ("Dynamic PCEN Map", "64 × 130 PCEN (α=0.98, r=0.04)"),
         ("CRNN Hybrid Backbone", "Conv2D (32→64) + Bi-GRU (128 Units)"),
         ("Sequence Collapse Head", "Temporal Collapse (32→1) → Dense → P(Shot)"))
    ]

    row_y = [2.62, 1.82, 1.02, 0.22]
    rh = 0.62

    for i, (m_name, m_tag, m_bg, m_border, s1, s2, s3) in enumerate(models):
        y = row_y[i]
        is_prop = (i == 2)

        # Col 0: Model Name Card
        b0 = FancyBboxPatch((col_x[0], y), col_w[0], rh,
                            boxstyle="round,pad=0.02,rounding_size=0.05",
                            facecolor="#ecfdf5" if is_prop else "#f1f5f9",
                            edgecolor=GREEN if is_prop else "#64748b",
                            linewidth=1.6 if is_prop else 1.2, zorder=3)
        ax.add_patch(b0)
        ax.text(col_x[0] + col_w[0]/2, y + rh - 0.20, m_name, ha="center", va="center",
                fontsize=9.2, fontweight="bold", color=GREEN if is_prop else DARK, zorder=4)
        ax.text(col_x[0] + col_w[0]/2, y + 0.18, m_tag, ha="center", va="center",
                fontsize=7.4, fontweight="bold", color=GREEN if is_prop else MUTED, zorder=4)

        # Arrow 0 -> 1
        ax.annotate("", xy=(col_x[1], y + rh/2), xytext=(col_x[0] + col_w[0], y + rh/2),
                    arrowprops=dict(arrowstyle="-|>", color=m_border, lw=1.3, mutation_scale=9), zorder=4)

        # Col 1: Input Representation
        b1 = FancyBboxPatch((col_x[1], y), col_w[1], rh,
                            boxstyle="round,pad=0.02,rounding_size=0.05",
                            facecolor=m_bg, edgecolor=m_border, lw=1.5 if is_prop else 1.1, zorder=3)
        ax.add_patch(b1)
        ax.text(col_x[1] + col_w[1]/2, y + rh - 0.20, s1[0], ha="center", va="center",
                fontsize=8.8, fontweight="bold", color=DARK, zorder=4)
        ax.text(col_x[1] + col_w[1]/2, y + 0.18, s1[1], ha="center", va="center",
                fontsize=7.4, fontweight="bold", color=MUTED, zorder=4)

        # Arrow 1 -> 2
        ax.annotate("", xy=(col_x[2], y + rh/2), xytext=(col_x[1] + col_w[1], y + rh/2),
                    arrowprops=dict(arrowstyle="-|>", color=m_border, lw=1.3, mutation_scale=9), zorder=4)

        # Col 2: Neural Backbone
        b2 = FancyBboxPatch((col_x[2], y), col_w[2], rh,
                            boxstyle="round,pad=0.02,rounding_size=0.05",
                            facecolor=m_bg, edgecolor=m_border, lw=1.5 if is_prop else 1.1, zorder=3)
        ax.add_patch(b2)
        ax.text(col_x[2] + col_w[2]/2, y + rh - 0.20, s2[0], ha="center", va="center",
                fontsize=8.8, fontweight="bold", color=DARK, zorder=4)
        ax.text(col_x[2] + col_w[2]/2, y + 0.18, s2[1], ha="center", va="center",
                fontsize=7.4, fontweight="bold", color=MUTED, zorder=4)

        # Arrow 2 -> 3
        ax.annotate("", xy=(col_x[3], y + rh/2), xytext=(col_x[2] + col_w[2], y + rh/2),
                    arrowprops=dict(arrowstyle="-|>", color=m_border, lw=1.3, mutation_scale=9), zorder=4)

        # Col 3: Classifier Head
        b3 = FancyBboxPatch((col_x[3], y), col_w[3], rh,
                            boxstyle="round,pad=0.02,rounding_size=0.05",
                            facecolor=m_bg, edgecolor=m_border, lw=1.5 if is_prop else 1.1, zorder=3)
        ax.add_patch(b3)
        ax.text(col_x[3] + col_w[3]/2, y + rh - 0.20, s3[0], ha="center", va="center",
                fontsize=8.8, fontweight="bold", color=DARK, zorder=4)
        ax.text(col_x[3] + col_w[3]/2, y + 0.18, s3[1], ha="center", va="center",
                fontsize=7.4, fontweight="bold", color=MUTED, zorder=4)

    save(fig, "fig_architectures.png")


# ==============================================================================
# FIGURE 3: TWO-PANEL 300-CLIP BENCHMARK (TWO-COLUMN, COMPACT)
# ==============================================================================
def fig_recall_specificity():
    models = ["Baseline 1D", "Baseline 2D", "Enhanced 2D", "CRNN-PCEN"]
    sliding_recall = [45, 66, 67, 57]
    peak_recall    = [30, 45, 38, 33]

    sliding_spec   = [85, 88, 82, 86]
    peak_spec      = [95, 94, 99, 97]

    x = np.arange(len(models))
    w = 0.32

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 3.8), dpi=300)

    # Subplot (a): Gunshot Recall
    b1 = ax1.bar(x - w / 2, sliding_recall, w, label="Sliding Window (750 ms)", color=NAVY, hatch="//", edgecolor="#0f172a", lw=1.1)
    b2 = ax1.bar(x + w / 2, peak_recall,    w, label="Peak-Centered Crop", color=AMBER, hatch="..", edgecolor="#0f172a", lw=1.1)

    for b in b1:
        val = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, val + 2.0, f"{int(val)}%",
                 ha="center", va="bottom", fontsize=11.0, fontweight="bold", color=NAVY)
    for b in b2:
        val = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, val + 2.0, f"{int(val)}%",
                 ha="center", va="bottom", fontsize=11.0, fontweight="bold", color=AMBER)

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=11.5, fontweight="bold")
    ax1.tick_params(axis="y", labelsize=11)
    ax1.set_ylim(0, 95)
    ax1.set_ylabel("Gunshot Recall Rate (%)", fontsize=12.0, fontweight="bold")
    ax1.set_title("(a) Gunshot Detection Recall", fontsize=12.5, fontweight="bold", pad=10)
    ax1.grid(axis="y", linestyle=":", alpha=0.6, linewidth=1.0)

    # Subplot (b): Ambient Specificity
    b3 = ax2.bar(x - w / 2, sliding_spec, w, label="Sliding Window (750 ms)", color=STEEL, edgecolor="#0f172a", lw=1.1)
    b4 = ax2.bar(x + w / 2, peak_spec,    w, label="Peak-Centered Crop", color="#c2410c", edgecolor="#0f172a", lw=1.1)

    for b in b3:
        val = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, val + 2.0, f"{int(val)}%",
                 ha="center", va="bottom", fontsize=11.0, fontweight="bold", color=STEEL)
    for b in b4:
        val = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, val + 2.0, f"{int(val)}%",
                 ha="center", va="bottom", fontsize=11.0, fontweight="bold", color="#c2410c")

    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=11.5, fontweight="bold")
    ax2.tick_params(axis="y", labelsize=11)
    ax2.set_ylim(60, 116)
    ax2.set_ylabel("Ambient Rejection Specificity (%)", fontsize=12.0, fontweight="bold")
    ax2.set_title("(b) Ambient Sound Specificity", fontsize=12.5, fontweight="bold", pad=10)
    ax2.grid(axis="y", linestyle=":", alpha=0.6, linewidth=1.0)

    # Clean shared legend at the very top with clear separation
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98),
               ncol=2, fontsize=11.0, frameon=True, facecolor="#f8fafc", edgecolor="#94a3b8")

    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.08, right=0.96, wspace=0.22)
    save(fig, "fig2_300clip_recall.png", do_tight=False)


# ==============================================================================
# FIGURE 4: SLEEK LATENCY VS REAL-TIME HOP BUDGET (SINGLE-COLUMN, COMPACT)
# ==============================================================================
def fig_latency():
    fig, ax = plt.subplots(figsize=(7.6, 3.6), dpi=300)

    models = [
        "Enhanced 2D CNN (Dual Head)",
        "Baseline 2D CNN (Log-Mel)",
        "Baseline 1D CNN (Raw Audio)",
        "Enhanced 1D CNN (Quant Fail)",
        "Robust CRNN-PCEN (Bi-GRU)",
    ]
    latencies = [0.09, 2.15, 2.29, 2.69, 11.86]
    hop_util  = [0.05, 1.15, 1.22, 1.43, 6.33]
    colors = [TEAL, BLUE, STEEL, "#64748b", NAVY]

    y_pos = np.arange(len(models))
    bars = ax.barh(y_pos, latencies, height=0.52, color=colors, edgecolor="#0f172a", lw=1.1)

    for i, (b, lat, util) in enumerate(zip(bars, latencies, hop_util)):
        x_val = b.get_width()
        text_str = f" {lat:.2f} ms ({util:.1f}%)" if lat >= 0.5 else f" {lat:.2f} ms (0.0%)"
        ax.text(x_val + 0.35, y_pos[i], text_str, va="center", ha="left",
                fontsize=10.5, fontweight="bold", color="#0f172a")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=10.5, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 25.0)
    ax.tick_params(axis="x", labelsize=11)
    ax.set_xlabel("Mean INT8 Latency (ms) per 750 ms Frame", fontsize=11.5, fontweight="bold")
    ax.set_title("Edge Inference Latency on Raspberry Pi 4\n(187.5 ms Real-Time Hop Constraint)",
                 fontsize=12.0, fontweight="bold", pad=8)
    ax.grid(axis="x", linestyle=":", alpha=0.6, linewidth=1.0)

    # Callout badge in upper right quadrant
    ax.text(17.8, 0.45,
            "Real-Time Hop Budget: 187.5 ms\nMax Load < 6.4% CPU Hop\nZero Frame Drop Across 100 Runs",
            ha="center", va="center", fontsize=9.0, fontweight="bold", color="#15803d",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f0fdf4", edgecolor="#15803d", lw=1.2))

    save(fig, "fig3_latency.png")


# ==============================================================================
# FIGURE 5: DOMAIN SHIFT GENERALIZATION (SINGLE-COLUMN, COMPACT)
# ==============================================================================
def fig_lab_vs_external():
    fig, ax = plt.subplots(figsize=(7.8, 4.0), dpi=300)

    models = ["Baseline 1D", "CRNN-PCEN", "Enhanced 2D"]
    holdout = [99.7, 99.7, 99.6]
    field   = [30.0, 60.0, 85.0]

    x = np.arange(len(models))
    w = 0.32

    b1 = ax.bar(x - w / 2, holdout, w, label="Hold-Out Validation (1,442 clips)",
                color=NAVY, edgecolor="#0f172a", lw=1.1)
    b2 = ax.bar(x + w / 2, field,   w, label="In-the-Wild Audio (75 clips)",
                color="#b45309", hatch="//", edgecolor="#0f172a", lw=1.1)

    for b in b1:
        val = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, val + 2.0, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=NAVY)
    for b in b2:
        val = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, val + 2.0, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#b45309")

    # Delta badges
    deltas = [
        ("Δ = -69.7%\n(Overfit Collapse)", RUST, "#fef2f2"),
        ("Δ = -39.7%\n(Robust PCEN)", AMBER, "#fffbeb"),
        ("Δ = -14.6%\n(Superior Generalized)", GREEN, "#f0fdf4"),
    ]
    for i, (txt, border_col, fill_col) in enumerate(deltas):
        ax.text(x[i], 122, txt, ha="center", va="center",
                fontsize=9.0, fontweight="bold", color=border_col,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=fill_col, edgecolor=border_col, lw=1.1))

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11.5, fontweight="bold")
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylim(0, 150)
    ax.set_ylabel("Gunshot Recall Rate (%)", fontsize=11.5, fontweight="bold")
    ax.set_title("Transducer Domain Shift Generalization\n(Hold-Out Split vs. External Field Audio)",
                 fontsize=12.0, fontweight="bold", pad=8)
    ax.legend(loc="upper center", ncol=2, fontsize=9.5, frameon=True, facecolor="#f8fafc", edgecolor="#94a3b8")
    ax.grid(axis="y", linestyle=":", alpha=0.6, linewidth=1.0)

    save(fig, "fig4_lab_vs_external.png")


# ==============================================================================
# FIGURE 6: ENSEMBLE 3-WAY CONFUSION MATRIX (SINGLE-COLUMN, COMPACT)
# ==============================================================================
def fig_confusion():
    fig, ax = plt.subplots(figsize=(6.2, 4.6), dpi=300)

    matrix = np.array([
        [77, 15,  8],
        [44, 46, 10],
        [20, 38, 42]
    ])

    rows = ["True Gunshots\n(100 clips)", "True Imposters\n(100 clips)", "True Ambient\n(100 clips)"]
    cols = ["Gunshot\n(Target)", "Imposter\n(Percussive)", "Ambient\n(Noise)"]

    colors = [
        ["#bbf7d0", "#fef3c7", "#fef3c7"],
        ["#fee2e2", "#dcfce7", "#f1f5f9"],
        ["#fee2e2", "#f1f5f9", "#dcfce7"]
    ]

    for i in range(3):
        for j in range(3):
            val = matrix[i, j]
            pct = val
            rect = Rectangle((j, 2 - i), 0.96, 0.96, facecolor=colors[i][j],
                             edgecolor="#475569", lw=1.5, zorder=2)
            ax.add_patch(rect)

            text_color = "#15803d" if (i == j and i != 1) else ("#991b1b" if (j == 0 and i > 0) else ("#b45309" if (i == 0 and j > 0) else "#1e293b"))
            ax.text(j + 0.48, 2 - i + 0.58, f"{val}", ha="center", va="center",
                    fontsize=20.0, fontweight="bold", color=text_color, zorder=3)
            ax.text(j + 0.48, 2 - i + 0.28, f"({pct}%)", ha="center", va="center",
                    fontsize=12.0, fontweight="bold", color=text_color, zorder=3)

    ax.set_xlim(-0.02, 2.98)
    ax.set_ylim(-0.02, 2.98)
    ax.set_xticks([0.48, 1.48, 2.48])
    ax.set_xticklabels(cols, fontsize=10.5, fontweight="bold")
    ax.set_yticks([0.48, 1.48, 2.48])
    ax.set_yticklabels(list(reversed(rows)), fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Predicted Acoustic Class", fontsize=11.5, fontweight="bold", labelpad=8)
    ax.set_ylabel("Ground Truth Class", fontsize=11.5, fontweight="bold", labelpad=8)
    ax.set_title("Ensemble 3-Way Classification Matrix\n(300 Diverse Audio Clips)",
                 fontsize=12.0, fontweight="bold", pad=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    save(fig, "fig5_ensemble_confusion.png")


# ==============================================================================
# FIGURE 7: PCEN DOMAIN DISTORTION INVARIANCE (SINGLE-COLUMN, COMPACT)
# ==============================================================================
def fig_pcen():
    fig, ax = plt.subplots(figsize=(7.4, 3.8), dpi=300)

    taus = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    clean_rec     = [69, 64, 62, 61, 61, 60, 59]
    distorted_rec = [69, 67, 66, 63, 61, 61, 59]

    ax.plot(taus, clean_rec, marker="o", markersize=10, lw=3.0, color=NAVY,
            label="Clean Audio (Calibrated)", zorder=3)
    ax.plot(taus, distorted_rec, marker="s", markersize=10, lw=3.0, color="#991b1b",
            linestyle="--", label="Distorted Audio (Degraded)", zorder=3)

    ax.fill_between(taus, clean_rec, distorted_rec, color="#7c3aed", alpha=0.15, zorder=2)

    ax.plot([0.50, 0.70], [61, 59], marker="o", markersize=14, color=GREEN,
            linestyle="none", zorder=4)

    # Label key points
    ax.text(0.10, 71.5, "69%", ha="center", fontsize=11.0, fontweight="bold", color=NAVY)
    ax.text(0.30, 68.0, "66%", ha="center", fontsize=11.0, fontweight="bold", color="#991b1b")
    ax.text(0.30, 59.5, "62%", ha="center", fontsize=11.0, fontweight="bold", color=NAVY)
    ax.text(0.50, 63.5, "61%", ha="center", fontsize=11.0, fontweight="bold", color=GREEN)
    ax.text(0.70, 61.5, "59%", ha="center", fontsize=11.0, fontweight="bold", color=GREEN)

    # Callout annotation
    ax.annotate("Zero Recall Degradation\nat τ = 0.50 & 0.70\n(Clean = Distorted)",
                xy=(0.50, 61), xytext=(0.10, 53.5),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.8),
                fontsize=9.0, fontweight="bold", color=GREEN,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#f0fdf4", edgecolor=GREEN, lw=1.1),
                zorder=5)

    ax.set_xlim(0.05, 0.75)
    ax.set_ylim(50, 78)
    ax.set_xticks(taus)
    ax.tick_params(axis="both", labelsize=10.5)
    ax.set_xlabel("Decision Threshold (τ)", fontsize=11.5, fontweight="bold")
    ax.set_ylabel("Gunshot Recall Rate (%)", fontsize=11.5, fontweight="bold")
    ax.set_title("CRNN-PCEN: Domain Shift Invariance\n(Simulated Low-Cost Capsule Distortion)",
                 fontsize=12.0, fontweight="bold", pad=8)
    ax.legend(loc="upper right", fontsize=9.5, frameon=True, facecolor="#ffffff", edgecolor="#94a3b8")
    ax.grid(True, linestyle=":", alpha=0.6, linewidth=1.0)

    save(fig, "fig6_pcen_distortion.png")


if __name__ == "__main__":
    print("Generating refined, compact publication figures...")
    fig_edge_pipeline()
    fig_architectures()
    fig_recall_specificity()
    fig_latency()
    fig_lab_vs_external()
    fig_confusion()
    fig_pcen()
    print("All 7 IEEE publication figures generated successfully at ultra-high resolution.")
