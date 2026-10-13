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
from matplotlib.patches import FancyBboxPatch, Rectangle
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
# FIGURE 1: STREAMLINED EDGE PROCESSING PIPELINE (TWO-COLUMN, COMPACT)
# ==============================================================================
def fig_edge_pipeline():
    fig, ax = plt.subplots(figsize=(15.2, 4.4), dpi=300)
    ax.set_xlim(0, 15.2)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    # Boundary Box for On-Device Raspberry Pi 4 (Stages 2 to 7)
    rpi_box = FancyBboxPatch((3.82, 0.38), 11.20, 3.82,
                             boxstyle="round,pad=0.08,rounding_size=0.12",
                             facecolor="#f8fafc", edgecolor="#475569",
                             linestyle="--", linewidth=1.8, zorder=0)
    ax.add_patch(rpi_box)
    ax.text(9.42, 4.05,
            "ON-DEVICE EMBEDDED COMPUTING BOUNDARY (RASPBERRY PI 4) — ZERO AUDIO STREAMING",
            ha="center", va="center", fontsize=11.5, fontweight="bold", color="#1e293b")

    cw = 2.92
    ch = 1.35

    col_x = [0.15, 3.98, 7.74, 11.50]
    top_y = 2.45
    bot_y = 0.52

    def render_card(x, y, w, h, num, title, line1, line2, border_col, fill_col, is_gateway=False):
        card = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.04,rounding_size=0.08",
                              facecolor=fill_col, edgecolor=border_col,
                              linewidth=1.8, zorder=2)
        ax.add_patch(card)

        if is_gateway:
            ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="center",
                    fontsize=13.0, fontweight="bold", color=border_col, zorder=4)
        else:
            badge = FancyBboxPatch((x + 0.12, y + h - 0.38), 0.36, 0.26,
                                   boxstyle="round,pad=0.02,rounding_size=0.04",
                                   facecolor=border_col, edgecolor=border_col, zorder=3)
            ax.add_patch(badge)
            ax.text(x + 0.30, y + h - 0.25, num, ha="center", va="center",
                    fontsize=11.5, fontweight="bold", color="white", zorder=4)
            ax.text(x + 0.58, y + h - 0.25, title, ha="left", va="center",
                    fontsize=13.0, fontweight="bold", color=border_col, zorder=4)

        # Subtle divider
        ax.plot([x + 0.12, x + w - 0.12], [y + h - 0.46, y + h - 0.46],
                color=border_col, linewidth=1.0, alpha=0.35, zorder=3)

        # Content lines
        ax.text(x + w / 2, y + 0.54, line1, ha="center", va="center",
                fontsize=11.0, fontweight="bold", color="#0f172a", zorder=4)
        ax.text(x + w / 2, y + 0.22, line2, ha="center", va="center",
                fontsize=10.0, fontweight="bold", color="#334155", zorder=4)

    # Row 1 (Top)
    render_card(col_x[0], top_y, cw, ch, "1", "Acoustic Input",
                "Continuous Electret Stream", "22,050 Hz Audio Capture",
                DARK, "#f1f5f9")
    render_card(col_x[1], top_y, cw, ch, "2", "Sliding Buffer",
                "750 ms Frame (16,537 smp)", "187.5 ms Hop (75% Overlap)",
                NAVY, "#f0f7ff")
    render_card(col_x[2], top_y, cw, ch, "3", "Conditioning",
                "DC Offset Removal (Eq. 1)", "RMS Energy Gating (Eq. 2)",
                TEAL, "#f0fdf4")
    render_card(col_x[3], top_y, cw, ch, "4", "Feature Map",
                "64-Band Log-Mel Spectrogram", "or Dynamic PCEN (Eq. 4)",
                PURPLE, "#faf5ff")

    # Row 2 (Bottom)
    render_card(col_x[0], bot_y, cw, ch, "G", "Central Gateway",
                "Spatial Triangulation (K ≥ 3)", "Civilian Privacy Preserved",
                DARK, "#f1f5f9", is_gateway=True)
    render_card(col_x[1], bot_y, cw, ch, "7", "Alert Package",
                "1-Bit Flag + Timestamp", "Zero Raw Audio Streamed",
                RUST, "#fef2f2")
    render_card(col_x[2], bot_y, cw, ch, "6", "Decision Gate",
                "Gunshot Prob τ ≥ 0.50", "Auxiliary Anomaly Gate",
                AMBER, "#fffbeb")
    render_card(col_x[3], bot_y, cw, ch, "5", "INT8 Inference",
                "TFLite Micro Runtime (ARM)", "0.09 - 11.86 ms Latency",
                GREEN, "#f0fdf4")

    # Connectors
    def draw_arrow(x1, y1, x2, y2, col):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0, mutation_scale=14),
                    zorder=5)

    # Top row forward
    draw_arrow(col_x[0] + cw, top_y + ch / 2, col_x[1], top_y + ch / 2, DARK)
    draw_arrow(col_x[1] + cw, top_y + ch / 2, col_x[2], top_y + ch / 2, NAVY)
    draw_arrow(col_x[2] + cw, top_y + ch / 2, col_x[3], top_y + ch / 2, TEAL)

    # Vertical 4 -> 5
    draw_arrow(col_x[3] + cw / 2, top_y, col_x[3] + cw / 2, bot_y + ch, PURPLE)

    # Bottom row backward: 5 -> 6
    draw_arrow(col_x[3], bot_y + ch / 2, col_x[2] + cw, bot_y + ch / 2, GREEN)

    # 6 -> 7 with YES badge
    gap_cx = (col_x[2] + col_x[1] + cw) / 2
    draw_arrow(col_x[2], bot_y + ch / 2, gap_cx + 0.28, bot_y + ch / 2, AMBER)
    draw_arrow(gap_cx - 0.28, bot_y + ch / 2, col_x[1] + cw, bot_y + ch / 2, AMBER)
    badge_yes = FancyBboxPatch((gap_cx - 0.25, bot_y + ch / 2 - 0.15), 0.50, 0.30,
                               boxstyle="round,pad=0.02,rounding_size=0.04",
                               facecolor="#fffbeb", edgecolor=AMBER, lw=1.4, zorder=6)
    ax.add_patch(badge_yes)
    ax.text(gap_cx, bot_y + ch / 2, "YES", ha="center", va="center",
            fontsize=10.0, fontweight="bold", color=AMBER, zorder=7)

    # 7 -> Gateway with RADIO badge
    gap_rg = (col_x[1] + col_x[0] + cw) / 2
    draw_arrow(col_x[1], bot_y + ch / 2, gap_rg + 0.36, bot_y + ch / 2, RUST)
    draw_arrow(gap_rg - 0.36, bot_y + ch / 2, col_x[0] + cw, bot_y + ch / 2, RUST)
    badge_radio = FancyBboxPatch((gap_rg - 0.34, bot_y + ch / 2 - 0.15), 0.68, 0.30,
                                 boxstyle="round,pad=0.02,rounding_size=0.04",
                                 facecolor="#fef2f2", edgecolor=RUST, lw=1.4, zorder=6)
    ax.add_patch(badge_radio)
    ax.text(gap_rg, bot_y + ch / 2, "RADIO", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=RUST, zorder=7)

    # Decision feedback loop: 6 -> 2
    fb_start_x = col_x[2] + 0.35
    fb_start_y = bot_y + ch
    fb_end_x = col_x[1] + cw - 0.45
    fb_end_y = top_y
    ax.annotate("", xy=(fb_end_x, fb_end_y), xytext=(fb_start_x, fb_start_y),
                arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.6, ls="--", mutation_scale=12),
                zorder=4)
    mid_x = (fb_start_x + fb_end_x) / 2
    mid_y = (fb_start_y + fb_end_y) / 2
    ax.text(mid_x - 0.35, mid_y, "NO: Advance Ring Buffer (187.5 ms)",
            ha="center", va="center", fontsize=9.0, fontweight="bold", color=SLATE,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#ffffff", edgecolor="#94a3b8", lw=1.0),
            zorder=6)

    # Bottom summary banner
    ax.text(7.60, 0.14,
            "Streamlined Telemetry: Only 1-Bit Flag, Timestamp & Confidence Transmitted  ·  Raw Audio NEVER Leaves the Edge Node",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color=RUST,
            bbox=dict(boxstyle="round,pad=0.20", facecolor="#fef2f2", edgecolor=RUST, lw=1.0), zorder=6)

    save(fig, "fig1_edge_pipeline.png")


# ==============================================================================
# FIGURE 2: MODULAR WORKING NEURAL ARCHITECTURES (TWO-COLUMN, COMPACT)
# ==============================================================================
def fig_architectures():
    fig, ax = plt.subplots(figsize=(15.2, 4.8), dpi=300)
    ax.set_xlim(0, 15.2)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    # Top front-end banner
    banner = FancyBboxPatch((0.15, 4.30), 14.90, 0.44,
                            boxstyle="round,pad=0.03,rounding_size=0.06",
                            facecolor="#e0f2fe", edgecolor=NAVY,
                            linewidth=1.8, zorder=2)
    ax.add_patch(banner)
    ax.text(7.60, 4.52,
            "SHARED FRONT-END: 750 ms Frame (16,537 smp @ 22.05 kHz)  →  DC Strip  →  RMS Energy Gate  →  INT8 Quantization",
            ha="center", va="center", fontsize=11.5, fontweight="bold", color=NAVY, zorder=3)

    cols = [
        (0.15, "Baseline 1D CNN", "92.5k params · 112.5 KB", NAVY, "#f0f7ff", [
            ("Input Waveform", "(16,537, 1) Raw Audio Samples", "#ffffff", NAVY),
            ("Convolutional Backbone", "Multi-Scale Conv1D (K=80 to 3)\nMaxPool + BatchNorm (1033 × 32)", "#eff6ff", NAVY),
            ("Classifier Head", "Global AvgPool 1D → Dense(64)\nSigmoid Head: P(Gunshot)", "#ecfdf5", GREEN),
        ]),
        (3.93, "Baseline 2D CNN", "250.0k params · 259.9 KB", BLUE, "#f8fafc", [
            ("Log-Mel Spectrogram", "64 Mel Bins × 130 Time Frames", "#ffffff", BLUE),
            ("Convolutional Backbone", "4-Stage Standard Conv2D Stack\n3×3 Filters (32 → 128) + 2×2 Pooling", "#eff6ff", BLUE),
            ("Classifier Head", "Global AvgPool 2D → Dense(64)\nSigmoid Head: P(Gunshot)", "#ecfdf5", GREEN),
        ]),
        (7.71, "Enhanced 2D CNN", "110.6k params · 123.6 KB", GREEN, "#f0fdf4", [
            ("Normalized Log-Mel", "64 × 130 + SpecAug & Mixup", "#ffffff", GREEN),
            ("Time-Preserving Backbone", "4-Stage Conv2D (16 → 128 Filters)\nEarly Pooling Only (Retains Timings)", "#f0fdf4", GREEN),
            ("Dual Classifier Heads", "Head 1: Gunshot Prob (λ₁ = 1.0)\nHead 2: Anomaly Score (λ₂ = 0.3)", "#ecfdf5", GREEN),
        ]),
        (11.49, "Robust CRNN-PCEN", "397.8k params · 478.9 KB", RUST, "#fef2f2", [
            ("Dynamic PCEN Map", "64 × 130 PCEN (α=0.98, r=0.04)\nTransducer Distortion Invariant", "#ffffff", RUST),
            ("CRNN Hybrid Backbone", "2D Conv Stack (32 → 64 Filters)\nBidirectional GRU (128 Units)", "#fff7ed", AMBER),
            ("Classifier Head", "Sequence Collapse (32 → 1) → Dense\nSigmoid Head: P(Gunshot)", "#ecfdf5", GREEN),
        ]),
    ]

    col_w = 3.56
    for x, title, meta, col_color, bg_fill, stages in cols:
        card = FancyBboxPatch((x, 0.08), col_w, 4.10,
                              boxstyle="round,pad=0.04,rounding_size=0.08",
                              facecolor=bg_fill, edgecolor=col_color,
                              linewidth=1.8, zorder=2)
        ax.add_patch(card)

        ax.text(x + col_w / 2, 3.96, title, ha="center", va="center",
                fontsize=13.0, fontweight="bold", color=col_color, zorder=3)
        ax.text(x + col_w / 2, 3.73, meta, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color="#475569", zorder=3)
        ax.plot([x + 0.12, x + col_w - 0.12], [3.58, 3.58],
                color=col_color, linewidth=1.0, alpha=0.4, zorder=3)

        box_y = [2.52, 1.34, 0.16]
        box_h = 0.96
        for idx, (b_title, b_sub, fill, border) in enumerate(stages):
            ly = box_y[idx]
            l_box = FancyBboxPatch((x + 0.14, ly), col_w - 0.28, box_h,
                                   boxstyle="round,pad=0.03,rounding_size=0.06",
                                   facecolor=fill, edgecolor=border,
                                   linewidth=1.4, zorder=3)
            ax.add_patch(l_box)
            ax.text(x + col_w / 2, ly + box_h - 0.24, b_title, ha="center", va="center",
                    fontsize=11.5, fontweight="bold", color="#0f172a", zorder=4)
            ax.text(x + col_w / 2, ly + 0.32, b_sub, ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color="#334155", zorder=4)

            # Arrow between stages
            if idx < len(stages) - 1:
                ax.annotate("", xy=(x + col_w / 2, ly - 0.04),
                            xytext=(x + col_w / 2, ly + 0.04),
                            arrowprops=dict(arrowstyle="-|>", color=col_color, lw=1.6, mutation_scale=10),
                            zorder=5)

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
