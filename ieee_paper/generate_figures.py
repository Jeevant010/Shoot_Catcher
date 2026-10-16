#!/usr/bin/env python3
"""
Refined IEEE Publication Figure Generator - Bent Pipeline & High-Legibility Edition
Gunshot Edge Detection Paper

Calibrated strictly for IEEE Conference double-column and single-column budgets:
  - Fig 1: Bent 2-row layout with large, high-legibility typography (~12 pt bold)
  - Fig 2: Spacious 3-model neural topology with large fonts (~11-12 pt bold)
  - Fig 3: Single-column 2-subplot stacked benchmark (a: Recall, b: Specificity)
  - Fig 4: Single-column latency bar chart with hop budget callout
  - Fig 5: Single-column domain shift bar chart with clean delta badges
  - Fig 6: Single-column confusion matrix with star markers
  - Fig 7: Single-column PCEN distortion invariance curve
  - Pure Monochrome (B&W/grayscale) with distinct hatches (//, xx, ..) and markers (●, ■, ▲, ◆, ★)
  - All titles and captions strictly 2 to 5 words
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
matplotlib.rcParams['axes.edgecolor'] = '#000000'
matplotlib.rcParams['axes.linewidth'] = 1.2

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
ROOT_DIR = Path("/home/jeevant/Desktop/GunShot")
OVERLEAF_DIR = ROOT_DIR / "overleaf_upload"

# Pure Monochrome Publication Palette
BLACK = "#000000"
WHITE = "#ffffff"
DARK_GRAY = "#333333"
MID_GRAY = "#808080"
LIGHT_GRAY = "#d4d4d4"
VERY_LIGHT = "#f5f5f5"


def save(fig, name, do_tight=True):
    if do_tight:
        fig.tight_layout()
    base = Path(name).stem
    targets = [OUT]
    if OVERLEAF_DIR.exists():
        targets.append(OVERLEAF_DIR)
        targets.append(OVERLEAF_DIR / "figures")
        targets.append(OVERLEAF_DIR / "ieee_paper" / "figures")

    for p in targets:
        if p.exists():
            fig.savefig(p / f"{base}.pdf", bbox_inches="tight", facecolor="white")
            fig.savefig(p / f"{base}.png", bbox_inches="tight", facecolor="white", dpi=600)
    plt.close(fig)
    print(f"Generated Figure: {base} (.pdf + .png)")


# ==============================================================================
# FIGURE 1: BENT 2-ROW EDGE PROCESSING PIPELINE (LARGE LEGIBLE TYPOGRAPHY)
# ==============================================================================
def fig_edge_pipeline():
    # 2-Row Bent Academic Pipeline with generous spacing and crystal-clear typography
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=300)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    bw = 2.24
    bh = 1.02
    y_bot = 0.52
    y_top = 3.12
    x_cols = [0.22, 2.80, 5.38, 7.96]

    # Raspberry Pi 4 Embedded Node Boundary (Encloses Col 1, 2, 3 in both rows)
    node_top = 4.60
    node_bot = 0.30
    node_box = FancyBboxPatch((2.60, node_bot), 7.78, node_top - node_bot,
                              boxstyle="round,pad=0.03,rounding_size=0.08",
                              facecolor=WHITE, edgecolor=BLACK,
                              linestyle="--", linewidth=1.5, zorder=0)
    ax.add_patch(node_box)
    
    # Boundary title with generous breathing room above top boxes
    ax.text(2.80, 4.38, "Embedded Node Boundary (Raspberry Pi 4)",
            ha="left", va="center", fontsize=11.5, fontweight="bold", color=BLACK, zorder=2)

    boxes = [
        # (col, row, title, subtitle)
        (0, "top", "Acoustic Input", "22.05 kHz Audio Stream"),
        (1, "top", "1. Sliding Buffer", "750 ms (187.5 ms Hop)"),
        (2, "top", "2. Conditioning", "DC Strip & RMS Gate"),
        (3, "top", "3. Feature Map", "64-Band Mel / PCEN"),
        (3, "bot", "4. INT8 Inference", "TFLite Micro (ARM)"),
        (2, "bot", "5. Decision Gate", r"Threshold $\tau \geq 0.50$"),
        (0, "bot", "Central Gateway", "Alert & TOA Logging"),
    ]

    for col, row, title, sub in boxes:
        y = y_top if row == "top" else y_bot
        x = x_cols[col]
        box = FancyBboxPatch((x, y), bw, bh,
                             boxstyle="round,pad=0.02,rounding_size=0.06",
                             facecolor=WHITE, edgecolor=BLACK, linewidth=1.4, zorder=3)
        ax.add_patch(box)
        ax.text(x + bw/2, y + bh * 0.64, title,
                ha="center", va="center", fontsize=11.2, fontweight="bold", color=BLACK, zorder=4)
        ax.text(x + bw/2, y + bh * 0.30, sub,
                ha="center", va="center", fontsize=9.2, color=DARK_GRAY, zorder=4)

    # Top row forward arrows
    for i in range(3):
        ax.annotate("", xy=(x_cols[i+1], y_top + bh/2), xytext=(x_cols[i] + bw, y_top + bh/2),
                    arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=1.4, mutation_scale=12), zorder=5)

    # Bend arrow: 3. Feature Map (Top Right) -> 4. INT8 Inference (Bottom Right)
    ax.annotate("", xy=(x_cols[3] + bw/2, y_bot + bh), xytext=(x_cols[3] + bw/2, y_top),
                arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=1.6, mutation_scale=14), zorder=5)

    # Bottom row arrow: 4 -> 5 (Right to Left)
    ax.annotate("", xy=(x_cols[2] + bw, y_bot + bh/2), xytext=(x_cols[3], y_bot + bh/2),
                arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=1.4, mutation_scale=12), zorder=5)

    # --- Symmetrical, Spacious Feedback Loop (5 -> 1: No Threat / Advance Hop) ---
    fb_x5 = x_cols[2] + bw * 0.70  # 5.38 + 1.57 = 6.95
    fb_x1 = x_cols[1] + bw * 0.35  # 2.80 + 0.78 = 3.58
    fb_mid_y = (y_bot + bh + y_top) / 2  # 2.33

    # Up from Decision Gate
    ax.plot([fb_x5, fb_x5], [y_bot + bh, fb_mid_y], color=BLACK, lw=1.3, ls="--", zorder=4)
    # Horizontal left connecting smoothly
    ax.plot([fb_x5, fb_x1], [fb_mid_y, fb_mid_y], color=BLACK, lw=1.3, ls="--", zorder=4)
    # Up into Sliding Buffer with arrow
    ax.annotate("", xy=(fb_x1, y_top), xytext=(fb_x1, fb_mid_y),
                arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=1.3, ls="--", mutation_scale=12),
                zorder=4)

    # Centered feedback badge (2 compact lines so it doesn't crowd horizontal line turns)
    ax.text((fb_x5 + fb_x1)/2, fb_mid_y,
            "No Threat Detected\nAdvance Hop (187.5 ms)",
            ha="center", va="center", fontsize=8.8, fontweight="bold", color=BLACK,
            linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.30,rounding_size=0.06", facecolor=WHITE, edgecolor=BLACK, lw=1.1),
            zorder=6)

    # --- RF Alert Telemetry: 5 -> Central Gateway (Across Boundary) ---
    arrow_y = y_bot + bh/2
    # Clean, unbroken arrow from Decision Gate across boundary into Central Gateway
    ax.annotate("", xy=(x_cols[0] + bw, arrow_y), xytext=(x_cols[2], arrow_y),
                arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=1.6, mutation_scale=14), zorder=5)
    
    alert_mid_x = (x_cols[2] + x_cols[0] + bw) / 2
    # Floating telemetry badge above arrow
    ax.text(alert_mid_x, arrow_y + 0.36, "RF Alert Telemetry",
            ha="center", va="center", fontsize=9.6, fontweight="bold", color=BLACK,
            bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.06", facecolor=WHITE, edgecolor=BLACK, lw=1.0),
            zorder=6)
    # Secondary condition label below arrow
    ax.text(alert_mid_x, arrow_y - 0.28, r"Threat Confirmed ($\tau \geq 0.50$)",
            ha="center", va="center", fontsize=8.6, fontstyle="italic", color=MID_GRAY, zorder=6)

    save(fig, "fig1_edge_pipeline.png", do_tight=False)


# ==============================================================================
# FIGURE 2: 3 NEURAL ARCHITECTURES (LARGE, HIGH-LEGIBILITY TYPOGRAPHY)
# ==============================================================================
def fig_architectures():
    fig, ax = plt.subplots(figsize=(10.5, 4.4), dpi=300)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    # Column Category Headers
    col_x = [0.20, 2.75, 5.40, 8.05]
    col_w = [2.25, 2.40, 2.40, 2.25]

    headers = ["Architecture", "Input Representation", "Feature Backbone", "Decision Head"]
    for i, h in enumerate(headers):
        ax.text(col_x[i] + col_w[i]/2, 4.05, h, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color=BLACK)

    models = [
        ("1D CNN", "Raw Waveform", False,
         ("Raw Audio Waveform", "16,537 Float Samples"),
         ("Conv1D Backbone", "Multi-Scale (K=80...3)"),
         ("Dense Classifier", "Global Temporal Pooling")),

        ("2D CNN", "[PROPOSED]", True,
         ("Normalized Mel Map", "64 Mel Bins (SpecAugment)"),
         ("Conv2D Backbone", "4 Stages (16→128 Filters)"),
         ("Dual Output Heads", "Gunshot + Anomaly Veto")),

        ("CRNN-PCEN", "Gain-Invariant", False,
         ("Dynamic PCEN Map", "AGC Norm (α=0.98, r=0.04)"),
         ("CRNN Hybrid", "Conv2D + Bi-GRU (128)"),
         ("Sequence Classifier", "Temporal Collapse → Dense"))
    ]

    row_y = [2.70, 1.45, 0.20]
    rh = 0.95

    for i, (m_name, m_tag, is_prop, s1, s2, s3) in enumerate(models):
        y = row_y[i]
        lw = 2.0 if is_prop else 1.2

        # Col 0: Model Card
        b0 = FancyBboxPatch((col_x[0], y), col_w[0], rh,
                            boxstyle="round,pad=0.02,rounding_size=0.04",
                            facecolor=WHITE, edgecolor=BLACK, linewidth=lw, zorder=3)
        ax.add_patch(b0)
        ax.text(col_x[0] + col_w[0]/2, y + rh - 0.30, m_name, ha="center", va="center",
                fontsize=12.0, fontweight="bold", color=BLACK, zorder=4)
        ax.text(col_x[0] + col_w[0]/2, y + 0.30, m_tag, ha="center", va="center",
                fontsize=10.0, fontweight="bold" if is_prop else "normal",
                color=BLACK if is_prop else DARK_GRAY, zorder=4)

        # Arrow 0 -> 1
        ax.annotate("", xy=(col_x[1], y + rh/2), xytext=(col_x[0] + col_w[0], y + rh/2),
                    arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=1.2, mutation_scale=9), zorder=4)

        # Col 1: Input Map
        b1 = FancyBboxPatch((col_x[1], y), col_w[1], rh,
                            boxstyle="round,pad=0.02,rounding_size=0.04",
                            facecolor=WHITE, edgecolor=BLACK, linewidth=lw, zorder=3)
        ax.add_patch(b1)
        ax.text(col_x[1] + col_w[1]/2, y + rh - 0.30, s1[0], ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=BLACK, zorder=4)
        ax.text(col_x[1] + col_w[1]/2, y + 0.30, s1[1], ha="center", va="center",
                fontsize=9.0, color=DARK_GRAY, zorder=4)

        # Arrow 1 -> 2
        ax.annotate("", xy=(col_x[2], y + rh/2), xytext=(col_x[1] + col_w[1], y + rh/2),
                    arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=1.2, mutation_scale=9), zorder=4)

        # Col 2: Feature Backbone
        b2 = FancyBboxPatch((col_x[2], y), col_w[2], rh,
                            boxstyle="round,pad=0.02,rounding_size=0.04",
                            facecolor=WHITE, edgecolor=BLACK, linewidth=lw, zorder=3)
        ax.add_patch(b2)
        ax.text(col_x[2] + col_w[2]/2, y + rh - 0.30, s2[0], ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=BLACK, zorder=4)
        ax.text(col_x[2] + col_w[2]/2, y + 0.30, s2[1], ha="center", va="center",
                fontsize=9.0, color=DARK_GRAY, zorder=4)

        # Arrow 2 -> 3
        ax.annotate("", xy=(col_x[3], y + rh/2), xytext=(col_x[2] + col_w[2], y + rh/2),
                    arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=1.2, mutation_scale=9), zorder=4)

        # Col 3: Decision Head
        b3 = FancyBboxPatch((col_x[3], y), col_w[3], rh,
                            boxstyle="round,pad=0.02,rounding_size=0.04",
                            facecolor=WHITE, edgecolor=BLACK, linewidth=lw, zorder=3)
        ax.add_patch(b3)
        ax.text(col_x[3] + col_w[3]/2, y + rh - 0.30, s3[0], ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=BLACK, zorder=4)
        ax.text(col_x[3] + col_w[3]/2, y + 0.30, s3[1], ha="center", va="center",
                fontsize=9.0, color=DARK_GRAY, zorder=4)

    save(fig, "fig_architectures.png")


# ==============================================================================
# FIGURE 3: 300-CLIP BENCHMARK (CLEAN PROFESSIONAL SIDE-BY-SIDE)
# ==============================================================================
def fig_recall_specificity():
    models = ["1D CNN", "2D CNN", "CRNN"]
    sliding_recall = [45, 67, 57]
    peak_recall    = [30, 38, 33]

    sliding_spec   = [85, 82, 86]
    peak_spec      = [95, 99, 97]

    x = np.arange(len(models))
    w = 0.35

    # Side-by-side clean academic subplots (fits standard column width)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.95), dpi=300)

    # Subplot (a): Gunshot Recall
    ax1.bar(x - w / 2, sliding_recall, w, label="Sliding Window",
            color=LIGHT_GRAY, hatch="//", edgecolor=BLACK, lw=1.0)
    ax1.bar(x + w / 2, peak_recall,    w, label="Peak-Centered",
            color=WHITE, hatch="xx", edgecolor=BLACK, lw=1.0)

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=7.5, fontweight="bold")
    ax1.tick_params(axis="y", labelsize=7.5)
    ax1.set_ylim(0, 80)
    ax1.set_ylabel("Recall (%)", fontsize=8.0, fontweight="bold")
    ax1.set_title("(a) Gunshot Recall", fontsize=8.5, fontweight="bold", pad=4)
    ax1.grid(axis="y", linestyle=":", color=MID_GRAY, alpha=0.5, linewidth=0.6)

    # Subplot (b): Ambient Specificity
    ax2.bar(x - w / 2, sliding_spec, w, label="Sliding Window",
            color=LIGHT_GRAY, hatch="//", edgecolor=BLACK, lw=1.0)
    ax2.bar(x + w / 2, peak_spec,    w, label="Peak-Centered",
            color=WHITE, hatch="xx", edgecolor=BLACK, lw=1.0)

    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=7.5, fontweight="bold")
    ax2.tick_params(axis="y", labelsize=7.5)
    ax2.set_ylim(60, 105)
    ax2.set_ylabel("Specificity (%)", fontsize=8.0, fontweight="bold")
    ax2.set_title("(b) Ambient Specificity", fontsize=8.5, fontweight="bold", pad=4)
    ax2.grid(axis="y", linestyle=":", color=MID_GRAY, alpha=0.5, linewidth=0.6)

    # Clean shared legend at top without marker symbols
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06),
               ncol=2, fontsize=7.5, frameon=True, facecolor=WHITE, edgecolor=BLACK)

    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.14, right=0.98, wspace=0.48)
    save(fig, "fig2_300clip_recall.png", do_tight=False)


# ==============================================================================
# FIGURE 4: LATENCY VS HOP BUDGET (SINGLE COLUMN)
# ==============================================================================
def fig_latency():
    fig, ax = plt.subplots(figsize=(3.5, 2.4), dpi=300)

    models = [
        "2D CNN",
        "1D CNN",
        "CRNN",
    ]
    latencies = [0.09, 2.29, 11.86]
    hop_util  = [0.05, 1.22, 6.33]
    hatches = ["//", "xx", ".."]
    fills = [WHITE, LIGHT_GRAY, DARK_GRAY]

    y_pos = np.arange(len(models))
    bars = ax.barh(y_pos, latencies, height=0.48, color=fills, edgecolor=BLACK, lw=1.2)

    for i, (b, h) in enumerate(zip(bars, hatches)):
        b.set_hatch(h)

    for i, (b, lat, util) in enumerate(zip(bars, latencies, hop_util)):
        x_val = b.get_width()
        text_str = f" {lat:.2f} ms ({util:.1f}%)" if lat >= 0.5 else f" {lat:.2f} ms (0.05%)"
        ax.text(x_val + 0.35, y_pos[i], text_str, va="center", ha="left",
                fontsize=8.2, fontweight="bold", color=BLACK)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=8.8, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 22.0)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_xlabel("Mean Latency (ms)", fontsize=9.0, fontweight="bold")
    ax.set_title("Edge Inference Latency", fontsize=10.0, fontweight="bold", pad=6)
    ax.grid(axis="x", linestyle=":", color=MID_GRAY, alpha=0.5, linewidth=0.7)

    # Clean callout badge
    ax.text(14.8, 0.45,
            "Hop: 187.5 ms\nMax Load: 6.3%",
            ha="center", va="center", fontsize=7.5, fontweight="bold", color=BLACK,
            bbox=dict(boxstyle="round,pad=0.20", facecolor=WHITE, edgecolor=BLACK, lw=0.9))

    save(fig, "fig3_latency.png")


# ==============================================================================
# FIGURE 5: DOMAIN SHIFT GENERALIZATION (SINGLE COLUMN, ZERO COLLISION)
# ==============================================================================
def fig_lab_vs_external():
    fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=300)

    models = ["1D CNN", "CRNN", "2D CNN"]
    holdout = [99.7, 99.7, 99.6]
    field   = [30.0, 60.0, 85.0]

    x = np.arange(len(models))
    w = 0.32

    b1 = ax.bar(x - w / 2, holdout, w, label="Hold-Out",
                color=LIGHT_GRAY, hatch="//", edgecolor=BLACK, lw=1.2)
    b2 = ax.bar(x + w / 2, field,   w, label="Field Audio",
                color=WHITE, hatch="xx", edgecolor=BLACK, lw=1.2)

    # Holdout percentage text directly above bar
    for b in b1:
        bx = b.get_x() + b.get_width() / 2
        ax.text(bx, 101.5, "100%", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color=BLACK)

    # Field audio percentage text directly above bar
    for b, val in zip(b2, field):
        bx = b.get_x() + b.get_width() / 2
        ax.text(bx, val + 1.5, f"{val:.0f}%", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color=BLACK)

    # Clean Delta badges centered above each model pair with ample clearance
    deltas = ["Δ = -70%", "Δ = -40%", "Δ = -15%"]
    for i, d in enumerate(deltas):
        ax.text(x[i], 117, d, ha="center", va="center", fontsize=7.5, fontweight="bold", color=BLACK,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=WHITE, edgecolor=BLACK, lw=0.85))

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8.8, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8.0)
    ax.set_ylim(0, 132)
    ax.set_ylabel("Recall (%)", fontsize=9.0, fontweight="bold")
    ax.set_title("Domain Shift Generalization", fontsize=9.8, fontweight="bold", pad=18)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=8.0, frameon=False)
    ax.grid(axis="y", linestyle=":", color=MID_GRAY, alpha=0.5, linewidth=0.7)

    save(fig, "fig4_lab_vs_external.png")


# ==============================================================================
# FIGURE 6: ENSEMBLE CONFUSION MATRIX (SINGLE COLUMN)
# ==============================================================================
def fig_confusion():
    fig, ax = plt.subplots(figsize=(3.5, 2.7), dpi=300)

    matrix = np.array([
        [77, 15,  8],
        [44, 46, 10],
        [20, 38, 42]
    ])

    rows = ["True Shot", "True Imp.", "True Amb."]
    cols = ["Gunshot", "Imposter", "Ambient"]

    for i in range(3):
        for j in range(3):
            val = matrix[i, j]
            is_diag = (i == j)
            face = "#e2e8f0" if is_diag else WHITE

            rect = Rectangle((j, 2 - i), 0.96, 0.96, facecolor=face,
                             edgecolor=BLACK, lw=1.5 if is_diag else 0.9, zorder=2)
            ax.add_patch(rect)

            ax.text(j + 0.48, 2 - i + 0.58, f"{val}", ha="center", va="center",
                    fontsize=14.0, fontweight="bold", color=BLACK, zorder=3)
            ax.text(j + 0.48, 2 - i + 0.28, f"({val}%)", ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color=DARK_GRAY, zorder=3)

            if is_diag:
                ax.plot(j + 0.82, 2 - i + 0.82, marker="*", markersize=6.5, color=BLACK, zorder=4)

    ax.set_xlim(-0.02, 2.98)
    ax.set_ylim(-0.02, 2.98)
    ax.set_xticks([0.48, 1.48, 2.48])
    ax.set_xticklabels(cols, fontsize=8.5, fontweight="bold")
    ax.set_yticks([0.48, 1.48, 2.48])
    ax.set_yticklabels(list(reversed(rows)), fontsize=8.5, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=9.0, fontweight="bold", labelpad=4)
    ax.set_ylabel("Ground Truth", fontsize=9.0, fontweight="bold", labelpad=4)
    ax.set_title("Ensemble Confusion Matrix", fontsize=10.0, fontweight="bold", pad=6)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    save(fig, "fig5_ensemble_confusion.png")


# ==============================================================================
# FIGURE 7: PCEN DOMAIN DISTORTION INVARIANCE (SINGLE COLUMN)
# ==============================================================================
def fig_pcen():
    fig, ax = plt.subplots(figsize=(3.5, 2.4), dpi=300)

    taus = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    clean_rec     = [69, 64, 62, 61, 61, 60, 59]
    distorted_rec = [69, 67, 66, 63, 61, 61, 59]

    ax.plot(taus, clean_rec, marker="o", markersize=6.0, markerfacecolor=WHITE, markeredgecolor=BLACK,
            markeredgewidth=1.6, lw=1.8, color=BLACK, linestyle="-",
            label="Clean (●)", zorder=3)

    ax.plot(taus, distorted_rec, marker="s", markersize=6.0, markerfacecolor=BLACK, markeredgecolor=BLACK,
            markeredgewidth=1.6, lw=1.8, color=BLACK, linestyle="--",
            label="Distorted (■)", zorder=3)

    ax.plot([0.50, 0.70], [61, 59], marker="D", markersize=7.5, markerfacecolor=WHITE,
            markeredgecolor=BLACK, markeredgewidth=1.8, linestyle="none", zorder=4)

    ax.annotate("Zero Drop at τ ≥ 0.50",
                xy=(0.50, 61), xytext=(0.10, 53.0),
                arrowprops=dict(arrowstyle="->", color=BLACK, lw=1.2),
                fontsize=7.8, fontweight="bold", color=BLACK,
                bbox=dict(boxstyle="round,pad=0.18", facecolor=WHITE, edgecolor=BLACK, lw=0.9),
                zorder=5)

    ax.set_xlim(0.05, 0.75)
    ax.set_ylim(49, 76)
    ax.set_xticks(taus)
    ax.tick_params(axis="both", labelsize=8.0)
    ax.set_xlabel("Threshold (τ)", fontsize=9.0, fontweight="bold")
    ax.set_ylabel("Recall (%)", fontsize=9.0, fontweight="bold")
    ax.set_title("PCEN Gain Invariance", fontsize=10.0, fontweight="bold", pad=6)
    ax.legend(loc="upper right", fontsize=7.8, frameon=True, facecolor=WHITE, edgecolor=BLACK)
    ax.grid(True, linestyle=":", color=MID_GRAY, alpha=0.5, linewidth=0.7)

    save(fig, "fig6_pcen_distortion.png")


if __name__ == "__main__":
    fig_edge_pipeline()
    fig_architectures()
    fig_recall_specificity()
    fig_latency()
    fig_lab_vs_external()
    fig_confusion()
    fig_pcen()
    print("All bent and high-legibility publication figures generated successfully!")
