# 🎤 03 — Microphone Test & Live Inference

Tests your laptop microphone and runs trained gunshot detection models on live audio.

## Quick Start

1. Train at least one model first (01_1D_CNN or 02_2D_CNN)
2. Open `mic_test_and_inference.ipynb`
3. Update `MODEL_PATH` in Cell 2
4. Run cells in order

## What Each Cell Does

| Cell | Purpose |
|------|---------|
| 3 | **Diagnostics** — lists devices, tests access, checks sample rates |
| 4 | **Record & Play** — records 3s, shows waveform, plays back |
| 5 | **Noise Floor** — measures background noise level |
| 6 | **Load Model** — loads your trained CNN |
| 7 | **Single Prediction** — record 250ms, run inference |
| 8 | **Continuous Monitor** — ring buffer with 50% overlap |

## Common Fix

If microphone doesn't work:
1. Open **Settings → Privacy → Microphone**
2. Turn ON "Let desktop apps access your microphone"
3. Restart Jupyter/VS Code

See [manual/README.md](manual/README.md) for full troubleshooting.
