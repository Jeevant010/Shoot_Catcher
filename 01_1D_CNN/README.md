# 🔫 01 — 1D CNN Gunshot Detector

A **raw waveform classifier** that detects gunshots using 1D convolutions directly on audio samples. No feature engineering required.

## What It Does

- Loads 250ms audio clips from your trimmed data folder
- Splits data using **GroupKFold** (prevents data leakage from same source recording)
- Trains a 4-layer 1D CNN with multi-scale kernels
- Evaluates using **PR-AUC, F2-score, confusion matrix** (not just accuracy)
- Exports to `.h5` and `.tflite` (INT8 quantized)

## Quick Start

1. Open `1d_cnn_gunshot_detector.ipynb` in Jupyter or VS Code
2. Set `DATA_DIR` in Cell 3 to your trimmed data folder
3. Choose `SAMPLE_RATE`: `22050` (quality) or `16000` (Arduino)
4. Run all cells

## Architecture

```
Conv1D(32, k=80, s=4) → BN → MaxPool(4)    # Captures 3.6ms transients
Conv1D(64, k=3)       → BN → MaxPool(4)    # Refines patterns
Conv1D(128, k=3)      → BN                 # Higher-level features
Conv1D(128, k=3)      → BN                 # Further refinement
GlobalAvgPool → Dense(64) → Dropout → Dense(1, sigmoid)
```

## Output

```
01_1D_CNN/output/
├── 1d_cnn_best.h5                  ← Best model checkpoint
├── 1d_cnn_gunshot_detector.h5      ← Final Keras model
├── 1d_cnn_float32.tflite           ← TFLite (full precision)
├── 1d_cnn_int8.tflite              ← TFLite (INT8 quantized)
├── 1d_cnn_results.json             ← Metrics (PR-AUC, F2, etc.)
├── 1d_cnn_training_history.png     ← Training curves
├── 1d_cnn_evaluation.png           ← Confusion matrix + curves
└── data_distribution.png           ← Dataset visualization
```

## Full Documentation

See [manual/README.md](manual/README.md) for complete theory including:
- What is a 1D CNN and why it works
- Every layer explained with your exact dimensions
- Class weights, GroupKFold, evaluation metrics
- Troubleshooting guide
