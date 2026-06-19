# 🔫 02 — 2D CNN Gunshot Detector

The **production model** — converts audio to Mel Spectrograms (or MFCCs) and uses 2D convolutions to classify gunshots with maximum accuracy.

## What It Does

- Extracts **Mel Spectrograms** or **MFCCs** from 250ms audio clips
- Trains a 4-layer 2D CNN (32→64→128→128 filters)
- Includes **SpecAugment** (frequency + time masking) for robustness
- Compares results head-to-head with the 1D CNN baseline
- Exports to `.h5` and `.tflite` (INT8 quantized)

## Quick Start

1. Open `2d_cnn_gunshot_detector.ipynb`
2. Set `DATA_DIR` in Cell 3
3. Choose `FEATURE_TYPE`: `mel_spectrogram` (best accuracy) or `mfcc` (smallest model)
4. Choose `SAMPLE_RATE`: `22050` (quality) or `16000` (Arduino)
5. Run all cells

## Architecture

```
Conv2D(32, 3×3) → BN → MaxPool(2×2)   # Local patterns
Conv2D(64, 3×3) → BN → MaxPool(2×2)   # Combined patterns
Conv2D(128, 3×3) → BN                  # High-level features
Conv2D(128, 3×3) → BN                  # Refinement
GlobalAvgPool → Dense(64) → Dropout → Dense(1, sigmoid)
```

## Feature Comparison

| Feature | Input Shape | Model Size (INT8) | Accuracy |
|---------|------------|-------------------|----------|
| Mel Spectrogram (64 bins) | (64, 44, 1) | ~80-120 KB | Highest |
| MFCC (13 coefficients) | (13, 44, 1) | ~30-50 KB | Good |

## Full Documentation

See [manual/README.md](manual/README.md) for complete theory:
- Fourier Transform, Mel Scale, MFCC explained from scratch
- Why spectrograms beat raw waveforms
- SpecAugment and domain shift mitigation
- Model compression path to Arduino
