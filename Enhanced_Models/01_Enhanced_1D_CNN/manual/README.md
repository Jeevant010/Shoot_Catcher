# Enhanced 1D CNN — Manual

## Architecture
```
Input (TARGET_SAMPLES, 1)
  → Conv1D(32, kernel=80, stride=4) + BN + MaxPool(4)
  → Conv1D(64, 3) + BN + MaxPool(4)
  → Conv1D(128, 3) + BN
  → Conv1D(128, 3) + BN
  → GlobalAvgPool
  ├→ Dense(64) + Dropout(0.4) → sigmoid → Gunshot Output
  └→ Dense(32) + Dropout(0.3) → sigmoid → Anomaly Output
```

## Augmentation Pipeline

| Aug | Probability | Parameters | Paper |
|-----|-------------|------------|-------|
| Time Shift | 50% | ±10% of clip length | — |
| Noise Injection | 50% | SNR 15-30 dB | — |
| Gain Variation | 50% | ±6 dB | — |
| SpecAugment (time) | 40% | Mask up to 15% of duration | Park et al. 2019 |
| Pitch Shift | 30% | ±2 semitones | Salamon & Bello 2017 |
| Speed Perturb | 30% | ±10% | Ko et al. 2015 |
| MixUp | 100% (post-load) | α=0.3 | Zhang et al. 2018 |

## How to Use
1. Set `DATA_DIR` to your trimmed data folder (must contain `class_0_nongunshot/` and `class_1_gunshot/`)
2. Set `CLIP_DURATION_MS` to match your trimmer's `TARGET_MS`
3. Run all cells
4. Check `output/` for the trained model and evaluation results

## Output Files
- `enhanced_1d_cnn_best.h5` — Best model checkpoint
- `enhanced_1d_cnn_confusion.png` — Confusion matrix plot
- `enhanced_1d_cnn_results.json` — F2 score, ROC-AUC, config
