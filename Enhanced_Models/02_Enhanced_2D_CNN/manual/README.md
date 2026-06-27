# Enhanced 2D CNN — Manual

## Architecture
```
Input (N_MELS, time_frames, 1)
  → Conv2D(16, 3x3) + BN + MaxPool(2x2)
  → Conv2D(32, 3x3) + BN + MaxPool(2x2)
  → Conv2D(64, 3x3) + BN + MaxPool(2x2)
  → Conv2D(128, 3x3) + BN
  → GlobalAvgPool2D
  ├→ Dense(64) + Dropout(0.4) → sigmoid → Gunshot Output
  └→ Dense(32) + Dropout(0.3) → sigmoid → Anomaly Output
```

## Why 2D CNN?
The 1D CNN looks at the raw waveform shape. The 2D CNN converts audio to a **Mel-Spectrogram** first, treating it like an image. This makes it:
- **More robust** to speaker/microphone differences (domain shift)
- **Better at distinguishing** gunshots from similar transients (door slams, claps)
- **Standard approach** used by gabemagee/gunshot_detection, Edge Impulse, and most research papers

## Mel-Spectrogram Parameters
| Param | Default | Meaning |
|-------|---------|---------|
| N_MELS | 64 | Number of Mel frequency bands |
| N_FFT | 2048 | FFT window size |
| HOP_LENGTH | 512 | Samples between FFT windows |
| FMIN | 20 Hz | Minimum frequency |
| FMAX | 8000 Hz | Maximum frequency |

## Augmentation Pipeline (same as 1D, plus spectrogram-level)

**Raw waveform level** (before spectrogram):
- Time shift, noise injection, gain, pitch shift, speed perturbation

**Spectrogram level** (after conversion):
- Frequency masking (SpecAugment)
- Time masking (SpecAugment)
- MixUp (applied to final spectrograms)

## How to Use
1. Set `DATA_DIR` to your trimmed data folder
2. Set `CLIP_DURATION_MS` to match your trimmer
3. Run all cells
4. Check `output/` for results

## Output Files
- `enhanced_2d_cnn_best.h5` — Best model
- `enhanced_2d_cnn_confusion.png` — Confusion matrix
- `enhanced_2d_cnn_results.json` — Metrics and config
