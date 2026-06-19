# 2D CNN Gunshot Detector — Complete Manual

> **Who is this for?** Anyone who wants to understand how audio becomes an image, why spectrograms work, and how this 2D CNN uses them to detect gunshots.

---

## Table of Contents

1. [From Sound to Image](#1-from-sound-to-image)
2. [The Fourier Transform](#2-the-fourier-transform)
3. [Spectrograms Explained](#3-spectrograms-explained)
4. [The Mel Scale](#4-the-mel-scale)
5. [MFCC vs Mel Spectrogram](#5-mfcc-vs-mel-spectrogram)
6. [2D Convolution — How It Reads Spectrograms](#6-2d-convolution)
7. [The Architecture](#7-the-architecture)
8. [SpecAugment](#8-specaugment)
9. [Domain Shift — The Microphone Problem](#9-domain-shift)
10. [Model Compression & Deployment](#10-model-compression--deployment)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. From Sound to Image

Audio is a 1D signal (amplitude over time). But a gunshot "looks" very different from a car horn when you visualize its **frequency content over time**.

**The key insight**: Convert audio to a 2D image → use computer vision techniques (CNN) → classify.

```
Raw Audio          →  Spectrogram          →  CNN          →  "GUNSHOT!"
[0.1, 0.3, -0.8]  →  [freq × time image]  →  Conv2D...   →  0.97
```

A gunshot in a spectrogram appears as a **bright vertical stripe** — all frequencies explode simultaneously for a very short time. This is fundamentally different from speech (horizontal bars), traffic (low-frequency blobs), or birds (thin diagonal lines).

---

## 2. The Fourier Transform

The **Fourier Transform** decomposes a signal into its frequency components. Think of it like a prism splitting white light into rainbow colors — except for sound.

### Plain English
Your audio signal is a mix of many frequencies. The Fourier Transform answers: "How much of each frequency is present?"

### The Math (Simplified)
```
Input:  A waveform of N samples
Output: N/2 frequency bins, each with a magnitude (how loud) and phase (timing)
```

For a 250ms clip at 22,050 Hz:
- N = 5,512 samples
- FFT with N_FFT=512: looks at 512 samples at a time
- Produces 257 frequency bins (0 Hz to 11,025 Hz)

---

## 3. Spectrograms Explained

A **spectrogram** is a sequence of Fourier Transforms computed on overlapping windows of audio.

```
Window 1:  audio[0:512]     → FFT → frequency snapshot at t=0
Window 2:  audio[128:640]   → FFT → frequency snapshot at t=1  (hop=128)
Window 3:  audio[256:768]   → FFT → frequency snapshot at t=2
...
```

The result is a 2D matrix: **frequency bins × time steps**.

### Time-Frequency Tradeoff
- **Larger N_FFT** (e.g., 2048): Better frequency resolution, worse time resolution
- **Smaller N_FFT** (e.g., 256): Better time resolution, worse frequency resolution
- **Our choice: N_FFT=512**: Good compromise for 250ms gunshot clips

### Hop Length
- `hop_length = 128` means each window overlaps the previous by `512-128 = 384` samples
- Smaller hop = more time steps = higher temporal resolution = larger matrix

---

## 4. The Mel Scale

Human hearing is **not linear**. We can tell the difference between 100 Hz and 200 Hz easily, but 10,000 Hz and 10,100 Hz sound almost the same.

The **Mel scale** maps frequencies to how humans perceive pitch:
- Low frequencies (0-1000 Hz): Fine resolution (many mel bins)
- High frequencies (1000-11000 Hz): Coarse resolution (fewer mel bins)

### Why Mel for Gunshots?
Gunshots produce energy across ALL frequencies. By using Mel scale:
1. Low-frequency boom (the explosion) gets detailed representation
2. High-frequency crack (supersonic bullet) is still captured, just compressed
3. The model focuses on the perceptually important parts

### Our Settings
- `n_mels = 64`: 64 Mel-frequency bins
- Input: (64 × 44) — 64 frequency rows, ~44 time columns
- Each pixel = "how much energy at this frequency at this time"

---

## 5. MFCC vs Mel Spectrogram

| | Mel Spectrogram | MFCC |
|---|---|---|
| **What it is** | Mel-scaled frequency × time image | Compressed version of Mel Spectrogram |
| **How it works** | Apply mel filterbank to FFT | Apply mel filterbank + DCT (Discrete Cosine Transform) |
| **Dimensions** | (64, 44) | (13, 44) |
| **Information loss** | None (keeps all spectral detail) | Significant (DCT discards fine correlations) |
| **Best for** | Maximum accuracy, research models | Smallest model, Arduino deployment |
| **Drawback** | Larger model | May miss subtle spectral features |

### Which should you use?
- **Start with `mel_spectrogram`** — it gives the CNN more to work with
- Switch to `mfcc` if the model is too large for your target hardware
- The notebook lets you switch with one config change: `FEATURE_TYPE = 'mfcc'`

---

## 6. 2D Convolution

A 2D convolution slides a small filter (e.g., 3×3) across the spectrogram image, looking for patterns.

### What the Filters Learn

After training, different filters detect different things:
- **Vertical edge detector**: Finds the sudden onset of a gunshot (all frequencies spike at once)
- **Horizontal edge detector**: Finds sustained tones (constant frequency over time)
- **Diagonal detector**: Finds rising/falling tones (sirens, birds)
- **Texture detector**: Finds the "grain" pattern of specific noise types

### Why (3,3) Kernels?

Research (2024-2025, including Kaggle competitions and published papers) found:
- (3,3) kernels are optimal for capturing local spectro-temporal patterns
- Larger kernels (5,5 or 7,7) don't improve accuracy but increase computation
- Stacking multiple (3,3) layers achieves the same receptive field as one large kernel, but with more non-linearities (better pattern detection)

---

## 7. The Architecture

```
Input: (64, 44, 1) — Mel Spectrogram with 1 channel
       or (13, 44, 1) — MFCC
        │
        ▼
[Conv2D: 32 filters, 3×3] → BatchNorm → MaxPool(2×2)
        │  Shape: (32, 22, 32)
        ▼
[Conv2D: 64 filters, 3×3] → BatchNorm → MaxPool(2×2)
        │  Shape: (16, 11, 64)
        ▼
[Conv2D: 128 filters, 3×3] → BatchNorm              ← NO pooling here!
        │  Shape: (16, 11, 128)
        ▼
[Conv2D: 128 filters, 3×3] → BatchNorm
        │  Shape: (16, 11, 128)
        ▼
[GlobalAveragePooling2D]                             ← Collapse to vector
        │  Shape: (128,)
        ▼
[Dense: 64, ReLU + Dropout(0.4)]
        │  Shape: (64,)
        ▼
[Dense: 1, Sigmoid]                                  ← 0.0=noise, 1.0=gunshot
```

### Why No Pooling in Blocks 3-4?

For 250ms clips, the spectrogram is already small (64×44). Two MaxPool(2×2) operations reduce it to 16×11. Further pooling would destroy the remaining temporal detail — we need those 11 time steps to distinguish a quick gunshot from sustained noise.

### Why GlobalAveragePooling instead of Flatten?

- **Flatten(16×11×128)** = 22,528 inputs to the Dense layer → huge, prone to overfitting
- **GlobalAveragePooling()** = 128 inputs → small, regularizes naturally
- Research shows GAP consistently outperforms Flatten for audio CNNs

---

## 8. SpecAugment

**SpecAugment** is a data augmentation technique from Google (2019) that works directly on spectrograms.

### Frequency Masking
Randomly zeros out a block of frequency bins:
```
Before:  [████████████████]    (all 64 mel bins visible)
After:   [████░░░░████████]    (bins 5-8 masked to zero)
```
Forces the model to not rely on any single frequency band. This fights **microphone bias** — different microphones have different frequency responses.

### Time Masking
Randomly zeros out a block of time steps:
```
Before:  [████████████]    (all 44 time steps visible)  
After:   [████░░░░████]    (steps 5-8 masked to zero)
```
Forces the model to detect gunshots even if part of the signal is missing — simulates the "Guillotine Effect" from the ring buffer discussion.

### Our Settings
- `FREQ_MASK_PARAM = 8`: Masks up to 8 consecutive frequency bins
- `TIME_MASK_PARAM = 8`: Masks up to 8 consecutive time steps

---

## 9. Domain Shift — The Microphone Problem

This is the "Bloody Flaw" from Chat.md. The model might learn the **microphone's frequency fingerprint** instead of the gunshot's actual acoustic signature.

### The Problem
- Training data recorded on Microphone A (e.g., YouTube audio ripped at 44.1kHz)
- Test data from Microphone B (e.g., laptop built-in mic at 48kHz)
- The spectrograms look different even for the same sound!

### How We Mitigate This

1. **SpecAugment**: Masks random frequency bands → model can't rely on mic-specific peaks
2. **Noise injection**: Adds random noise to raw audio before feature extraction
3. **Normalization**: We normalize spectrograms using training set mean/std
4. **Data augmentation**: Gain variation simulates different mic sensitivities

### The Reality Check
No augmentation can fully eliminate domain shift. When you test with a real microphone (in the `03_Mic_Test` notebook), expect lower accuracy than your test set shows. The gap is your **domain shift penalty**.

---

## 10. Model Compression & Deployment

### Path from Notebook to Arduino

```
Keras Model (.h5)          → ~500 KB, runs on laptop
    ↓ TFLite float32       → ~200 KB, runs on Raspberry Pi
    ↓ TFLite INT8           → ~50 KB, runs on Arduino Nano 33 BLE
    ↓ C array header        → Embedded in Arduino sketch
```

### INT8 Quantization
Converts all weights from 32-bit floating point to 8-bit integers:
- 4x smaller model
- 2-4x faster inference
- ~1-2% accuracy loss (usually acceptable)

The notebook does this automatically using a **representative dataset** for calibration.

### Model Size Budget for Arduino Nano 33 BLE
- Flash: 1 MB → Model must be < ~200 KB
- RAM: 256 KB → Activations must fit in < ~100 KB
- Our 2D CNN with MFCC (13×44): typically ~30-50 KB INT8 ✅
- Our 2D CNN with Mel Spec (64×44): typically ~80-120 KB INT8 ⚠️ (tight)

---

## 11. Troubleshooting

### Feature extraction is very slow
- This is normal for large datasets. The bottleneck is `librosa.load()` reading WAV files.
- Consider running once and caching the extracted features as `.npy` files.

### Model only predicts one class
- Check class weights in the training output
- Try reducing learning rate to 0.0001
- Ensure your data has both classes

### 2D CNN performs WORSE than 1D CNN
- Feature extraction parameters might not match your data
- Try: `N_MELS=128` (more frequency detail) or `HOP_LENGTH=64` (more time detail)
- Check the SpecAugment parameters — too aggressive masking can hurt

### "OOM" (Out of Memory) during training
- Reduce `BATCH_SIZE` to 32 or 16
- Reduce `N_MELS` to 32
- Use `FEATURE_TYPE = 'mfcc'` (13 rows instead of 64)

---

> **Next step**: After evaluating both 1D and 2D CNN, use the `03_Mic_Test` notebook to test your best model with real microphone input.
