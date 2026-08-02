# 🛡️ Module 04: Robust CRNN-PCEN Gunshot Finder — Technical Manual

Welcome to **Module 04**, a dedicated, fully isolated module built specifically to solve **Acoustic Domain Shift (Microphone Mismatch)** in gunshot detection.

---

## 1. Why Standard Models Fail on Real Microphones

Standard machine learning audio classifiers rely on **Log-Mel Spectrograms**. While Log-Mel works well in clean benchmark datasets, it crumbles on real-world hardware for three reasons:
1. **Microphone Frequency Bias**: Cheap laptop, phone, or embedded mic capsules roll off low/high frequencies and introduce non-linear distortion.
2. **Automatic Gain Control (AGC)**: Mics dynamically compress or boost volume based on background noise.
3. **Background Noise Floor**: Static HVAC hum, wind, and mic self-noise alter the magnitude baseline across frequency bins.

When an internet-trained model receives real microphone audio, it sees a completely altered spectrogram "image" and fails to trigger.

---

## 2. Theoretical Foundations: PCEN + CRNN

Module 04 adopts research from **Forest Acoustic & Chainsaw Detection Literature (DCASE / NYU Bioacoustics)**:

### A. Per-Channel Energy Normalization (PCEN)
PCEN replaces static Log-Mel compression with an adaptive gain control envelope:
$$P[f, t] = \left( \frac{S[f, t]}{(\epsilon + M[f, t])^{\alpha}} + \delta \right)^{r} - \delta^r$$

* **How it solves domain shift**: The temporal IIR filter $M[f, t]$ tracks stationary background noise for each frequency channel. Static background hum and microphone frequency bias are continuously subtracted. **Only sharp, explosive onset transients (gunshots) stand out**, regardless of which microphone recorded the sound!

### B. Synthetic Microphone Augmentation
During training, clean dataset WAV files are convolved on-the-fly with synthetic microphone transfer functions:
- Bandpass filtering (200Hz - 8kHz cutoffs)
- Non-linear peak clipping (simulating mic overload)
- Variable noise injection (HVAC, wind, room hum)
- Random gain scaling (0.3x to 1.7x)

### C. Open-Source CRNN Architecture
- **Conv2D Layers**: Extract spatial PCEN time-frequency features.
- **Bidirectional GRU**: Captures the temporal sequence of muzzle blast onset followed by reverberation decay.
- **Global Average Pooling & Dense Head**: Generates final gunshot probability.

---

## 3. Module 04 File Overview

```text
04_Robust_CRNN_PCEN/
├── pcen_mic_pipeline.py         ← PCEN & Synthetic Mic Distortion Engine
├── train_crnn_pcen.ipynb        ← Phase 1: Notebook Training & 1:1 Balanced Evaluation
├── run_live_crnn.py             ← Phase 2: Isolated Live Mic Monitor with VU-Meter
├── test_file_crnn.py            ← Offline Audio File Benchmark Runner
├── quantize_crnn.py             ← Phase 3: TFLite INT8 Quantizer & C++ Export
└── manual/
    ├── README.md                ← System Manual (This file)
    ├── RASPBERRY_PI_GUIDE.md    ← Phase 4: Raspberry Pi Deployment Guide
    └── ARDUINO_NANO_BLE_GUIDE.md← Phase 5: Arduino Nano 33 BLE Guide & C++ Code
```

---

## 4. How to Use Module 04 Step-by-Step

### Phase 1: Train the Model
Open and run `train_crnn_pcen.ipynb` in Jupyter Notebook. This trains the CRNN on PCEN features with synthetic mic distortion and evaluates on a **strict 1:1 balanced test split**.

### Phase 2: Run Live Microphone Test
Run the dedicated isolated live script:
```bash
python 04_Robust_CRNN_PCEN/run_live_crnn.py
```
- **Visual VU-Meter**: Confirms your microphone is actively receiving sound (`[██████░░░░] -24.2 dBFS`).
- **Clean Mic Selection**: Shows physical input devices, filtering out fake driver aliases.

### Phase 3: Quantize for Edge Devices
Run the quantizer:
```bash
python 04_Robust_CRNN_PCEN/quantize_crnn.py
```
Generates `output/crnn_pcen_int8.tflite` and `output/crnn_model_data.h` for microcontrollers.
