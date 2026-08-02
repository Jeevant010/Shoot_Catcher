# 🛡️ Master Plan: End-to-End Robust CRNN-PCEN Gunshot Finder

## System Strategy & Vision
This plan addresses the **exact real-world bottleneck where gunshot detection projects fail**: passing the **Step 2 Live Script Test** on laptop/microphone hardware after high accuracy in Jupyter Notebooks.

### Why Previous Live Scripts Failed (The Step 2 Bottleneck)
1. **Acoustic Domain Shift**: Clean dataset WAV files do not match real microphone hardware (which applies bandpass filtering, automatic gain control, room reverberation, and mic self-noise).
2. **Microphone Diagnostic Blindness**: The user cannot tell if the laptop mic is actively listening, muted, or selected to a fake virtual audio driver (e.g. 12 duplicate Intel/Realtek aliases).
3. **Overly Complex Scripting**: Multi-model options crammed into a single script create confusion, buffer lag, and missed detections.
4. **Imbalanced Test Bias**: Test sets inflated with majority non-gunshots give fake high accuracy in notebooks that crumbles on real audio.

---

## 🎯 Solution Architecture: Fully Isolated Module (`04_Robust_CRNN_PCEN`)

All new work will be **100% isolated** in a brand new folder `Shoot_Catcher/04_Robust_CRNN_PCEN/` without touching or altering any existing code or models.

```text
Shoot_Catcher/
├── 04_Robust_CRNN_PCEN/                    ← COMPLETELY ISOLATED MODULE
│   ├── pcen_mic_pipeline.py               ← PCEN & Synthetic Mic Distortion Pipeline
│   ├── train_crnn_pcen.ipynb              ← Notebook Phase (Step 1)
│   ├── run_live_crnn.py                   ← Dedicated Isolated Live Runner (Step 2)
│   ├── test_file_crnn.py                  ← Dedicated Isolated File Benchmark
│   ├── quantize_crnn.py                   ← TFLite INT8 Quantization (Step 3)
│   ├── manual/
│   │   ├── README.md                      ← Theoretical & Practical Guide
│   │   ├── RASPBERRY_PI_GUIDE.md          ← Raspberry Pi Deployment (Step 4)
│   │   └── ARDUINO_NANO_BLE_GUIDE.md      ← Arduino Nano 33 BLE Deployment (Step 5)
│   └── output/
│       ├── crnn_pcen_model.h5
│       ├── crnn_pcen_int8.tflite
│       └── pcen_stats.json
```

---

## 🔬 Key Innovations in Module 04

### 1. PCEN (Per-Channel Energy Normalization) Preprocessing
Replaces static Log-Mel Spectrograms with dynamic PCEN:
$$P[f, t] = \left( \frac{S[f, t]}{(\epsilon + M[f, t])^{\alpha}} + \delta \right)^{r} - \delta^r$$
- **What it does**: Automatically suppresses background noise, room hum, and microphone gain variations. **Static noise disappears; only sharp impulsive onset transients (gunshots) stand out**, regardless of which microphone recorded it!

### 2. Synthetic Microphone & Room Impulse Response (RIR) Augmentation
During notebook training, we pass clean dataset clips through a synthetic microphone transformation:
- **Bandpass Filtering**: Simulates low-cost mic capsules (200Hz - 8kHz cutoffs).
- **Dynamic Range Compression & Clipping**: Simulates mic capsule overload.
- **Room/Outdoor Echo Simulation**: Convolves audio with synthetic impulse responses.
- **Random Gain Scaling & Noise Injection**: Simulates mic sensitivity variations.

### 3. Open-Source CRNN Architecture (Conv2D + GRU + Temporal Attention)
- **Conv2D Blocks**: Extract spatial time-frequency features from PCEN.
- **Bidirectional GRU**: Captures the temporal sequence (explosive muzzle blast → reverberation decay).
- **Temporal Attention**: Weights the blast onset higher than background tail noise.
- *Open-source architecture from DCASE / Bioacoustics literature—100% free of copyright restrictions.*

### 4. 1:1 Strict Balanced Test & Validation Split
- Validation and Test sets are forced to a **strict 1:1 ratio (50% gunshot, 50% non-gunshot)**.
- Prevents the model from "cheating" or inflating metrics via majority class bias.
- Uses strict **GroupKFold** by source recording to prevent data leakage.

### 5. Isolated Live Runner with Real-Time VU-Meter (`run_live_crnn.py`)
- **Microphone Signal VU-Meter**: Displays a live ASCII volume bar (`[██████░░░░] -24 dBFS`) so you can **visually confirm** your microphone is receiving live audio before testing.
- **Clean Hardware Selection**: Filters out duplicate virtual audio drivers, showing only physical input devices.
- **Manual Start/Stop Recording (`[R]` key)**: Allows you to record ambient or phone playback audio at any point to save `.wav` clips for inspection.
- **Zero Emoji Clutter**: Professional, industrial log outputs (`YYYY-MM-DD HH:MM:SS [INFO]`).

---

## 🚦 5-Phase End-to-End Progression Roadmap

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: Jupyter Notebook Training & 1:1 Balanced Evaluation            │
│ └── Notebook: train_crnn_pcen.ipynb                                     │
│ └── Target: >90% Precision/Recall on 1:1 Balanced GroupKFold Test Set   │
└────────────────────┬────────────────────────────────────────────────────┘
                     │ (Pass Gate: Metrics Verified)
                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: Isolated Laptop Microphone Live Script                         │
│ └── Script: run_live_crnn.py (with Live VU-Meter & Manual Rec)           │
│ └── Target: Catch live phone/mic gunshot sounds with 0 false alarms     │
└────────────────────┬────────────────────────────────────────────────────┘
                     │ (Pass Gate: Real-World Mic Verified)
                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: TFLite INT8 Quantization & Memory Budget Inspection            │
│ └── Script: quantize_crnn.py                                            │
│ └── Target: Model size < 150KB, RAM footprint < 80KB                    │
└────────────────────┬────────────────────────────────────────────────────┘
                     │ (Pass Gate: Edge Memory Verified)
                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: Raspberry Pi Field Deployment                                  │
│ └── Guide & Script: manual/RASPBERRY_PI_GUIDE.md                        │
│ └── Target: Real-time inference on Raspberry Pi with USB / I2S Mic      │
└────────────────────┬────────────────────────────────────────────────────┘
                     │ (Pass Gate: Embedded Linux Verified)
                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 5: Arduino Nano 33 BLE Sense Microcontroller Deployment           │
│ └── Guide & C++ Code: manual/ARDUINO_NANO_BLE_GUIDE.md                 │
│ └── Target: TFLite Micro C++ code running on Cortex-M4 PDM Mic          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Proposed Changes

### Component 1: PCEN & Mic Pipeline
#### [NEW] `04_Robust_CRNN_PCEN/pcen_mic_pipeline.py`
- Preprocessing module with PCEN and synthetic mic distortion logic.

### Component 2: Notebook Phase (Step 1)
#### [NEW] `04_Robust_CRNN_PCEN/train_crnn_pcen.ipynb`
- Jupyter notebook implementing dataset loading, GroupKFold, PCEN extraction, CRNN training, and 1:1 balanced test evaluation.

### Component 3: Live Script Phase (Step 2)
#### [NEW] `04_Robust_CRNN_PCEN/run_live_crnn.py`
- Isolated, single-purpose live mic monitoring script featuring real-time VU-meter, mic auto-selection, manual recording toggle, and clean logging.
#### [NEW] `04_Robust_CRNN_PCEN/test_file_crnn.py`
- Dedicated offline file benchmark script.

### Component 4: Quantization Phase (Step 3)
#### [NEW] `04_Robust_CRNN_PCEN/quantize_crnn.py`
- Script to convert model to INT8 TFLite and generate C++ header array for microcontrollers.

### Component 5: Deployment Guides (Steps 4 & 5)
#### [NEW] `04_Robust_CRNN_PCEN/manual/README.md`
#### [NEW] `04_Robust_CRNN_PCEN/manual/RASPBERRY_PI_GUIDE.md`
#### [NEW] `04_Robust_CRNN_PCEN/manual/ARDUINO_NANO_BLE_GUIDE.md`

---

## Verification Plan

### Automated Tests
1. **1:1 Balanced Evaluation**: Verify test metrics on a strict 50/50 gunshot to non-gunshot dataset split.
2. **Synthetic Mic Invariance Test**: Confirm that accuracy on synthetic mic-distorted audio is within 3% of clean audio.
3. **Quantization Validation**: Confirm INT8 model predictions match float32 predictions within 1% margin.

### Manual Verification
1. **Live VU-Meter Test**: Run `run_live_crnn.py` and verify the terminal volume bar reacts in real time to speaking/clapping.
2. **Live Phone Audio Test**: Play gunshot audio from phone to mic and confirm detection trigger.
3. **Manual Record Test**: Press `[R]` to save a 5-second sample and inspect the saved `.wav` file.
