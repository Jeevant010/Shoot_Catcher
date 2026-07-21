# 📘 Shoot_Catcher — Complete Operational Manual & System Documentation

Welcome to the **Shoot_Catcher System Documentation**. This manual explains the complete architecture, folder structure, trained models, operational workflows, and exact file storage locations.

---

## 📂 1. Directory Structure & Overview

```text
Shoot_Catcher/
├── 01_1D_CNN/                             ← Module 1: Baseline 1D Waveform CNN
│   ├── 1d_cnn_gunshot_detector.ipynb     ← Training Notebook
│   ├── manual/README.md                   ← Theory & Manual
│   └── output/                            ← Trained Model Storage
│       ├── 1d_cnn_best.h5                 ← Trained Keras Model Weights
│       ├── 1d_cnn_float32.tflite          ← Full-precision TFLite Model
│       ├── 1d_cnn_int8.tflite             ← INT8 Quantized Edge Model
│       └── 1d_cnn_results.json            ← Metric Evaluation Results
│
├── 02_2D_CNN/                             ← Module 2: Baseline 2D Mel Spectrogram CNN
│   ├── 2d_cnn_gunshot_detector.ipynb     ← Training Notebook
│   ├── manual/README.md                   ← Spectrogram & 2D CNN Manual
│   └── output/                            ← Trained Model Storage
│       ├── 2d_cnn_mel_spectrogram_best.h5 ← Trained Keras Model Weights
│       ├── 2d_cnn_norm_stats.json         ← Spectrogram Mean & Std Normalization Stats
│       └── 2d_cnn_mel_spectrogram_results.json
│
├── Enhanced_Models/                       ← Advanced & Multi-Head Architectures
│   ├── 01_Enhanced_1D_CNN/               ← Module 3: Enhanced 1D CNN (Dual-Head)
│   │   ├── enhanced_1d_cnn.ipynb
│   │   └── output/
│   │       ├── enhanced_1d_cnn_best.h5    ← Gunshot + Anomaly Dual-Head Model
│   │       └── enhanced_1d_cnn_results.json
│   │
│   └── 02_Enhanced_2D_CNN/               ← Module 4: Enhanced 2D CNN
│       └── enhanced_2d_cnn.ipynb          (Notebook ready for training)
│
├── 03_Mic_Test/                           ← Intelligence Hub & Live Testing Suite
│   ├── mic_test_and_inference.ipynb      ← Interactive Jupyter Test Suite
│   ├── test_recording.wav                 ← Default Audio File for Offline Testing
│   ├── manual/README.md                   ← Mic Setup & Ring Buffer Theory
│   └── scripts/                           ← Main Operational Scripts
│       ├── live_demo.py                   ⭐ Multi-Model Live Dashboard & Runner
│       ├── test_on_recording.py           🎵 Offline Audio File Benchmark Runner
│       ├── record_test.py                 🎙️ Standalone 5-second Audio Recorder
│       ├── detections_log.txt             📝 Saved Detection Log File
│       ├── recordings/                    💾 Saved Triggered WAV Audio Snippets
│       └── test_output/                   💾 Saved Window Audio Clips
│
├── DOCUMENTATION.md                       ← This Comprehensive Manual
├── RUN_GUIDE.md                           ← Execution Quick Start Guide
└── README.md                              ← High-Level Project Overview
```

---

## 🎯 2. The 4 Model Modules & Status

Shoot_Catcher is organized into 4 distinct model modules:

| # | Module Name | Model File Path | Status | Model Type | Input Shape |
|---|-------------|-----------------|--------|------------|-------------|
| **1** | **Baseline 1D CNN** | `01_1D_CNN/output/1d_cnn_best.h5` | ✅ **Trained & Loaded** | 1D Raw Waveform | `(None, 16537, 1)` |
| **2** | **Baseline 2D CNN** | `02_2D_CNN/output/2d_cnn_mel_spectrogram_best.h5` | ✅ **Trained & Loaded** | 2D Mel Spectrogram | `(None, 64, 130, 1)` |
| **3** | **Enhanced 1D CNN** | `Enhanced_Models/01_Enhanced_1D_CNN/output/enhanced_1d_cnn_best.h5` | ✅ **Trained & Loaded** | 1D Dual-Head (Gunshot + Anomaly) | `(None, 16537, 1)` |
| **4** | **Enhanced 2D CNN** | `Enhanced_Models/02_Enhanced_2D_CNN/output/enhanced_2d_cnn_best.h5` | ⚠️ **Not Trained / Missing** | 2D Spectrogram + Augmentations | Auto-detected on train |

> 💡 **Graceful Fallback**: If a model file is missing (such as `Enhanced 2D CNN`), the system automatically tags it as `[Untrained / File Missing]`, skips it during live inference, and runs all trained models without throwing errors or crashing. When you train it in the future, it is auto-discovered!

---

## 💾 3. Where Everything Is Stored

When running live monitoring, benchmarks, or tests, all outputs are automatically organized into clean, isolated directories:

### 📝 A. Detections Text Log
* **Location:** [Shoot_Catcher/03_Mic_Test/scripts/detections_log.txt](file:///d:/Desktop/GunShot/Shoot_Catcher/03_Mic_Test/scripts/detections_log.txt)
* **What it contains:** Every gunshot detection event triggered by any model, formatted with exact date, timestamp, RMS energy level, and confidence scores across all active models.
* **Format:**
  ```text
  2026-07-22 [04:15:32] GUNSHOT DETECTED | RMS: 0.04821 | Baseline 1D CNN: 0.8540 | Baseline 2D CNN: 0.9421 | Enhanced 1D CNN: 0.9182
  ```

### 🔊 B. Triggered Audio Recordings (.wav)
* **Location:** [Shoot_Catcher/03_Mic_Test/scripts/recordings/](file:///d:/Desktop/GunShot/Shoot_Catcher/03_Mic_Test/scripts/recordings/)
* **What it contains:** Exact audio snippets captured during gunshot detection events, saved so you can listen to what the model "heard".
* **Filenames:**
  * `GUNSHOT_HHMMSS_rms0.048.wav` — Audio clip captured when gunshot threshold was triggered.
  * `ambient_HHMMSS_prob0.003.wav` — Periodic ambient background noise samples saved every ~8 seconds for baseline verification.

### 🎵 C. Benchmark Window Audio Clips (.wav)
* **Location:** [Shoot_Catcher/03_Mic_Test/scripts/test_output/](file:///d:/Desktop/GunShot/Shoot_Catcher/03_Mic_Test/scripts/test_output/)
* **What it contains:** Individual 250ms/750ms processed sliding window audio slices produced when running audio file benchmarks in `live_demo.py` or `test_on_recording.py`.
* **Filenames:** `window_001.wav`, `window_002.wav`, ...

---

## 🚀 4. How to Work Here (Operational Workflows)

All live testing, monitoring, and benchmarking is managed through the main operational script:
```bash
python 03_Mic_Test/scripts/live_demo.py
```

### Main Menu Options

When launched, `live_demo.py` scans your system, prints the Model Audit Table, and opens the main menu:

```text
⚙️  MAIN MENU — CHOOSE AN ACTION:
 ─────────────────────────────────────────────────────────────
 [1] 🚀 Run Multi-Model Live Real-Time Dashboard (All Trained Models)
 [2] 🎯 Run Single-Model Live Microphone Stream
 [3] 🎵 Run Multi-Model Benchmark on Audio File (.wav)
 [4] 🎙️ Quick Record 5s & Run Multi-Model Benchmark
 [5] 📋 View Model Status Audit & Architecture Info
 [0] 🚪 Exit
 ─────────────────────────────────────────────────────────────
```

---

### Step-by-Step Option Guide:

#### 1️⃣ Option `[1]` — Multi-Model Live Real-Time Dashboard
* **What it does:** Streams audio continuously from your chosen microphone and evaluates **ALL active trained models simultaneously** in real-time.
* **Output:** Displays live side-by-side confidence progress bars for each model:
  ```text
  🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
  [04:20:15] 🔫 GUNSHOT DETECTED! (RMS: 0.05214)
  ----------------------------------------------------------------------
    ├─ Baseline 1D CNN                 :  85.4% [████████████████████░░░░░]
    ├─ Baseline 2D CNN (Mel Spectrogram):  94.2% [███████████████████████░░]
    └─ Enhanced 1D CNN (Dual-Head)     :  91.8% [██████████████████████░░░] | Anomaly: 0.1240
  🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
  ```
* **Logs & Audio:** Automatically writes entry to `detections_log.txt` and saves `.wav` clip to `recordings/`.

#### 2️⃣ Option `[2]` — Single-Model Live Stream
* **What it does:** Allows you to pick 1 specific model (e.g. `Baseline 2D CNN`) to monitor live audio with minimal CPU overhead.

#### 3️⃣ Option `[3]` — Multi-Model Audio File Benchmark
* **What it does:** Loads a `.wav` file (e.g. `test_recording.wav` or a custom downloaded file), slides a 750ms window across the entire audio with 50% overlap, and outputs a side-by-side benchmark comparing every model's predictions frame-by-frame.
* **Summary Output:** Prints a comparison table showing total detection counts per model.

#### 4️⃣ Option `[4]` — Quick Record 5s & Benchmark
* **What it does:** Records 5 seconds from your microphone on the spot, saves `test_recording.wav`, and immediately runs Option `[3]` multi-model benchmark on it!

#### 5️⃣ Option `[5]` — Model Status Audit
* **What it does:** Displays the system audit table showing file paths, load status, and input/output tensor shapes for all 4 modules.

---

## 🎤 5. Microphone Selection Guide

When running live options (`[1]`, `[2]`, or `[4]`), you will be prompted to select a microphone:

```text
🎤 --- AVAILABLE MICROPHONES ---
 [1] Microsoft Sound Mapper - Input (Rate: 44100Hz, Channels: 2)
 [2] Microphone Array (Intel® Smart...) (Rate: 44100Hz, Channels: 2)
 [7] Stereo Mix (Realtek HD Audio...) (Rate: 48000Hz, Channels: 2)
```

### Which Option Should You Choose?

* 🎙️ **To test real-world room sound (clapping, speaking, playing sound on phone):**
  * **Select `2` or `4`** (`Microphone Array - Intel Smart Sound`).
  * This uses your laptop's physical built-in microphone array.

* 🔊 **To test sound playing DIRECTLY on your computer (YouTube, VLC, audio files):**
  * **Select `7`** (`Stereo Mix`).
  * This routes internal laptop sound output directly into the model without background room noise!

---

## 🔧 6. Troubleshooting & FAQs

### Q1: "ModuleNotFoundError: No module named 'sounddevice'"
* **Fix:** Install the missing package:
  ```bash
  pip install sounddevice soundfile scipy librosa tensorflow
  ```

### Q2: "Microphone is recording pure silence (Peak < 0.01)"
* **Fix:** Check Windows privacy settings:
  1. Open **Start → Settings → Privacy & Security → Microphone**.
  2. Ensure **"Microphone access"** is turned **ON**.
  3. Ensure **"Let desktop apps access your microphone"** is turned **ON**.

### Q3: "What if I train Enhanced 2D CNN later?"
* **Answer:** Simply run `enhanced_2d_cnn.ipynb` to generate `enhanced_2d_cnn_best.h5` in `Enhanced_Models/02_Enhanced_2D_CNN/output/`. On next launch of `live_demo.py`, it will automatically switch status to `✅ TRAINED & LOADED` and include it in live multi-model dashboards!
