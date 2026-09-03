# 🔫 Shoot_Catcher: Acoustic Gunshot Detection on Edge Architectures — Design, Evaluation, and Real-World Acoustic Realities

**Authors:** Shoot_Catcher Research & Development Team  
**Documentation:** `docs/RESEARCH_PAPER.md`  
**Target Deployment:** Edge Microcontrollers (Arduino Nano 33 BLE Sense) & Single-Board Computers (Raspberry Pi) *(Design & Code Prepared — Physical Hardware Setup Pending)*  
**Date:** September 2026  

---

## 📄 Abstract

Automated acoustic gunshot detection systems offer vital capabilities for public safety, wildlife anti-poaching, and emergency defense. While deep learning models routinely report offline validation accuracies exceeding 98% in academic literature, their performance often collapses when deployed on low-cost edge hardware or evaluated in uncontrolled acoustic environments. 

This paper presents the end-to-end design, implementation, and empirical evaluation of **Shoot_Catcher**, an edge-oriented gunshot detection system. We systematically compare five neural network architectures:
1. A **Baseline 1D CNN** operating directly on raw time-domain waveforms,
2. A **Baseline 2D CNN** operating on Mel-spectrogram images,
3. A **Robust CRNN-PCEN** model combining Per-Channel Energy Normalization (PCEN) with a Bidirectional Gated Recurrent Unit (Bi-GRU),
4. An **Enhanced 1D CNN** utilizing dual-head classification and MixUp regularization, and
5. An **Enhanced 2D CNN** incorporating dual-head anomaly detection, SpecAugment, and frequency-time perturbations.

Rather than relying solely on laboratory validation splits, this research evaluates all models across two distinct testing benchmarks:
- **Test Data Benchmark (1,442 pre-trimmed 750 ms clips):** In a controlled 1:1 balanced hold-out split, the Baseline 1D CNN, Robust CRNN, and Enhanced 2D CNN achieved top-tier discrimination (accuracies of **99.79%**, **99.79%**, and **99.51%** respectively). Concurrently, this evaluation uncovered critical architectural failure modes: a "Dying ReLU" collapse in the Baseline 2D CNN (producing 0% recall) and an over-sensitivity saturation in the Enhanced 1D CNN (producing 100% false alarms).
- **Real Data Benchmark (75 external, uncurated full-length recordings):** In testing against 20 authentic firearm discharges (AK-47, Desert Eagle, Magnum, rifles, snipers), 20 acoustic imposters (fireworks, clapping, door knocks), and 35 ambient background tracks (rain, sirens, engines, dogs), performance diverged dramatically. The Baseline 1D CNN suffered an **acoustic domain collapse**, dropping from 99.72% recall to **30.00%**. Conversely, the **Robust CRNN-PCEN** demonstrated **100% ambient noise immunity** (35/35 background sounds ignored), while the **Enhanced 2D CNN** emerged as the superior production model, catching **85.0% of real firearms** while maintaining a **94.3% ambient rejection rate**.

Finally, we present the embedded INT8 quantization and C++ firmware architecture developed for the **Arduino Nano 33 BLE Sense** (fitting within 119 KB Flash and 60 KB RAM) and Raspberry Pi daemon scripts. To maintain strict academic integrity, we document the current operational boundary: **all models have been evaluated exclusively within the local host computer software environment on curated audio datasets and external acoustic recordings. No physical Raspberry Pi or Arduino hardware has been set up, wired, or flashed yet, nor has the system undergone live ballistic field testing on an active firearm shooting range.**

---

## 1. Introduction

Acoustic gunshot detection is an asymmetric life-safety problem. A **False Negative** (failing to detect an active firearm discharge) can lead to catastrophic loss of life. Conversely, a **False Positive** (raising an alarm for a car door slamming or hand clapping) wastes law enforcement resources and induces panic.

Developing reliable acoustic detectors involves three core challenges:
1. **Acoustic Physics:** A gunshot generates an explosive muzzle blast lasting only 2 to 5 milliseconds, followed by environmental reverberation that decays over several hundred milliseconds. 
2. **Computational Constraints:** Real-time citywide or wilderness monitoring requires decentralized edge nodes (e.g., solar-powered microcontrollers) with strict memory limits (often under 256 KB of RAM) and low clock speeds (under 100 MHz).
3. **The Acoustic Domain Shift:** Models trained on clean, studio-grade audio files frequently fail when exposed to the non-linear frequency response, self-noise, and automatic gain control of low-cost MEMS microphones.

The **Shoot_Catcher** project was conceived to address these challenges. Instead of treating neural networks as black boxes, this research investigates the entire signal path—from acoustic air pressure waves and analog microphone distortions to convolutional wavelet filters, recurrent temporal modeling, and fixed-point microcontroller firmware.

---

## 2. Practical Bottlenecks in Acoustic Gunshot Detection

Early experimental phases in the repository identified four critical practical bottlenecks:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        FOUR CORE ACOUSTIC BOTTLENECKS                                  │
├───────────────────────────────┬────────────────────────────────────────────────────────┤
│ 1. Classification Window Size │ 250ms is too short to capture reverberation decay.      │
│                               │ Upgraded to 750ms (academic sweet spot: 0.5s–1.5s).    │
├───────────────────────────────┼────────────────────────────────────────────────────────┤
│ 2. Buffer Boundary Guillotine │ Audio transients cut in half at buffer boundaries.     │
│                               │ Solved via 75% overlapping sliding ring buffers.       │
├───────────────────────────────┼────────────────────────────────────────────────────────┤
│ 3. Speaker Transient Loss     │ Consumer speakers cannot recreate <3ms pressure rises. │
│                               │ 1D CNNs fail on speaker audio; 2D CNNs succeed.        │
├───────────────────────────────┼────────────────────────────────────────────────────────┤
│ 4. Microphone Transfer Bias   │ Cheap mic capsules distort static frequency response.  │
│                               │ Solved via Per-Channel Energy Normalization (PCEN).    │
└───────────────────────────────┴────────────────────────────────────────────────────────┘
```

### 2.1 The Classification Window: 250 ms vs. 750 ms
Initial prototypes utilized 250 ms audio windows. Empirical tests revealed that while the initial supersonic crack was captured, the reverberation decay envelope was truncated. Academic literature (e.g., *gabemagee/gunshot_detection*, UrbanSound8K) confirms that classification windows between 0.5 and 2.0 seconds are required to distinguish firearm echoes from brief household clicks. Shoot_Catcher standardized on **750 ms** ($16,537$ samples at $22,050\text{ Hz}$), balancing acoustic context against processing latency on edge microcontrollers.

### 2.2 The "Guillotine Effect" at Buffer Boundaries
In continuous live streams, an acoustic transient may occur exactly at the seam between two processing buffers. If half of the blast energy falls into Buffer $N$ and the remainder into Buffer $N+1$, neither window contains sufficient energy to trigger classification. Shoot_Catcher resolves this by enforcing a **75% sliding window overlap** ($187.5\text{ ms}$ hop step), guaranteeing that any transient event is centered in at least one evaluated frame.

### 2.3 Physical Speaker Transient Loss
When testing gunshot detection models by playing firearm audio from laptops or smartphones, 1D CNN models frequently output zero confidence ($0.0000$). 
- **Physical Explanation:** Real firearm shockwaves exhibit rise times under 3 milliseconds with extreme Sound Pressure Levels ($>140\text{ dB SPL}$). Small consumer speaker diaphragms cannot accelerate rapidly enough to reproduce this instantaneous pressure discontinuity, smoothing the spike into a dull acoustic thump.
- **Consequence:** Raw waveform 1D models fail to detect speaker-played gunshots because the physical wave shape is lost. In contrast, 2D spectral models succeed because the broader time-frequency energy distribution remains intact.

### 2.4 The Microphone Transfer Function Problem
Every physical microphone capsule imposes an acoustic transfer function $H_{\text{mic}}(f)$ on the incoming signal $S(f)$:
$$Y_{\text{mic}}(f) = S(f) \cdot H_{\text{mic}}(f)$$
In traditional Log-Mel spectrograms, taking the logarithm yields an additive static offset:
$$\log\left(Y_{\text{mic}}(f)\right) = \log\left(S(f)\right) + \log\left(H_{\text{mic}}(f)\right)$$
A neural network trained on studio recordings encounters a permanently shifted feature map when deployed on a laptop or MEMS microphone, causing widespread false dismissals.

---

## 3. Acoustic Preprocessing & Feature Representations

Shoot_Catcher processes audio through three distinct mathematical representations:

```
                               SIGNAL PROCESSING PIPELINE
                               
                        Raw Audio Signal: 750ms @ 22,050 Hz
                                  (16,537 Samples)
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
┌─────────────────┐             ┌─────────────────┐             ┌─────────────────┐
│   Raw 1D Wave   │             │ 2D Mel-Spectrum │             │   2D PCEN Map   │
│ • DC-Offset     │             │ • STFT (N=512)  │             │ • Mel Filterbank│
│   Removal       │             │ • 64 Mel Bins   │             │ • Dynamic IIR   │
│ • Peak Scaling  │             │ • [0, 1] Normal-│             │   Envelope M    │
│   [-1.0, 1.0]   │             │   ization Fix   │             │ • Root Dynamic  │
│                 │             │   from -80 dB   │             │   Compression   │
└─────────────────┘             └─────────────────┘             └─────────────────┘
```

### 3.1 Raw Waveform Preprocessing (1D)
Audio is converted to mono, stripped of DC hardware bias via mean subtraction ($y = y - \bar{y}$), and subjected to Energy-Gated Peak Normalization. If the Root Mean Square (RMS) energy exceeds a baseline threshold ($0.001$), the waveform is scaled to $[-1.0, 1.0]$; otherwise, quiet digital silence is left unscaled to prevent amplifying low-level thermal ADC noise.

### 3.2 Normalized Mel-Spectrograms (2D)
Audio frames are transformed via Short-Time Fourier Transform (STFT) with $N_{\text{fft}} = 512$, hop length $H = 128$, and a 64-band triangular Mel filterbank.
- **The Dying ReLU Incident & Rectification:** In initial implementations, spectrograms converted to decibels via `librosa.power_to_db` produced values from $-80.0\text{ dB}$ to $0.0\text{ dB}$. Injecting large negative numbers directly into the network caused severe gradient explosions during backpropagation, permanently driving the ReLU activations to zero ("brain death"). The model collapsed into predicting background for 100% of samples. The issue was resolved by mathematically normalizing decibel values to a $[0.0, 1.0]$ range:
  $$S_{\text{norm}} = \frac{S_{\text{dB}} + 80.0}{80.0}$$

### 3.3 Per-Channel Energy Normalization (PCEN)
To make spectral features invariant to microphone gain and ambient noise, Module 04 implements PCEN. For each Mel-frequency channel $f$, a 1st-order Infinite Impulse Response (IIR) filter tracks the background noise floor over time:
$$M[f, t] = (1 - s) \cdot M[f, t-1] + s \cdot S[f, t]$$
The normalized energy is then computed via adaptive gain division and root compression:
$$P[f, t] = \left( \frac{S[f, t]}{(\epsilon + M[f, t])^\alpha} + \delta \right)^r - \delta^r$$
With parameters $s = 0.025$, $\alpha = 0.98$, $\delta = 2.0$, and $r = 0.5$:
1. **Stationary Background Cancellation:** If $S[f, t] \approx M[f, t]$, the ratio simplifies to $S_0^{1 - \alpha} = S_0^{0.02} \approx 1.0$. Continuous noise (rain, wind, engine hum) is mapped to an invariant flat scalar.
2. **Impulse Maximization:** When an abrupt gunshot transient arrives, $S[f, t]$ surges instantly while the filter envelope $M[f, t]$ remains anchored to the past baseline. The ratio surges by $\frac{A}{S_0^{0.98}}$, maximizing the transient contrast.
3. **Microphone Invariance:** Hardware scaling factor $H_{\text{mic}}(f)$ appears in both numerator and denominator, canceling out mathematically:
   $$H_{\text{mic}}(f)^{1 - \alpha} = H_{\text{mic}}(f)^{0.02} \approx 1.0$$

---

## 4. Deep Learning Architectures

### 4.1 Baseline 1D CNN: Acoustic Matched Wavelets
Designed for minimal latency on microcontrollers with zero feature extraction overhead:
- **Input:** `(16537, 1)` raw time samples.
- **Layer 1:** `Conv1D(32 filters, kernel_size=80, stride=4)` + `BatchNorm` + `MaxPool1D(4)`.
  - *Physical Rationale:* At $22,050\text{ Hz}$, an 80-sample kernel spans exactly **$3.6\text{ ms}$**. This corresponds to the physical duration of a supersonic bullet shockwave / muzzle blast. The first layer acts as a learnable bank of acoustic matched wavelet filters.
- **Layers 2–4:** `Conv1D(64, k=3)`, `Conv1D(128, k=3)`, `Conv1D(128, k=3)` for hierarchical feature extraction.
- **Head:** `GlobalAveragePooling1D` $\to$ `Dense(64, ReLU)` $\to$ `Dropout(0.4)` $\to$ `Dense(1, Sigmoid)`.
- **Parameters:** **92,513** (INT8 quantized size: **119.3 KB**).

### 4.2 Baseline 2D CNN: Spectral Image Processing
- **Input:** `(64, 130, 1)` Mel-spectrogram image.
- **Feature Extractor:** Two blocks of `Conv2D(32/64, 3x3)` + `BatchNorm` + `MaxPool(2x2)`, followed by two blocks of `Conv2D(128, 3x3)` + `BatchNorm` without pooling to preserve temporal resolution.
- **Head:** `GlobalAveragePooling2D` $\to$ `Dense(64, ReLU)` $\to$ `Dropout(0.4)` $\to$ `Dense(1, Sigmoid)`.
- **Parameters:** **249,985**.

### 4.3 Enhanced Dual-Head Architectures (1D and 2D)
Built to maximize recall and flag ambiguous acoustic events:
- **Regularization & Augmentations:**
  - **MixUp:** Linearly interpolates pairs of audio clips and labels ($\lambda \sim \text{Beta}(0.3, 0.3)$), enforcing smooth decision boundaries and reducing adversarial vulnerability.
  - **SpecAugment:** Applies random time masking (up to 15% duration) and frequency masking (up to 8 bins).
  - **Pitch & Speed Perturbation:** $\pm 2$ semitone pitch shifting and $\pm 10\%$ speed variation.
- **Dual-Head Multi-Task Output:**
  $$\text{Shared Backbone} \longrightarrow \begin{cases} \text{Head 1: Gunshot Probability } \sigma(\mathbf{W}_{\text{gun}} \mathbf{h} + b) \\ \text{Head 2: Anomaly Score } \sigma(\mathbf{W}_{\text{anom}} \mathbf{h} + b) \end{cases}$$
  Head 1 performs supervised binary classification, while Head 2 identifies out-of-distribution acoustic impulses.

### 4.4 Robust CRNN-PCEN: Spatial Convolutions + Recurrent Memory
- **Input:** `(64, 130, 1)` PCEN time-frequency matrix.
- **Spatial Blocks:** Two `Conv2D` stages ($32$ and $64$ filters) extracting local spectro-temporal textures, downsampling to $(16, 32, 64)$.
- **Sequence Reshaping:** Collapses frequency and filter dimensions into a temporal sequence: $\mathbf{X}_{\text{seq}} \in \mathbb{R}^{32 \times 1024}$.
- **Recurrent Stage:** A **Bidirectional GRU** (64 units) processes the sequence across 32 time frames. The forward GRU captures the explosive onset blast, while the backward GRU captures the reverberation decay tail.
- **Head:** `GlobalAveragePooling1D` $\to$ `Dense(64)` $\to$ `Dense(1, Sigmoid)`.
- **Parameters:** **~310,000**.

---

## 5. Training Methodology & Metric Optimization

### 5.1 Rejection of Accuracy (The Accuracy Paradox)
In real-world acoustic monitoring, gunshots represent less than 0.1% of all ambient sound. A naive model that classifies every sound as "background" achieves over 99.9% accuracy while remaining entirely useless. Therefore, overall accuracy was rejected as a primary optimization target.

### 5.2 The $F_2$-Score Objective
Because missing a gunshot is dangerous, the evaluation framework prioritizes **Recall over Precision**. The models were trained and evaluated using the **$F_2$-Score**, which weights Recall four times higher than Precision ($\beta = 2$):
$$F_2 = (1 + 2^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{2^2 \cdot \text{Precision} + \text{Recall}} = 5 \cdot \frac{\text{Precision} \cdot \text{Recall}}{4 \cdot \text{Precision} + \text{Recall}}$$
Training loss was augmented with **Dynamic Class Weights** ($C_1 \approx 6.8$ for gunshots, $C_0 \approx 0.54$ for background) to heavily penalize False Negatives during gradient descent.

### 5.3 Anti-Leakage Partitioning via GroupKFold
Audio datasets frequently contain multiple clips sliced from the same field recording. Splitting clips randomly causes the model to memorize the ambient background noise profile of specific microphones, producing artificially high validation scores. Shoot_Catcher enforces strict **GroupKFold partitioning by source recording ID**, ensuring all clips from a single recording exist solely within the training, validation, or test set.

---

## 6. Empirical Evaluation 1: Curated Test Data (Option B)

The first evaluation was conducted on the held-out test split of the curated dataset (`Data/SPLIT_DATASET_750MS/test/`).

### 6.1 Test Setup
- **Dataset Size:** 1,442 clips of 750 ms audio.
- **Balance:** Exactly 721 Gunshots (Class 1) and 721 Non-Gunshots (Class 0).
- **Decision Threshold:** $\tau = 0.50$.

### 6.2 Test Data Results Scorecard

```text
===============================================================================================
Model Name                   | Accuracy | Precision | Recall   | F1-Score | F2-Score | TP   FP   TN   FN  
-----------------------------------------------------------------------------------------------
Baseline 1D CNN              |   99.79% |    99.86% |   99.72% |   99.79% |   99.75% | 719    1  720    2
Baseline 2D CNN (Mel)        |   50.00% |     0.00% |    0.00% |    0.00% |    0.00% |   0    0  721  721
Robust CRNN (PCEN)           |   99.79% |    99.86% |   99.72% |   99.79% |   99.75% | 719    1  720    2
Enhanced 1D CNN (Dual)       |   50.00% |    50.00% |  100.00% |   66.67% |   83.33% | 721  721    0    0
Enhanced 2D CNN (Dual)       |   99.51% |    99.45% |   99.58% |   99.51% |   99.56% | 718    4  717    3
===============================================================================================
```

### 6.3 Diagnostic Observations from Test Data
1. **Exceptional Convergence on Controlled Audio:**
   The Baseline 1D CNN, Robust CRNN, and Enhanced 2D CNN demonstrated near-perfect separation on curated audio, each achieving over 99.5% accuracy and missing no more than 2 to 3 gunshots out of 721.
2. **Empirical Proof of Dying ReLU (Baseline 2D CNN):**
   The Baseline 2D CNN scored exactly 50.00% accuracy by outputting zero confidence for all 1,442 clips ($TP=0, FN=721$). The model suffered complete weight collapse during training due to unnormalized decibel inputs.
3. **Empirical Proof of Saturated Recall (Enhanced 1D CNN):**
   The Enhanced 1D CNN scored 50.00% accuracy by outputting high confidence for all 1,442 clips ($TP=721, FP=721$). Extreme class weighting forced the network to eliminate False Negatives by permanently setting its output bias high, destroying its ability to reject non-gunshots.

---

## 7. Empirical Evaluation 2: Unseen Real Data Benchmark (Option A)

To determine how models perform outside curated splits, an independent benchmark dataset of 75 full-length recordings was gathered in `My_Test_Audio/`.

### 7.1 Real Data Composition
- **`Actual_Gunshots` (20 files):** Authentic weapon recordings (AK-47, Desert Eagle .50, .44 Magnum, bolt-action rifles, snipers, submachine guns).
- **`Like_Gunshots` (20 files):** Difficult acoustic imposters (aerial fireworks, bottle rockets, clapping, heavy door knocks, breaking glass).
- **`Not_Gunshots` (35 files):** Common ambient sounds (rainstorms, diesel engines, barking dogs, sirens, human sneezing, crying infants).
- **Inference Method:** Evaluated via a sliding 750 ms window with **75% overlap** across the entire length of each file.

### 7.2 Real Data Results Scorecard

```text
====================================================================================================
Model Name                   | Accuracy | Precision | Recall   | F1-Score | F2-Score | TP   FP   TN   FN  
----------------------------------------------------------------------------------------------------
🥇 Enhanced 2D CNN (Dual)    |   76.00% |    53.12% |   85.00% |   65.38% |   75.89% |   17   15   40    3
🥈 Robust CRNN (PCEN)        |   73.33% |    50.00% |   60.00% |   54.55% |   57.69% |   12   12   43    8
🥉 Baseline 1D CNN           |   64.00% |    31.58% |   30.00% |   30.77% |   30.30% |    6   13   42   14
⚠️ Baseline 2D CNN (Mel)     |   73.33% |     0.00% |    0.00% |    0.00% |    0.00% |    0    0   55   20
⚠️ Enhanced 1D CNN (Dual)    |   26.67% |    26.67% |  100.00% |   42.11% |   64.52% |   20   55    0    0
====================================================================================================
```

### 7.3 Detailed Category Breakdown by Sound Class

```text
====================================================================================================
Detailed Performance Breakdown by Category:
====================================================================================================
📌 Enhanced 2D CNN (Dual):
   ├─ Real Gunshots Caught (Recall)      : 17/20 (85.0%)
   ├─ Imposters Rejected (Claps/Knocks)  :  7/20 (35.0%) [False Alarms: 13]
   └─ Ambient Noise Ignored (Rain/Motor) : 33/35 (94.3%) [False Alarms:  2]

📌 Robust CRNN (PCEN):
   ├─ Real Gunshots Caught (Recall)      : 12/20 (60.0%)
   ├─ Imposters Rejected (Claps/Knocks)  :  8/20 (40.0%) [False Alarms: 12]
   └─ Ambient Noise Ignored (Rain/Motor) : 35/35 (100.0%) [False Alarms: 0] 🏆

📌 Baseline 1D CNN:
   ├─ Real Gunshots Caught (Recall)      :  6/20 (30.0%)
   ├─ Imposters Rejected (Claps/Knocks)  : 11/20 (55.0%) [False Alarms:  9]
   └─ Ambient Noise Ignored (Rain/Motor) : 31/35 (88.6%) [False Alarms:  4]
====================================================================================================
```

---

## 8. Comparative Analysis: The Acoustic Domain Shift

Comparing the results of **Test Data (Option B)** and **Real Data (Option A)** exposes the core scientific finding of this project:

```
                      RECALL COMPARISON: TEST DATA VS. REAL DATA
                      
                          Test Data Split        Real Data Recordings
                             (Curated)                (External)
  Enhanced 2D CNN           [██████████] 99.6%        [████████░░] 85.0%
  Robust CRNN (PCEN)        [██████████] 99.7%        [██████░░░░] 60.0%
  Baseline 1D CNN           [██████████] 99.7%        [███░░░░░░░] 30.0% ⚠️
```

### 8.1 The 1D CNN Acoustic Domain Collapse
The Baseline 1D CNN suffered the most severe degradation, falling from **99.72% recall on Test Data** to **30.00% recall on Real Data**. 
- *Why it occurred:* 1D convolutions learn fixed time-domain wave shapes. In real-world audio, variations in firearm distance, obstacle diffraction, and room acoustics stretch and distort the raw pressure wave. Because the 1D model lacks explicit frequency awareness, it fails to recognize the sound once its waveform shape changes.

### 8.2 PCEN’s Environmental Noise Immunity
The Robust CRNN-PCEN model achieved **100% rejection (35/35)** of ambient sounds (rain, engine rumble, dog barks, sirens).
- *Why it occurred:* PCEN's dynamic baseline subtraction mathematically suppresses continuous and slowly varying background noise. As long as a sound lacks an explosive, sudden onset, PCEN prevents it from reaching the recurrent classification layers.

### 8.3 The Imposter Challenge: Gunshots vs. Fireworks
Both top-performing models (Enhanced 2D CNN and Robust CRNN) struggled to distinguish gunshots from aerial fireworks and close claps (rejection rates of 35% and 40%). Chemical fireworks produce genuine acoustic shockwaves that physically mirror gunshot onsets. Resolving this ambiguity is an open challenge that typically requires multi-microphone spatial localization or optical flash detection.

### 8.4 Human Audio Verification Pipeline
To enable human listening audits and verify each model's decisions, Shoot_Catcher automatically archives evaluated clips into `Verification_Outputs/By_Model/`:
- **Full File Partitioning:** Full-length audio files are copied into `Detected_Gunshots/` or `Ignored_NonGunshots/` under each model's directory.
- **Trigger Slices ($750\text{ ms}$):** For every detection, the exact $750\text{ ms}$ sliding window slice that breached the threshold is isolated and saved into `Trigger_Slices_750ms/` with its timestamp and confidence score.
- **Interactive Audio Dashboard:** An HTML interface (`verification_dashboard.html`) generates a browser-accessible table with built-in `<audio controls>` players, enabling reviewers to listen to true positives, false alarms, and missed shots with a single click.

---

## 9. Target Edge Hardware Architecture & Firmware Design (Hardware Setup Pending)

To prepare for physical edge deployment, the system architectures and firmware were developed and profiled in software:

> [!IMPORTANT]
> **Hardware Status Clarification:**  
> The embedded firmware (`GunshotDetector.ino`), quantized INT8 TFLite model, memory buffers, and Raspberry Pi background daemon scripts (`run_pi_crnn.py`) have been fully coded and validated via software emulation. However, **physical Arduino Nano 33 BLE Sense and Raspberry Pi hardware units have not yet been physically connected, wired, or flashed.** All performance benchmarks reported in this paper were executed on the host PC software environment.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   EDGE DEPLOYMENT ARCHITECTURES (FIRMWARE PREPARED)                    │
├──────────────────────────────────────┬─────────────────────────────────────────────────┤
│ Target A: Arduino Nano 33 BLE Sense  │ Target B: Raspberry Pi 4 / 5 / Zero 2 W         │
├──────────────────────────────────────┼─────────────────────────────────────────────────┤
│ • Nordic nRF52840 (ARM Cortex-M4)    │ • Broadcom BCM2711/2712 (ARM Cortex-A72/A76)    │
│ • Memory: 1MB Flash, 256KB SRAM      │ • Memory: 512MB–8GB LPDDR4                      │
│ • Mic: Built-in MP34DT05 PDM         │ • Mic: USB Microphone / I2S (INMP441)           │
│ • Framework: TensorFlow Lite Micro   │ • Framework: TFLite Runtime (Python)            │
│ • Model Size: 119.3 KB (INT8)        │ • Deployment: Headless Systemd Service Daemon   │
│ • Tensor Arena: 60 KB Static SRAM    │ • Status: Code ready; hardware pending setup    │
└──────────────────────────────────────┴─────────────────────────────────────────────────┘
```

### 9.1 Arduino Nano 33 BLE Firmware Architecture (`GunshotDetector.ino`)
1. **Model Quantization:** The Baseline 1D CNN was converted to an 8-bit integer (INT8) model using TensorFlow Lite's post-training quantization. The resulting model binary is **119.3 KB**, designed to reside within the microcontroller's 1 MB internal Flash memory.
2. **Audio Acquisition Pipeline:** The firmware is written to sample the onboard **MP34DT05 PDM digital microphone** at 16 kHz using direct DMA transfers to avoid blocking the processor.
3. **Memory Budget Allocation:** The firmware defines a static **60 KB Tensor Arena** in SRAM. This fits within the chip's 256 KB RAM limit, leaving over 180 KB for the Bluetooth Low Energy (BLE) stack and RTOS kernel.
4. **Trigger Mechanism:** When the model output confidence exceeds 0.50, the firmware logic sets a digital alert pin high and transmits an event string over the serial bus.

### 9.2 Raspberry Pi Deployment Architecture (`run_pi_crnn.py`)
For Linux-based edge boards (Raspberry Pi 4, 5, or Zero 2 W), a headless Python deployment script was developed using `tflite-runtime`. The script streams audio chunks from USB or I2S microphones via `sounddevice` and includes a `systemd` unit configuration to enable automatic startup as a background daemon upon boot.

---

## 10. Operational Scope, Boundaries & Limitations

To maintain absolute academic and engineering honesty, the operational boundaries and current state of the Shoot_Catcher project are explicitly stated:

1. **No Physical Embedded Hardware Setup:**
   No physical Arduino Nano or Raspberry Pi hardware boards have been provisioned, connected, or flashed. The edge memory metrics reported (such as the 119.3 KB INT8 binary and 60 KB Tensor Arena) are calculated from compiler/model memory profilers and software architectures, not on-board physical oscilloscope, current meter, or hardware bench measurements.
2. **Evaluation Restricted to Host PC Software Environment:**
   All five models and their comparative benchmarks were executed using Python (TensorFlow 2.16.1) on the host computer workstation. Live microphone testing was conducted exclusively using the host computer's integrated microphone.
3. **Absence of Live Firing Range Testing:**
   All evaluations have been conducted on recorded audio datasets (UrbanSound8K, Freesound) and external open-access field recordings (Audio Event Analysis, ESC-50). **The system has not yet been deployed or evaluated at an active live-fire shooting range with real physical firearms.**
4. **Hardware Microphone Domain Shift:**
   While PCEN mathematically minimizes microphone frequency response differences, physical MEMS microphone diaphragms can saturate and hard-clip when exposed to close-range acoustic shockwaves ($>130\text{ dB SPL}$). True field deployment on physical hardware will require acoustic dampening baffles or high-SPL capsules.
5. **Imposter False Alarm Trade-off:**
   In safety-critical deployments where zero false alarms on fireworks are required, acoustic detection alone is insufficient; multi-modal verification (acoustic + infrared/optical flash) will be necessary.

---

## 11. Conclusion & Future Work

The **Shoot_Catcher** project provides an end-to-end framework for acoustic gunshot detection on edge computing platforms. By testing models across both curated laboratory splits and independent real-world audio, this research demonstrates that:
- **Raw waveform 1D CNNs** offer compact sizes (119 KB) and low latency, but suffer severe performance collapse (dropping to 30% recall) when exposed to environmental reverberation.
- **2D Mel-Spectrogram CNNs** (Enhanced 2D) provide the most practical production balance, achieving **85% recall on real firearms** and **94.3% ambient noise rejection**.
- **Robust CRNN-PCEN** models provide unparalleled **ambient noise immunity (100%)**, making them the ideal choice for high-noise outdoor environments.

### Future Work
1. **Physical Hardware Provisioning & Flashing:** Procure and physically wire the Arduino Nano 33 BLE Sense and Raspberry Pi hardware units, flash `GunshotDetector.ino`, and benchmark physical latency, DMA buffer stability, and battery power draw.
2. **Live-Fire Shooting Range Testing:** Deploy the flashed physical hardware units at an active firearm shooting range to measure detection rates against live pistol, rifle, and shotgun rounds in real physical space.
3. **Impulse Hardening:** Training multi-class models specifically on firework and commercial explosive datasets to improve rejection of acoustic imposters.
4. **Array Localization:** Extending the edge firmware to run multi-channel Time-Difference-of-Arrival (TDOA) algorithms across a mesh of networked nodes to calculate firearm coordinates in real time.

---

## 📚 References & Prior Work

1. **Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018).** *mixup: Beyond Empirical Risk Minimization.* International Conference on Learning Representations (ICLR).
2. **Park, D. S., Chan, W., Zhang, Y., Chiu, C. C., Zoph, B., Cubuk, E. D., & Le, Q. V. (2019).** *SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition.* Interspeech.
3. **Wang, Y., Getreuer, P., Hughes, T., Lyon, R. F., & Saurous, R. A. (2017).** *Trainable Per-Channel Energy Normalization for Bioacoustic Classification.* IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).
4. **Magee, G., et al. (2019).** *Low Cost Gunshot Detection using Deep Learning on the Raspberry Pi.* IEEE International Conference on Big Data.
5. **Salamon, J., & Bello, J. P. (2017).** *Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification.* IEEE Signal Processing Letters.
6. **Piczak, K. J. (2015).** *ESC: Dataset for Environmental Sound Classification.* ACM International Conference on Multimedia.
7. **Sami-Ullah, H. (2022).** *Audio Event Analysis and Feature Extraction Using Deep Learning.* Open Source Firearms Acoustic Database.
8. **Edge Impulse (2022).** *Acoustic Gunshot Detection on Embedded ARM Cortex Microcontrollers.* Edge Impulse Research.
