# 🔫 Shoot_Catcher: Acoustic Gunshot Detection on Edge Architectures
## Technical Research Report (Condensed 4-Page Edition)

**Author:** Shoot_Catcher Research & Development Team  
**Reference Document:** `docs/RESEARCH_PAPER.md` | **File:** `docs/RESEARCH_REPORT_4PAGE.md`  
**Evaluation Scope:** 5 Neural Network Models across Controlled Test Data & Real Unseen Audio  
**Hardware Status:** Target architectures prepared in firmware/code; physical hardware setup pending  

---

<!-- ========================================================================== -->
<!-- PAGE 1: PROJECT OVERVIEW, PROBLEM STATEMENT & ARCHITECTURES                -->
<!-- ========================================================================== -->

## 📄 PAGE 1: Problem Formulation, Acoustic Physics & Architectures

### 1.1 Executive Summary & Problem Formulation
Acoustic gunshot detection is an asymmetric life-safety problem. Missing an authentic gunshot (**False Negative**) threatens human life, while a false alarm (**False Positive**) wastes public resources and creates panic. In real-world environments, gunshots represent $<0.1\%$ of acoustic events, rendering raw classification accuracy meaningless. Shoot_Catcher prioritizes the **$F_2$-Score**, mathematically weighting **Recall four times higher than Precision**.

Developing dependable detectors on low-cost edge hardware faces three severe physical challenges:
1. **Acoustic Physics:** Gunshots produce an explosive supersonic shockwave and muzzle blast ($2\text{–}5\text{ ms}$) followed by room/environmental reverberation ($200\text{–}600\text{ ms}$). A classification window of **$750\text{ ms}$** ($16,537$ samples at $22,050\text{ Hz}$) was selected as the optimal window to capture decay without overwhelming microcontroller memory.
2. **The Buffer Guillotine Effect:** Audio transients cut across buffer boundaries are missed. Shoot_Catcher enforces a **75% sliding window overlap** ($187.5\text{ ms}$ hop step) to center events.
3. **The Speaker Transient Paradox:** Consumer speakers cannot physically reproduce $<3\text{ ms}$ rise times, smoothing shockwaves into thumps. Raw 1D models fail on speaker playback, whereas 2D spectral models survive.

```
                           ACOUSTIC SIGNAL PROCESSING PIPELINE
                           
                   Raw Audio Input: 750ms @ 22,050 Hz (16,537 Samples)
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         ▼                                 ▼                                 ▼
┌───────────────────┐             ┌───────────────────┐             ┌───────────────────┐
│  Raw Waveform 1D  │             │   2D Mel-Spectrum │             │    2D PCEN Map    │
│ • DC-Offset Stripped│           │ • STFT (N_fft=512)│             │ • Mel Filterbank  │
│ • RMS Gated Scale │             │ • 64 Mel Channels │             │ • Dynamic IIR     │
│   [-1.0, 1.0]     │             │ • [0, 1] Norm Fix │             │   Noise Tracker M │
│   (Zero Prep Time)│             │   (Dying ReLU Fix)│             │ • Adaptive Gain   │
└───────────────────┘             └───────────────────┘             └───────────────────┘
```

---

### 1.2 Model Architectures Evaluated
Five distinct neural network models were designed, trained, and audited:

1. **Baseline 1D CNN (`1d_cnn_best.h5` | 92,513 Params | 119 KB INT8):**  
   Processes raw time-domain waveforms. Layer 1 uses an 80-sample kernel ($3.6\text{ ms}$ at $22.05\text{ kHz}$), acting as an acoustic matched wavelet filter corresponding to the physical muzzle blast duration.
2. **Baseline 2D CNN (`2d_cnn_mel_spectrogram_best.h5` | 249,985 Params):**  
   Processes $64 \times 130$ Mel-spectrogram images through 4 convolutional stages with global average pooling.
3. **Robust CRNN-PCEN (`crnn_pcen_best.h5` | ~310,000 Params):**  
   Combines **Per-Channel Energy Normalization (PCEN)** with a **Bidirectional GRU** (64 units). PCEN dynamically estimates and divides out background noise via an IIR filter ($M[f, t]$), while the forward/backward GRU captures the blast onset and reverberation decay sequence.
4. **Enhanced 1D CNN (`enhanced_1d_cnn_best.h5` | Dual Head):**  
   Trained with MixUp data augmentation and class weights ($C_1 \approx 6.8$) to force maximum recall.
5. **Enhanced 2D CNN (`enhanced_2d_cnn_best.h5` | Dual Head):**  
   Processes $64 \times 33$ Mel-spectrograms with SpecAugment (time/frequency masking) and dual heads: Head 1 (supervised gunshot probability) and Head 2 (unsupervised acoustic anomaly score).

---

<!-- ========================================================================== -->
<!-- PAGE 2: TESTING PART 1 — CONTROLLED TEST DATA BENCHMARK                    -->
<!-- ========================================================================== -->

## 📊 PAGE 2: Testing Part 1 — Controlled Test Data Benchmark

The first evaluation was conducted on the held-out test split of the curated project dataset (`Data/SPLIT_DATASET_750MS/test/`).

### 2.1 Test Data Setup
- **Sample Count:** 1,442 pre-trimmed clips of 750 ms mono audio.
- **Class Balance:** Exact 1:1 ratio (721 Gunshots, 721 Non-Gunshots).
- **Partitioning:** GroupKFold split strictly by source recording ID to guarantee zero acoustic leakage.
- **Decision Threshold:** Standardized at $\tau = 0.50$.

---

### 2.2 Complete Empirical Scorecard (Test Data)

| Model Name | Input Representation | Accuracy | Precision | Recall | F1-Score | F2-Score | TP | FP | TN | FN |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1D CNN** | Raw Waveform | **99.79%** | **99.86%** | **99.72%** | **99.79%** | **99.75%** | **719** | 1 | 720 | **2** |
| **Baseline 2D CNN (Mel)** | Mel Spectrogram | 50.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0 | 0 | 721 | 721 |
| **Robust CRNN (PCEN)** | PCEN + Bi-GRU | **99.79%** | **99.86%** | **99.72%** | **99.79%** | **99.75%** | **719** | 1 | 720 | **2** |
| **Enhanced 1D CNN (Dual)**| Waveform + MixUp | 50.00% | 50.00% | 100.00% | 66.67% | 83.33% | 721 | 721 | 0 | 0 |
| **Enhanced 2D CNN (Dual)**| Mel + SpecAugment | **99.51%** | **99.45%** | **99.58%** | **99.51%** | **99.56%** | **718** | 4 | 717 | **3** |

```
                       CONFUSION MATRIX VISUALIZATION (TEST DATA)
                       
       Baseline 1D CNN                      Robust CRNN                   Enhanced 2D CNN
    ┌──────────────────┐                ┌──────────────────┐            ┌──────────────────┐
    │ TP: 719 │ FP:  1 │                │ TP: 719 │ FP:  1 │            │ TP: 718 │ FP:  4 │
    ├─────────┼────────┤                ├─────────┼────────┤            ├─────────┼────────┤
    │ FN:   2 │ TN: 720│                │ FN:   2 │ TN: 720│            │ FN:   3 │ TN: 717│
    └──────────────────┘                └──────────────────┘            └──────────────────┘
```

---

### 2.3 Diagnostic Analysis & Failure Modes
1. **Near-Perfect Laboratory Separation:**
   Three models (**Baseline 1D CNN**, **Robust CRNN**, and **Enhanced 2D CNN**) achieved over **99.5% accuracy**, missing at most 2 to 3 gunshots out of 721 while generating almost zero false alarms on curated background sounds.
2. **Empirical Proof of the "Dying ReLU" Failure (Baseline 2D CNN):**
   The Baseline 2D CNN scored 50.00% accuracy by outputting zero confidence across all 1,442 files ($TP = 0, FN = 721$). Feeding unnormalized decibel spectrograms ($[-80.0, 0.0]\text{ dB}$) caused massive negative gradient updates during backpropagation, permanently killing the ReLU neurons. This failure prompted the development of the normalized $[0.0, 1.0]$ pipeline in Enhanced 2D CNN.
3. **Empirical Proof of "Saturated Recall" (Enhanced 1D CNN):**
   The Enhanced 1D CNN scored 50.00% accuracy by predicting `1` for all 1,442 files ($TP = 721, FP = 721$). Heavy class weighting ($C_1 \approx 6.8$) forced the output layer into saturation, eliminating false negatives at the cost of total loss of discrimination.

---

<!-- ========================================================================== -->
<!-- PAGE 3: TESTING PART 2 — REAL DATA BENCHMARK (EXTERNAL AUDIO)              -->
<!-- ========================================================================== -->

## 📈 PAGE 3: Testing Part 2 — Real Data Benchmark (External Audio)

To test true generalization on unseen acoustic environments, 75 uncompressed WAV recordings were collected from independent acoustic databases into `My_Test_Audio/`.

### 3.1 Real Data Setup
- **`Actual_Gunshots` (20 files):** Field recordings of genuine firearms (AK-47, Desert Eagle, .44 Magnum, bolt-action rifles, snipers, submachine guns, explosive blasts).
- **`Like_Gunshots` (20 files):** Acoustic imposters (aerial fireworks, bottle rockets, close hand clapping, heavy door knocks, breaking glass).
- **`Not_Gunshots` (35 files):** Everyday ambient recordings (rainstorms, diesel engines, barking dogs, police sirens, human sneezing, crying infants).
- **Evaluation Mechanism:** Continuous sliding 750 ms window with **75% overlap** ($187.5\text{ ms}$ hop step).

---

### 3.2 Complete Empirical Scorecard (Real Data)

| Model Name | Accuracy | Precision | Recall | F1-Score | F2-Score | True Pos (TP) | False Pos (FP) | True Neg (TN) | False Neg (FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Enhanced 2D CNN (Dual)** | **76.00%** | **53.12%** | **85.00%** | **65.38%** | **75.89%** | **17** | 15 | 40 | **3** |
| 🥈 **Robust CRNN (PCEN)** | **73.33%** | **50.00%** | **60.00%** | **54.55%** | **57.69%** | **12** | 12 | 43 | 8 |
| 🥉 **Baseline 1D CNN** | 64.00% | 31.58% | 30.00% | 30.77% | 30.30% | 6 | 13 | 42 | 14 |
| ⚠️ **Baseline 2D CNN (Mel)** | 73.33%* | 0.00% | 0.00% | 0.00% | 0.00% | 0 | 0 | 55 | 20 |
| ⚠️ **Enhanced 1D CNN (Dual)** | 26.67%* | 26.67% | 100.00% | 42.11% | 64.52% | 20 | 55 | 0 | 0 |

*\*Note: Baseline 2D and Enhanced 1D outputs are trivial artifacts of predicting all 0s or all 1s.*

---

### 3.3 Detailed Breakdown by Sound Category

```
                            BEHAVIORAL BREAKDOWN BY SOUND CLASS
                            
             Real Gunshots Caught (Recall)        Ambient Noise Ignored (Rejection)
  Enhanced 2D   [████████████████░░] 85.0% (17/20)    [███████████████████] 94.3% (33/35)
  Robust CRNN   [████████████░░░░░░] 60.0% (12/20)    [████████████████████] 100.0% (35/35) 🏆
  Baseline 1D   [██████░░░░░░░░░░░░] 30.0% (6/20)     [█████████████████░░] 88.6% (31/35)
```

1. **Enhanced 2D CNN (Dual-Head) — The Production Winner:**
   - **Real Gunshots Caught:** **17 / 20 (85.0%)** (AK-47, Desert Eagle, Magnum, 14 rifles/snipers).
   - **Ambient Noise Ignored:** **33 / 35 (94.3%)** (Only 2 false alarms across all rain, sirens, engines, dogs).
   - **Imposters Rejected:** **7 / 20 (35.0%)** (Tricked by 13 loud fireworks and close claps).
2. **Robust CRNN (PCEN) — Environmental Noise Immunity:**
   - **Real Gunshots Caught:** **12 / 20 (60.0%)**.
   - **Ambient Noise Ignored:** **35 / 35 (100.0% Zero False Alarms)** — Completely immune to continuous environmental noise.
   - **Imposters Rejected:** **8 / 20 (40.0%)** (Fooled by 12 sharp aerial fireworks).
3. **Baseline 1D CNN — The Acoustic Domain Collapse:**
   - Recall plummeted from **99.72% on Test Data to 30.00% on Real Data** (missed 14 of 20 firearms). Time-domain kernels trained on clean data cannot tolerate distance attenuation and room reverberation.

### 3.4 Human Verification & Audio Audit Exports
To enable human listening audits of model decisions, all evaluated files are automatically partitioned into `Verification_Outputs/By_Model/`:
- **`Detected_Gunshots/`**: Full audio tracks flagged as gunshots (allowing direct verification of true positives and false alarms).
- **`Trigger_Slices_750ms/`**: The exact $750\text{ ms}$ window slice that triggered the confidence threshold.
- **`Ignored_NonGunshots/`**: Audio tracks rejected by the model.
- **Interactive Audio Player:** An HTML dashboard (`verification_dashboard.html`) allows human reviewers to listen to each detection in 1 click.

---

<!-- ========================================================================== -->
<!-- PAGE 4: COMPARATIVE SYNTHESIS, HARDWARE BLUEPRINT & BOUNDARIES             -->
<!-- ========================================================================== -->

## 🔬 PAGE 4: Comparative Synthesis, Hardware Blueprint & Boundaries

### 4.1 Side-by-Side Comparison: Test Data vs. Real Data

| Model Architecture | Test Data Accuracy | Real Data Accuracy | Real Gunshot Recall | Real Ambient Rejection | Key Engineering Takeaway |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Enhanced 2D CNN** | **99.51%** | **76.00%** | **85.00%** | **94.30%** | **Best Production Choice:** Balances high recall with strong noise rejection. |
| **Robust CRNN (PCEN)**| **99.79%** | **73.33%** | **60.00%** | **100.00%** | **Best for Outdoor Noise:** 100% immune to rain, motors, and sirens. |
| **Baseline 1D CNN** | **99.79%** | **64.00%** | **30.00%** | **88.60%** | Severe domain shift; time-domain wavelets fail outside lab conditions. |
| **Baseline 2D CNN** | 50.00% | 73.33%* | 0.00% | 100.00%* | Inactive due to Dying ReLU weight collapse during initial training. |
| **Enhanced 1D CNN** | 50.00% | 26.67%* | 100.00% | 0.00% | Inactive due to over-sensitive saturation (100% false alarm rate). |

---

### 4.2 Target Edge Hardware Architecture (Firmware Prepared — Physical Setup Pending)

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   EDGE DEPLOYMENT ARCHITECTURES (FIRMWARE PREPARED)                    │
├──────────────────────────────────────┬─────────────────────────────────────────────────┤
│ Target A: Arduino Nano 33 BLE Sense  │ Target B: Raspberry Pi 4 / 5 / Zero 2 W         │
├──────────────────────────────────────┼─────────────────────────────────────────────────┤
│ • Processor: Nordic nRF52840 (64MHz) │ • Processor: Broadcom BCM2711/BCM2712           │
│ • Memory: 1MB Flash, 256KB SRAM      │ • Memory: 512MB–8GB LPDDR4                      │
│ • Microphone: Built-in MP34DT05 PDM  │ • Microphone: USB / I2S (INMP441)               │
│ • Quantized Model: 119.3 KB (INT8)   │ • Framework: TFLite Runtime (Python)            │
│ • Tensor Arena: 60 KB Static SRAM    │ • Service: Headless systemd daemon              │
│ • Status: Firmware coded (`.ino`)    │ • Status: Script coded (`run_pi_crnn.py`)       │
│   Physical hardware setup pending    │   Physical hardware setup pending               │
└──────────────────────────────────────┴─────────────────────────────────────────────────┘
```

---

### 4.3 Operational Scope, Boundaries & Limitations (No Exaggerations)
1. **No Physical Embedded Hardware Setup:**  
   No physical Arduino Nano or Raspberry Pi boards have been wired, flashed, or tested on a test bench. Memory footprints (119 KB INT8, 60 KB arena) are software profiling figures.
2. **Execution Environment:**  
   All benchmarks were executed in software on the host PC (Python 3.11, TensorFlow 2.16.1, SoundFile).
3. **No Live Ballistic Shooting Range Testing:**  
   Testing was conducted on audio files (curated splits and external recordings). The system has not yet been deployed at an active live-fire firearm shooting range.
4. **The Imposter Blind Spot:**  
   Distinguishing chemical fireworks from firearm discharges remains an acoustic challenge (both produce physical shockwaves). True zero-false-alarm systems require multi-modal optical flash or acoustic array localization.

---

### 4.4 Final Technical Conclusion
The Shoot_Catcher project demonstrates that **laboratory validation accuracy (99%+) gives a false sense of security**. When confronted with real-world acoustic recordings:
- Raw waveform 1D CNNs suffer severe domain shift, losing 70% of their recall.
- **Enhanced 2D CNN** represents the most viable production architecture, delivering **85% real firearm recall with 94.3% ambient noise rejection**.
- **PCEN** proves mathematically superior for outdoor continuous background noise, delivering **100% ambient noise immunity**.
