

## 🧭 Executive Overview & Research Timeline

The development of **Shoot_Catcher** represents a continuous 6 to 8 month applied research cycle. The system progressed from primitive amplitude thresholding scripts to classical machine learning pipelines, raw time-domain deep convolutional networks, and ultimately bioacoustically inspired PCEN-CRNN architectures. 

This document presents the factual, unvarnished technical record: documenting what was built, the exact physical and mathematical reasons early approaches failed, and how each subsequent architecture was designed to resolve specific real-world acoustic bottlenecks.

```
                           THE 5-PHASE RESEARCH JOURNEY
                           
 ┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
 │   Phase 1: Data-Cleaner │ ──> │ Phase 2: Classical ML   │ ──> │ Phase 3: Hardware Edges │
 │ • 10,000+ Raw Files     │     │ • RF, SVM, KNN, GBDT    │     │ • DC Voltage Drift      │
 │ • The 250ms Mistake     │     │ • 13–40 Static MFCCs    │     │ • Thermal ADC Hiss      │
 │ • Updated_Trimmer Gates │     │ • Loss of Shockwave Rise│     │ • Diaphragm Saturation  │
 └─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
              │                                                               │
              ▼                                                               ▼
 ┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
 │ Phase 4: Dual-Head CNNs │ ──> │ Phase 5: Bioacoustics   │ ──> │ Phase 6: Human Auditing  │
 │ • Baseline 1D (3.63ms)  │     │ • NYU DCASE Chainsaw Res│     │ • Unseen Real Audio (75)│
 │ • Dying ReLU Decibel Fix│     │ • Dynamic PCEN Filter   │     │ • 750ms Trigger Slices  │
 │ • MixUp + SpecAugment   │     │ • Bi-GRU Temporal Logic │     │ • Interactive HTML Player│
 └─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
```

---

## ⏱️ Master Chronological Evolution Table

| Phase & Timeline | Methodology / Approach | What We Used & Built | Problems Encountered (Why It Failed) | Engineering Fix & Lesson Learned | Outcome |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Phase 1**<br>*(Months 1–2)* | **Raw Data Collection & Trimming Pipeline**<br>`Data-Cleaner/` | • 4,138 raw firearm files<br>• 6,230 ambient audio files<br>• Early trimmers (`Old_Work`, `New_Work`, `The_real_Trimmer`) | 1. **250ms Window Mistake:** Cut off the $200\text{–}600\text{ ms}$ acoustic reverberation tail.<br>2. **Transient Slicing:** Fixed slicing split shockwaves in half.<br>3. **Contaminated Negatives:** Ambient clips contained hidden metallic clicks. | **`Updated_Trimmer` Suite:**<br>• Upgraded window to **$750\text{ ms}$ & $1000\text{ ms}$**.<br>• 3-way candidate onset detector (Spectral Flux, Raw Peak, 1D Envelope).<br>• 3 Quality Gates (Crest Factor, Prominence, Attack Ratio). | 📦 Clean, uncorrupted dataset split established |
| **Phase 2**<br>*(Month 3)* | **Classical Supervised ML Benchmark**<br>`audio_pipeline.py` | • Random Forest (100 trees)<br>• SVM (RBF Kernel)<br>• KNN, Logistic Regression, GBDT<br>• 13–40 static MFCCs + Spectral Centroid | 1. **Timing Got Erased:** Averaging numbers over time flattened out the fast 3ms gunshot blast.<br>2. **False Alarm Crisis:** Everyday sounds (keys jingling, squeaky brakes, dropped spoons) share the same pitch as gunshots, fooling the model. | Classical manual feature engineering fails on fast blast events. **Deep learning (CNN) with learnable filters is mandatory.** | ❌ Abandoned classical ML |
| **Phase 3**<br>*(Month 4)* | **Microphone Hardware Reality & Edge Conditioning** | • Low-cost MEMS (INMP441) & electrets<br>• High-SPL firearm field playback<br>• PC audio interface capture | 1. **DC Voltage Bias:** 3.3V/5V rails caused baseline center drift.<br>2. **Thermal ADC Hiss:** Peak normalization on silence amplified hiss by 500x.<br>3. **Diaphragm Saturation:** Real shots ($>120\text{ dB SPL}$) clipped waves flat. | **Hardware Preconditioning Stage:**<br>1. Sample mean subtraction ($y - \text{mean}(y)$) to strip DC bias.<br>2. **Energy-Gated Peak Normalization** (bypasses scaling if RMS $< 0.005$). | ⚡ Real-world hardware resilience unlocked |
| **Phase 4**<br>*(Months 5–6)* | **Edge Impulse TinyML & Dual-Head CNN Models**<br>`01_1D`, `02_2D`, `Enhanced_Models/` | • Baseline 1D CNN (kernel 80 = $3.63\text{ ms}$)<br>• Baseline 2D CNN ($64 \times 130$ Mel)<br>• Enhanced Dual-Head 1D & 2D CNNs<br>• MixUp ($\text{Beta}(0.3, 0.3)$) + SpecAugment | 1. **Baseline 1D Domain Collapse:** Failed on external mics ($99.7\% \to 30.0\%$ recall).<br>2. **Baseline 2D Dying ReLU:** Unnormalized decibels ($-80\text{ dB}$ to $0\text{ dB}$) killed ReLU gradients ($0.0\%$ recall).<br>3. **Enhanced 1D Saturation:** Heavy class weights ($C_1 \approx 12.5 \times C_0$) caused $100\%$ false alarms. | **Enhanced 2D CNN Breakthrough:**<br>• Normalized decibels: $S_{\text{norm}} = (S_{\text{dB}} + 80)/80 \in [0, 1]$.<br>• Dual-head loss: Head 1 (Gunshot) + Head 2 (Anomaly).<br>• **$85.0\%$ Real Recall, $94.3\%$ Ambient Rejection.** | 🏆 **Primary Production Champion** |
| **Phase 5**<br>*(Months 7–8)* | **Bioacoustics Adaptation: Robust PCEN-CRNN**<br>`04_Robust_CRNN_PCEN` | • Per-Channel Energy Norm (PCEN)<br>• Synthetic microphone RIR augmentation<br>• Conv2D + Bidirectional GRU + Attention<br>• 1:1 strict balanced evaluation | **Percussive Imposter Triggering:**<br>PCEN's dynamic transient-boosting filter amplifies rapid percussive hand claps into false alarms ($23/35$ triggers). | **Bioacoustics Chainsaw Solution:**<br>• Steady-state noise converges to $S_0^{0.02} \approx 1.0$, mathematically erasing rain/engines.<br>• Gain invariance ($g^{0.04} \approx 1.0$).<br>• **$100.0\%$ Zero False Alarms** on continuous ambient noise. | 🛡️ **Best Outdoor Noise Immunity** |
| **Phase 6**<br>*(Final)* | **Human Verification & Real Audio Audit**<br>`Verification_Outputs/` | • 75-clip external benchmark (Real Data)<br>• Live Freesound MP3 suite (`custom_sound/`)<br>• HTML interactive audio dashboard | **Physical Boundary of Single Mics:**<br>Chemical aerial fireworks produce explosive shockwaves that physically mimic muzzle blasts; single microphones cannot separate them without spatial triangulation. | **Automated Human Verification:**<br>• Automatically saves full recordings + exact $750\text{ ms}$ trigger slices.<br>• Generates interactive browser player for 1-click human verification audits. | ✅ **Complete Auditability & Transparency** |

---

## 🔬 Phase 1: Raw Data Collection & The Data-Cleaner Pipeline

### 1.1 Audio Sources & Volume
* **Academic Gunshot Dataset:** *"A multi-firearm, multi-orientation audio dataset of gunshots"* (*Data in Brief*, 2023; DOI: [10.1016/j.dib.2023.109091](https://doi.org/10.1016/j.dib.2023.109091)) — calibrated multi-caliber and multi-angle recordings.
* **Firearm Files:** **4,138 raw recordings** (handguns, rifles, shotguns across varied distances/environments).
* **Background Files:** **6,230 ambient recordings** (urban soundscapes, traffic, rain, speech, room acoustics).

### 1.2 The Early Trimmer Suite (`Data-Cleaner/research/`)
Raw field recordings ($10\text{s}$ to $60\text{min}$) were sliced into isolated events across three iterations:
* **`Old_Work/`:** Naive amplitude thresholding (fixed energy spike = event).
* **`New_Work/`:** OpenCV video muzzle flash detection paired with audio energy z-scores.
* **`The_real_Trimmer/`:** Automated audio-only extraction script.

### 1.3 Why Early Trimmers Failed
1. **Truncated 250ms Windows:** Cut off the $200\text{–}600\text{ ms}$ reverberation tail needed to separate gunshots from door slams and claps.
2. **Shockwave Slicing:** Fixed-interval cuts split blast attacks and decays across separate clips.
3. **Contaminated Negatives:** Background recordings contained hidden metallic clicks and impacts mislabeled as ambient noise.

### 1.4 The Engineering Fix: `Updated_Trimmer`
Upgraded window duration to **$750\text{ ms}$ / $1000\text{ ms}$** with a **3-way candidate onset detector**:
* **Spectral Flux:** Detects sudden frame energy jumps: $SF(t) = \sum_{k} H(|X(t, k)| - |X(t-1, k)|)$.
* **Raw Peak Detection:** Pins absolute maximum amplitude.
* **Smoothed 1D Envelope:** Rolling-average convolution ignores single-sample electrical spikes.

**Three Automated Quality Gates:**
* **Crest Factor ($\text{Peak} / \text{RMS}$):** Validates impulse sharpness.
* **Prominence ($\text{Peak} / \text{Median}$):** Confirms emergence above ambient floor.
* **Attack Ratio ($\text{Energy}_{\text{first\_half}} / \text{Energy}_{\text{second\_half}}$):** Rejects contaminated background clicks and non-decaying transients.

---

## 📊 Phase 2: The Supervised Machine Learning Benchmark (And Why It Failed)

Classical ML pipelines were systematically evaluated via `audio_pipeline.py` and `use_SupervisedModel.md`:
* **Models Tested:** Logistic Regression, KNN, Random Forest (100 estimators), GBDT, and SVM (RBF kernel).
* **Handcrafted Features:** 13–40 static MFCCs, Spectral Centroid, Spectral Rolloff, and Zero-Crossing Rate (ZCR).

### Why Classical Supervised Learning Failed
1. **Erased Temporal Dynamics:** Time-averaged MFCCs smoothed out the ultra-fast ($<3\text{ ms}$) shockwave rise-time, destroying the transient blast profile.
2. **False Alarm Crisis:** Everyday impulse sounds (cutlery, squeaky brakes, jingling keys) share the same static frequency spectra as gunshots, triggering rampant false positives.

> **Key Takeaway:** Handcrafted static feature extraction fails on explosive events. End-to-end **Convolutional Neural Networks (CNNs)** with learnable time-frequency filters are mandatory.

---

## ⚡ Phase 3: Hardware Reality & Microphone Edge Cases

Physical microphones deviate drastically from pristine training WAVs:

```
                            HARDWARE PRECONDITIONING STAGE
Raw Mic Signal ──────> [ DC-Offset Stripping ] ──────> [ Energy-Gated Peak Norm ] ──────> To Model
 (Bias + Hiss)              y = y - mean(y)               Bypass if RMS < 0.005           [-1.0, 1.0]
```

### Hardware Failure Modes & Fixes
1. **DC Voltage Bias:** 3.3V/5V rails on MEMS (INMP441) and electrets cause baseline drift away from 0.0.  
   * **Fix:** **DC Stripping** via mean subtraction: $y = y - \text{mean}(y)$.
2. **Thermal ADC Hiss Amplification:** In quiet rooms (peak $\approx 0.002$), standard peak-normalization amplifies low-level ADC hiss by **500x**, triggering false alarms.  
   * **Fix:** **Energy-Gated Peak Norm** (bypasses scaling if $\text{RMS} < 0.005$).
3. **Diaphragm Saturation:** High-SPL blasts ($>120\text{ dB SPL}$) clip cheap mic diaphragms flat into square waves.

---

## 🧩 Phase 4: Edge Impulse TinyML & Dual-Head CNN Models

TinyML exploration on Edge Impulse (targeting microcontrollers like the Arduino Nano 33 BLE Sense) revealed that pure binary classification is brittle, motivating a dual-head anomaly approach.

### 1. Baseline 1D CNN (`01_1D_CNN`)
* **Architecture:** Raw waveforms with an 80-sample first filter ($3.63\text{ ms}$ at $22.05\text{ kHz}$ matching gunshot N-waves); 92,513 parameters ($<119\text{ KB}$ in INT8).
* **Failure Mode:** Acoustic domain collapse on unseen microphones ($99.7\% \to 30.0\%$ recall).

### 2. Baseline 2D CNN (`02_2D_CNN`)
* **Architecture:** $64 \times 130$ Mel-spectrograms.
* **Failure Mode:** Unnormalized decibels ($-80\text{ dB}$ to $0\text{ dB}$) triggered Dying ReLU collapse (**$0.0\%$ recall**).

### 3. Enhanced Dual-Head Models (`Enhanced_Models/`)
* **Decibel Normalization Fix:** Scaled to $[0.0, 1.0]$ via $S_{\text{norm}} = (S_{\text{dB}} + 80)/80$, rescuing ReLU gradients.
* **Dual-Head Output:** Shared backbone branching into **Head 1 (Gunshot Probability)** and **Head 2 (Anomaly / Novelty Score)**.
* **Data Augmentation:** MixUp ($\text{Beta}(0.3, 0.3)$) + SpecAugment (time/frequency masking).
* **Result:** **Enhanced 2D CNN became Primary Production Champion** (**$99.51\%$ test accuracy, $85.0\%$ real-world recall, $94.3\%$ ambient rejection**).

---

## 🌲 Phase 5: The "Chainsaw" Bioacoustics Research & PCEN-CRNN

To solve **Acoustic Domain Shift** (performance collapse under rain, traffic, or microphone variations), the project adopted techniques from **Forest Acoustic & Illegal Chainsaw Detection Research** (NYU Bioacoustics / DCASE, Lostanlen et al.):

* **The Rainforest Chainsaw Problem:** Heavy continuous background noise (rain, insects, wind) drowns out chainsaws on static Log-Mel spectrograms, mirroring gunshots in noisy outdoor soundscapes.
* **PCEN (Per-Channel Energy Normalization):** Replaces static decibels with adaptive AGC against a causal background noise envelope:
  * *Stationary Noise Erasure:* Continuous noise converges under steady state to $S_0^{1 - 0.98} = S_0^{0.02} \approx 1.0$, mathematically flattening rain, fans, and engine hum.
  * *Hardware Invariance:* Compresses microphone gain scaling to $g^{0.04} \approx 1.0$.
* **CRNN (Conv2D + Bidirectional GRU):** Forward GRU tracks the $<3\text{ ms}$ shockwave attack; backward GRU verifies the exponential reverberation decay tail, rejecting sustained tones (sirens, horns).
* **Result:** **$100.0\%$ Zero False Alarms** on continuous ambient noise (rain, engines, sirens, dogs).

---

## 🎧 Phase 6: Independent Real-World Testing & Human Verification

* **Evaluation Scripts:** [`run_benchmark_external.py`](file:///c:/Users/aadit/Desktop/Shoot_Catcher/03_Mic_Test/scripts/run_benchmark_external.py) & [`test_on_recording.py`](file:///c:/Users/aadit/Desktop/Shoot_Catcher/03_Mic_Test/scripts/test_on_recording.py)
* **Outputs:** `Verification_Outputs/By_Model/` & `Verification_Outputs/verification_dashboard.html`

### Testing Workflow
Dedicated Python evaluation scripts process audio recordings by sliding a **$750\text{ ms}$ analysis window** across each file. For every window, each model evaluates the acoustic features and predicts a gunshot probability:
* **Detection ($\ge 50\%$ threshold):** The window is classified as a gunshot event.
* **Rejection ($< 50\%$ threshold):** The window is rejected as ambient background noise.

To eliminate dataset bias and data leakage, models were evaluated across **75 variable-length tracks** from external repositories and raw Freesound recordings:
1. **Unseen Gunshot Confirmation:** Both Robust CRNN ($100.0\%$) and Enhanced 2D CNN ($90.6\%$) produced instant detections on external community audio within $750\text{ ms}$.
2. **The Acoustic Imposter Boundary (Fireworks & Clapping):**
   * *Fireworks:* Chemical fireworks produce explosive shockwaves that physically mirror gunshot onsets, triggering false alarms during heavy mortar explosions ($>90\%$).
   * *Clapping:* Percussive hand clapping produces rapid transients that trigger PCEN's dynamic baseline tracker ($23/35$ triggers), whereas the Enhanced 2D CNN rejected $77.1\%$ of clapping windows.
   * *Conclusion:* Single-microphone systems cannot reliably separate loud mortar fireworks from gunfire without multi-sensor spatial triangulation.
3. **Automated Human Verification Pipeline:** An interactive web dashboard (`verification_dashboard.html`) and dedicated export folders save the exact $750\text{ ms}$ sound bite of every trigger for 1-click human listening verification.

---

## 📊 Comprehensive Model Comparison Matrix

| Model Name | Input Representation | Number of Parameters | Controlled Test Accuracy | Unseen Real Audio Recall | Ambient Noise Immunity | Architectural Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline 1D CNN** | Raw 1D Waveform | 92,513 | 99.79% | 30.00% | 88.57% | Ultra-low compute; suffers severe domain collapse on real mics |
| **Baseline 2D CNN** | Unnormalized Mel dB | 249,985 | 50.00% | 0.00% | 100.00%* | Inactive (Dying ReLU weight collapse from negative decibels) |
| **Enhanced 1D CNN** | Waveform + MixUp | ~110,000 | 50.00% | 100.00%* | 0.00% | Inactive (Over-penalized loss caused saturated recall collapse) |
| **Enhanced 2D CNN** | Normalized Mel $[0, 1]$ | 249,985 | **99.51%** | **85.00%** | **94.29%** | 🏆 **Primary Production Choice (Best Overall Generalizer)** |
| **Robust CRNN** | 2D PCEN + Bi-GRU | ~310,000 | **99.79%** | **60.00%** | **100.00%** | 🛡️ **Noise Immunity Choice (100% immune to rain/engines)** |

*\*Note: 0% and 100% reflect trivial constant outputs caused by numerical saturation or weight collapse.*

---

## 🎯 Key Architectural Takeaways for Future Deployment

1. **Clean Dataset Accuracy is a Vanity Metric:** Models achieving $>99\%$ in Jupyter notebooks can plunge to $30\%$ when evaluated on real-world recordings with microphone and distance variations.
2. **Frequency Representations Beat Waveforms:** While raw 1D waveforms have zero preprocessing overhead, 2D spectral representations (Mel and PCEN) are essential to survive physical microphone differences and room reverberation.
3. **Decibel Normalization is Life or Death:** Feeding raw decibels ($[-80, 0]$) into neural networks with ReLU activations triggers permanent neuron death. Shifting to $[0, 1]$ rescues the network.
4. **Window Geometry is Crucial:** A $750\text{ ms}$ window with $75\%$ overlap is the mathematical sweet spot to capture muzzle blasts plus acoustic decay tails while remaining feasible on edge microcontrollers.
5. **Physical Hardware Status:** All benchmarks were conducted in software emulation on local CPU. No physical edge hardware (Raspberry Pi, Arduino) has been wired or flashed yet; target firmware stands ready for future physical deployment.
