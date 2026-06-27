# 🧠 Enhanced Models — Robust Gunshot Detection

> **Date**: 2026-06-27  
> **Purpose**: Production-grade model training notebooks with advanced augmentation, dual-head anomaly detection, and paper-backed best practices.

---

## Models

| # | Model | Architecture | Key Features |
|---|-------|-------------|--------------|
| **01** | [Enhanced 1D CNN](01_Enhanced_1D_CNN/) | Raw waveform → Conv1D → Dual-head output | MixUp, SpecAugment, pitch/speed perturbation, anomaly head |
| **02** | [Enhanced 2D CNN](02_Enhanced_2D_CNN/) | Mel-spectrogram → Conv2D → Dual-head output | Same augmentations applied pre-spectrogram, frequency/time masking |

---

## What's New vs. Original Models

### Augmentation Enhancements (Both Models)

| Technique | What It Does | Why It Matters | Paper Reference |
|-----------|-------------|----------------|-----------------|
| **MixUp** | Blends two audio clips at random ratios (α=0.2-0.4) | Prevents overfitting to specific gunshot signatures; smooths decision boundaries | Zhang et al. (2018) "mixup: Beyond Empirical Risk Minimization" |
| **SpecAugment (Time Masking)** | Randomly zeros out time segments | Simulates the "guillotine effect" where a gunshot gets split across buffer boundaries | Park et al. (2019) "SpecAugment: A Simple Data Augmentation Method for ASR" |
| **Pitch Shifting** | Shifts pitch by ±2 semitones | Different firearms produce different frequency signatures | Salamon & Bello (2017) "Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification" |
| **Speed Perturbation** | Speeds up/slows down by ±10% | Accounts for variable reverb decay times and recording distances | Ko et al. (2015) "Audio Augmentation for Speech Recognition" |

### Dual-Head Architecture

Instead of a single sigmoid output, the model has two output heads:

```
Input → [Shared CNN Backbone] → Head 1: Gunshot Probability (sigmoid)
                               → Head 2: Anomaly Score (sigmoid)
```

- **Head 1 (Gunshot):** Traditional binary classification. "Is this a gunshot?"
- **Head 2 (Anomaly):** Trained on reconstruction error. "Is this sound unusual/out-of-distribution?"

This catches edge cases where a sound is neither clearly gunshot nor clearly background (e.g., firecrackers, car backfires).

---

## Design Goal: Near-Zero False Negatives

Our priority hierarchy:
1. **False Negatives → NEAR ZERO** (if a gunshot happens, we MUST detect it)
2. **False Positives → Acceptable** (some false alarms are OK, but not constantly)

This is achieved through:
- **High recall training** via class weights and F2-score optimization
- **Low confidence threshold** (0.5 default, tunable down to 0.3 for maximum recall)
- **Anomaly head** catches sounds the main classifier is uncertain about

---

## Paper References & Prior Work

### Augmentation Papers
1. **Zhang et al. (2018)** — *"mixup: Beyond Empirical Risk Minimization"*, ICLR. The foundational MixUp paper.
2. **Park et al. (2019)** — *"SpecAugment: A Simple Data Augmentation Method for ASR"*, Interspeech. Google's SpecAugment.
3. **Salamon & Bello (2017)** — *"Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification"*, IEEE Signal Processing Letters.
4. **Ko et al. (2015)** — *"Audio Augmentation for Speech Recognition"*, Interspeech. Speed perturbation for robustness.

### Gunshot Detection Papers
5. **Saha et al. (2025)** — *"Comparative Analysis of Deep Learning Architectures and Data Augmentation Strategies for Automated Gunshot Detection in Forest Environments"*
6. **Magee et al. (2019)** — *"Low Cost Gunshot Detection using Deep Learning on the Raspberry Pi"*, IEEE BigData.
7. **arXiv (2026)** — *"Exploring Feature Extraction Technique Parameters for Acoustic Gunshot Classification"*

### GitHub Repositories
- [`gabemagee/gunshot_detection`](https://github.com/gabemagee/gunshot_detection) — 1D+2D CNN ensemble on RPi, IEEE-published
- [`mariamkhmahran/gunshot-detection-system`](https://github.com/mariamkhmahran/gunshot-detection-system) — Urban Mel-spectrogram 2D CNN
- [`hasnainnaeem/Gunshot-Detection-in-Audio`](https://github.com/hasnainnaeem/Gunshot-Detection-in-Audio) — TF 2.0 binary classifier

### Arduino Nano 33 BLE — Deployment Proof
- **Edge Impulse Public Project**: [`Gunshot Detection (ID: 133765)`](https://studio.edgeimpulse.com/public/133765/latest)
- **Edge Impulse Blog**: *"Go Ahead, Give AI a Shot"* — Full pipeline demo
- **Hardware**: Arduino Nano 33 BLE Sense (ARM Cortex-M4, 1MB Flash, 256KB RAM, MP34DT05 mic)

> **Reality Check:** Successful Arduino deployments use Edge Impulse's auto-optimized INT8 models (~10-30KB). Our enhanced models are designed for **PC/RPi training and evaluation first**, then can be quantized for edge deployment.

---

## Folder Structure

```
Enhanced_Models/
├── README.md                          ← You are here
├── 01_Enhanced_1D_CNN/
│   ├── enhanced_1d_cnn.ipynb          ← Training notebook
│   ├── manual/
│   │   └── README.md                  ← Detailed documentation
│   └── output/                        ← Trained models, plots, metrics
├── 02_Enhanced_2D_CNN/
│   ├── enhanced_2d_cnn.ipynb          ← Training notebook
│   ├── manual/
│   │   └── README.md                  ← Detailed documentation
│   └── output/                        ← Trained models, plots, metrics
```
