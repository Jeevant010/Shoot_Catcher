# 1D CNN Gunshot Detector — Complete Manual

> **Who is this for?** Anyone who wants to understand what a 1D CNN is, why it works for audio, and how this notebook uses it to detect gunshots from raw waveforms.

---

## Table of Contents

1. [What is a CNN?](#1-what-is-a-cnn)
2. [What is a 1D Convolution?](#2-what-is-a-1d-convolution)
3. [Why 1D CNN for Audio?](#3-why-1d-cnn-for-audio)
4. [The Architecture Explained](#4-the-architecture-explained)
5. [Data Pipeline](#5-data-pipeline)
6. [Class Weights & Imbalanced Data](#6-class-weights--imbalanced-data)
7. [GroupKFold — Preventing Data Leakage](#7-groupkfold--preventing-data-leakage)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [When 1D CNN Wins vs Loses](#9-when-1d-cnn-wins-vs-loses)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. What is a CNN?

A **Convolutional Neural Network** is a type of neural network designed to find **patterns** in data by sliding small "filters" (also called "kernels") across the input.

### The Analogy
Imagine you're looking at a long receipt printed on a strip of paper. Instead of reading every character, you slide a magnifying glass across it. The magnifying glass is your **filter** — it looks at a small window at a time and tells you "this part looks like a number" or "this part looks like a letter."

A CNN does the same thing. It has many filters, each looking for a different pattern. One filter might detect "a sudden spike" (gunshot!), another might detect "a steady hum" (background noise).

### The Layers
| Layer | What It Does |
|-------|-------------|
| **Conv** (Convolutional) | Slides filters across input, producing a "feature map" of detected patterns |
| **BatchNorm** | Normalizes the feature map to stabilize training (like auto-leveling brightness in a photo) |
| **MaxPooling** | Shrinks the feature map by keeping only the strongest activations (compression) |
| **GlobalAveragePooling** | Collapses the entire feature map into a single number per filter |
| **Dense** | Standard neural network layer that combines all features into a final decision |
| **Dropout** | Randomly turns off neurons during training to prevent memorization |

---

## 2. What is a 1D Convolution?

A **1D Convolution** slides a filter along ONE axis — time. This is perfect for audio because audio is a 1D signal: a sequence of numbers over time.

### The Math (Simplified)

Your audio clip is 5,512 numbers long. A filter of size 80 looks at 80 consecutive numbers at a time:

```
Audio:   [0.1, 0.3, -0.2, 0.8, 0.9, -0.1, ...]  (5,512 values)
Filter:  [w1, w2, w3, ..., w80]                    (80 learnable weights)

Output at position 0: sum(audio[0:80] × filter)
Output at position 1: sum(audio[1:81] × filter)
...and so on
```

The filter LEARNS what pattern to look for. After training, one filter might have learned to detect the explosive transient of a gunshot (a sudden jump from 0 to 1.0), while another might detect the echo that follows.

### Why Kernel Size 80?

At 22,050 Hz sample rate, 80 samples = **3.6 milliseconds**. A gunshot's initial explosive transient lasts about 2-5ms. So a kernel of 80 is perfectly sized to capture that "bang."

---

## 3. Why 1D CNN for Audio?

| Advantage | Explanation |
|-----------|-------------|
| **Zero feature engineering** | No need to compute spectrograms or MFCCs — the model learns directly from raw audio |
| **End-to-end learning** | The network discovers its own "best features" during training |
| **Lightweight** | Fewer parameters than a 2D CNN working on spectrograms |
| **Fast inference** | Less computation per prediction |
| **Great baseline** | If the 1D CNN fails, it means your DATA has a problem, not your model |

### The Tradeoff
1D CNNs typically achieve **lower accuracy** than 2D CNNs on spectrograms because:
- Spectrograms explicitly expose frequency information (which pitch is loud at which time)
- The 1D CNN must learn this frequency decomposition from scratch, which requires more data

---

## 4. The Architecture Explained

Our model has 4 convolutional layers, chosen based on 2024-2025 research on environmental sound classification:

```
Input: (5512, 1) — 250ms mono audio at 22kHz
        │
        ▼
[Conv1D: 32 filters, kernel=80, stride=4] ← "Wide lens" — captures 3.6ms transients
[BatchNorm]
[MaxPool: 4]
        │  Shape: (345, 32)
        ▼
[Conv1D: 64 filters, kernel=3]            ← "Zoom in" — refines local patterns
[BatchNorm]
[MaxPool: 4]
        │  Shape: (86, 64)
        ▼
[Conv1D: 128 filters, kernel=3]           ← Higher-level pattern combinations
[BatchNorm]
        │  Shape: (86, 128)
        ▼
[Conv1D: 128 filters, kernel=3]           ← Further refinement
[BatchNorm]
        │  Shape: (86, 128)
        ▼
[GlobalAveragePooling1D]                  ← Compress to single vector
        │  Shape: (128,)
        ▼
[Dense: 64, ReLU]                         ← Classification head
[Dropout: 0.4]
        │  Shape: (64,)
        ▼
[Dense: 1, Sigmoid]                       ← Output probability: 0.0=noise, 1.0=gunshot
```

### Why This Architecture?
- **Layer 1 wide kernel (80)**: Research shows initial layers need large receptive fields for audio (captures the gunshot transient pulse)
- **Subsequent small kernels (3)**: Once raw temporal features are extracted, small kernels find refined patterns
- **BatchNorm everywhere**: Stabilizes training (prevents gradient explosion/vanishing)
- **GlobalAveragePooling**: Better generalization than Flatten (proven in 2024 ablation studies)
- **4 layers, not 2 or 8**: Research shows 4-6 layers is optimal for short clips — more layers overfit, fewer underfit

---

## 5. Data Pipeline

### The Shuffler
Before any processing, the notebook shuffles all files randomly. This prevents any ordering bias (e.g., all gunshots being loaded first).

### Flexible Data Loader
The notebook auto-detects your folder structure. It works with:
- `class_0_nongunshot/` and `class_1_gunshot/` (your current structure)
- Any folders containing "gunshot", "gun", "noise", "background", "nongunshot" in the name

### Sample Rate Support
Set `SAMPLE_RATE = 22050` for full quality or `SAMPLE_RATE = 16000` for Arduino-compatible processing. The notebook resamples automatically.

---

## 6. Class Weights & Imbalanced Data

If you have 11,000 gunshots and 8,950 non-gunshots, the model might get lazy and just predict the larger class. We fix this with **class weights**.

```python
class_weight = compute_class_weight('balanced', classes=[0, 1], y=y_train)
```

This tells the model: "If you miss a gunshot, I'll penalize you MORE than if you miss a background noise." The penalty is calculated automatically based on your actual data ratio — no hardcoded numbers.

---

## 7. GroupKFold — Preventing Data Leakage

### The Problem
If `SA_004A_S01.wav` was trimmed into 5 clips, those clips share the same background noise. If clips 1-4 go to training and clip 5 goes to testing, the model memorizes the background noise and gets a fake 99% accuracy.

### The Fix
We extract the source recording name from each filename and group clips together. ALL clips from the same source go to EITHER training OR testing — never split.

```
Source: SA_004A_S01.wav
  ├── clip0_clean.wav    → ALL go to TRAIN
  ├── clip1_clean.wav    → ALL go to TRAIN
  └── clip2_clean.wav    → ALL go to TRAIN

Source: SA_084B_S06.wav
  ├── clip0_clean.wav    → ALL go to TEST
  └── clip1_clean.wav    → ALL go to TEST
```

---

## 8. Evaluation Metrics

### Why NOT Accuracy?
If 80% of your data is non-gunshots, a model that ALWAYS predicts "non-gunshot" gets 80% accuracy. It's completely useless but looks great on paper. This is called the **Accuracy Paradox**.

### What We Track Instead

| Metric | What It Means | Why It Matters |
|--------|--------------|----------------|
| **Recall** | "Of all real gunshots, how many did I catch?" | Missing a gunshot = dangerous |
| **Precision** | "Of all my 'gunshot' predictions, how many were correct?" | False alarms waste resources |
| **F2-Score** | Weighted average favoring recall (2:1 over precision) | Best single metric for safety applications |
| **PR-AUC** | Area under Precision-Recall curve | Better than ROC-AUC for imbalanced data |
| **Confusion Matrix** | Grid showing exact counts of correct/incorrect predictions | See exactly WHERE the model fails |

---

## 9. When 1D CNN Wins vs Loses

| Scenario | 1D CNN Performance |
|----------|-------------------|
| Clean, high-SNR gunshot recordings | ✅ Excellent |
| Ultra-constrained edge devices (<10KB model) | ✅ Best choice |
| Fastest possible inference | ✅ Winner |
| Noisy urban environments | ⚠️ Decent but 2D CNN is better |
| Distinguishing gunshot from similar impulses (clap, slam) | ❌ Struggles — needs spectral features |
| Limited training data (<1,000 samples) | ❌ Needs more data than 2D CNN |

### Rule of Thumb
If the 1D CNN gets **>80% recall** on your test set, your data pipeline is solid. Move to the 2D CNN for the final production model.

---

## 10. Troubleshooting

### "GPU not detected"
TensorFlow needs CUDA-compatible GPU. For CPU-only training, it will just be slower (~5-10x).

### Training takes too long
- Reduce `EPOCHS` to 20 for a quick test
- Reduce data: set `USE_ONLY_CLEAN = True` and remove augmentation temporarily

### Accuracy is ~50% (random)
- Your data might not be properly labeled. Listen to random samples from each class.
- The sample rate might be wrong. Check that your WAV files actually match `SAMPLE_RATE`.

### Very high train accuracy but low test accuracy
- Data leakage! Make sure GroupKFold is working. Check that `all_groups` has diverse values.
- Try increasing Dropout from 0.4 to 0.5.

### Model predicts only one class
- Class weights might not be applied. Check the training output for `class_weight` usage.
- Learning rate might be too high. Try `LEARNING_RATE = 0.0001`.

---

> **Remember**: The 1D CNN is your sanity check and baseline. Compare its results with the 2D CNN (in folder `02_2D_CNN/`) to see how much feature extraction (Mel Spectrograms) improves performance.
