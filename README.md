# 🔫 Shoot_Catcher — CNN Gunshot Detection System

A complete gunshot detection pipeline using 1D and 2D Convolutional Neural Networks, designed for edge deployment on Arduino Nano 33 BLE Sense.

## Project Structure

```
Shoot_Catcher/
├── 01_1D_CNN/                         ← Raw waveform baseline model
│   ├── 1d_cnn_gunshot_detector.ipynb
│   ├── manual/README.md               ← Complete CNN theory
│   └── README.md
│
├── 02_2D_CNN/                         ← Mel Spectrogram production model
│   ├── 2d_cnn_gunshot_detector.ipynb
│   ├── manual/README.md               ← Spectrogram & 2D CNN theory
│   └── README.md
│
├── 03_Mic_Test/                       ← Live microphone testing
│   ├── mic_test_and_inference.ipynb
│   ├── manual/README.md               ← Mic setup & troubleshooting
│   └── README.md
│
├── Chat.md                            ← Research discussion log
├── README.md                          ← This file
└── RUN_GUIDE.md                       ← Step-by-step execution guide
```

## Quick Start

See [RUN_GUIDE.md](RUN_GUIDE.md) for step-by-step instructions.

## Models

| Model | Input | Expected Accuracy | Model Size (INT8) | Best For |
|-------|-------|------------------|-------------------|----------|
| **1D CNN** | Raw waveform (5,512 samples) | ~80% | ~20 KB | Baseline, ultra-constrained devices |
| **2D CNN (Mel)** | Mel Spectrogram (64×44) | ~93%+ | ~80-120 KB | Production model |
| **2D CNN (MFCC)** | MFCC (13×44) | ~88% | ~30-50 KB | Arduino deployment |

## Key Design Decisions

- **GroupKFold splitting**: Prevents data leakage from same source recording
- **Dynamic class weights**: Computed from actual data ratio, not hardcoded
- **PR-AUC & F2 metrics**: Better than accuracy for imbalanced detection tasks
- **SpecAugment**: Frequency + time masking for microphone robustness
- **Option C augmentation**: Clean files for splits, augmented for training boost
- **Both sample rates**: 22,050 Hz (quality) and 16,000 Hz (Arduino)

## Data Requirements

Point `DATA_DIR` in each notebook to a folder containing:
```
your_data_folder/
├── class_0_nongunshot/    (or any name with 'nongunshot', 'noise', 'background')
│   ├── file1.wav
│   └── ...
└── class_1_gunshot/       (or any name with 'gunshot', 'gun')
    ├── file1.wav
    └── ...
```

All WAV files should be 250ms, mono, 16-bit PCM.
