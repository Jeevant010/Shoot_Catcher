# 🚀 Shoot_Catcher — Step-by-Step Testing & Benchmark Execution Guide

This guide provides exact, reproducible instructions on how to evaluate all five Shoot_Catcher models on offline recorded audio data.

---

## 🛠️ 1. Environment & Prerequisites

### 1.1 Python Environment
All models require the `Shooter_model` Conda environment:
- **Python Executable:** `C:\Users\aadit\.conda\envs\Shooter_model\python.exe`
- **Conda Activation Command:**
  ```bash
  conda activate Shooter_model
  ```

### 1.2 Required Packages
The environment already includes all necessary packages. If running on a new machine, install via:
```bash
pip install tensorflow==2.16.1 soundfile scipy numpy librosa
```

---

## 📂 2. File & Folder Organization

```text
Shoot_Catcher/
├── Data/
│   └── SPLIT_DATASET_750MS/
│       └── test/                           ← Option B Dataset (1,442 pre-trimmed clips)
│           ├── class_0_nongunshot/         (721 files)
│           └── class_1_gunshot/            (721 files)
│
├── My_Test_Audio/                          ← Option A Dataset (75 full-length recordings)
│   ├── Actual_Gunshots/                    (20 firearm files: AK47, Magnum, Rifles)
│   ├── Like_Gunshots/                      (20 imposter files: Fireworks, Claps, Knocks)
│   └── Not_Gunshots/                       (35 ambient files: Rain, Engines, Sirens)
│
├── download_test_audio.py                  ← Script to re-download external test audio
│
└── 03_Mic_Test/scripts/
    ├── run_benchmark_dataset.py            ⭐ Permanent Option B Benchmark Runner
    ├── run_benchmark_external.py           ⭐ Permanent Option A Benchmark Runner
    ├── test_on_recording.py                🎵 Single-file sliding-window tester
    └── live_demo.py                        🎙️ Multi-model interactive dashboard
```

---

## ⚡ 3. How to Run the Benchmarks

### 🧪 Test 1: Run Held-Out Dataset Benchmark (Option B)
Evaluates all 5 models on the 1,442 pre-trimmed clips in `Data/SPLIT_DATASET_750MS/test/`.

**Command:**
```bash
& "C:\Users\aadit\.conda\envs\Shooter_model\python.exe" 03_Mic_Test/scripts/run_benchmark_dataset.py
```
*(Or if Conda environment is active: `python 03_Mic_Test/scripts/run_benchmark_dataset.py`)*

#### What the Script Does:
1. Scans and validates all 5 trained models in the workspace.
2. Loads each 750 ms `.wav` file from `class_1_gunshot` and `class_0_nongunshot`.
3. Performs model inference and outputs a formatted scorecard showing Accuracy, Precision, Recall, F1, F2, and Confusion Matrix (TP, FP, TN, FN).
4. Typical execution time: ~5 to 7 minutes on CPU for all 1,442 files across all 5 models.

---

### 🧪 Test 2: Run External Audio Benchmark (Option A)
Evaluates all 5 models on the variable-length audio tracks in `My_Test_Audio/`.

**Command:**
```bash
& "C:\Users\aadit\.conda\envs\Shooter_model\python.exe" 03_Mic_Test/scripts/run_benchmark_external.py
```
*(Or if Conda environment is active: `python 03_Mic_Test/scripts/run_benchmark_external.py`)*

#### What the Script Does:
1. Iterates through `Actual_Gunshots/`, `Like_Gunshots/`, and `Not_Gunshots/`.
2. Slides a 750 ms window with **75% overlap** across each audio file.
3. Records the maximum confidence score across all windows for each model.
4. Outputs:
   - **Overall Scorecard:** Accuracy, Precision, Recall, F1, F2, and Counts.
   - **Detailed Category Breakdown:** Gunshots caught, Imposters rejected, Ambient noise ignored.
5. Typical execution time: ~6 to 7 minutes on CPU for 75 tracks.

---

### 📥 Test 3: Download or Refresh External Audio Files
To download fresh audio files or restore the `My_Test_Audio/` directory:

**Command:**
```bash
& "C:\Users\aadit\.conda\envs\Shooter_model\python.exe" download_test_audio.py
```

#### What the Script Does:
- Downloads 20 real firearm recordings from the Audio Event Analysis database into `My_Test_Audio/Actual_Gunshots/`.
- Downloads 20 acoustic imposter files (fireworks, clapping, door knocks) from ESC-50 into `My_Test_Audio/Like_Gunshots/`.
- Downloads 35 everyday ambient files (rain, traffic, dogs, sirens) from ESC-50 into `My_Test_Audio/Not_Gunshots/`.
- Automatically skips any files that are already downloaded.

---

### 🎵 Test 4: Run a Quick Test on Any Single Audio File
To inspect how all 5 models react to any specific `.wav` file:

**Command:**
```bash
& "C:\Users\aadit\.conda\envs\Shooter_model\python.exe" 03_Mic_Test/scripts/test_on_recording.py --input path/to/your_sound.wav --threshold 0.50
```

#### What the Script Does:
- Displays a window-by-window analysis of the audio file.
- Generates amplified output clips for any detected events in `03_Mic_Test/scripts/test_output/`.
- Prints side-by-side model predictions and gives an overall verdict.

---

## ➕ 4. How to Add Custom Audio Files in the Future

You can drop your own recorded audio files directly into `My_Test_Audio/`:

1. **If it is a real firearm:**
   - Drop the file into `My_Test_Audio/Actual_Gunshots/`.
2. **If it is a loud/sudden sound (Clap, Slam, Firework, Balloon):**
   - Drop the file into `My_Test_Audio/Like_Gunshots/`.
3. **If it is background noise (Talking, Street, Music, Silence):**
   - Drop the file into `My_Test_Audio/Not_Gunshots/`.

### Recommended Audio Specifications:
- **Format:** Uncompressed `.wav` (16-bit PCM).
- **Channels:** Mono is preferred. *(Stereo files are automatically averaged to mono).*
- **Duration:** Any length from 1 second to 2 minutes. The runner automatically applies sliding windows.
- **Sample Rate:** Any rate (22.05 kHz, 44.1 kHz, 48 kHz). The runner resamples to 22.05 kHz automatically.

---

## 🔧 5. Troubleshooting & FAQs

### Q1: "ModuleNotFoundError: No module named 'soundfile'"
- **Cause:** Python ran outside the `Shooter_model` Conda environment.
- **Fix:** Ensure you prefix your command with `& "C:\Users\aadit\.conda\envs\Shooter_model\python.exe"` or activate the environment via `conda activate Shooter_model`.

### Q2: "Why does Baseline 2D CNN always output 0.00% Recall?"
- **Answer:** The weights in `02_2D_CNN/output/2d_cnn_mel_spectrogram_best.h5` suffered from the "Dying ReLU" issue during initial training. Use **Enhanced 2D CNN** (`Enhanced_Models/02_Enhanced_2D_CNN/output/enhanced_2d_cnn_best.h5`) instead, which contains the normalized decibel patch and scores 85% recall on external files.

### Q3: "Why does Enhanced 1D CNN flag everything as a Gunshot?"
- **Answer:** The model was trained with heavy class weighting prioritizing near-zero False Negatives, causing its decision boundary to saturate. In the future, its detection threshold can be raised from `0.50` to `0.95+` in code to reduce false alarms.

---

> [!TIP]
> **Quick Audit Check:**  
> You can verify the status of all model checkpoints at any time without running full benchmarks:
> ```bash
> & "C:\Users\aadit\.conda\envs\Shooter_model\python.exe" 03_Mic_Test/scripts/live_demo.py
> ```
> Choose option `[5]` from the menu to print the Model Status Audit Table.
