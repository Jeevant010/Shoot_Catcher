# 📖 Run Guide — Shoot_Catcher CNN Models

Step-by-step instructions to train and test gunshot detection models.

---

## Prerequisites

### Python Packages
All notebooks auto-install dependencies via `%pip install`. No manual setup needed.

### Required:
- Python 3.8+
- Jupyter Notebook or VS Code with Jupyter extension
- Your trimmed audio data folder (from Data-Cleaner → Trimmer)

### Optional (for GPU training):
- NVIDIA GPU with CUDA support
- TensorFlow GPU package

---

## Step 1: Prepare Your Data

Your data folder must look like:
```
your_data_folder/
├── class_0_nongunshot/
│   ├── *.wav files (background, street, car sounds)
└── class_1_gunshot/
    ├── *.wav files (gunshot clips)
```

The notebooks auto-detect the folder names. They handle:
- Clean files only (filtering out `_aug_` variants)
- Augmented files added to training only (Option C)
- Shuffling before splitting
- GroupKFold by source recording

---

## Step 2: Train 1D CNN (Baseline)

1. Open `01_1D_CNN/1d_cnn_gunshot_detector.ipynb`
2. In **Cell 3**, set `DATA_DIR` to your data folder path
3. Choose `SAMPLE_RATE`:
   - `22050` for best quality
   - `16000` for Arduino-compatible
4. Run all cells (Cell 1 through Cell 11)
5. Check results:
   - **Cell 9**: Confusion matrix, PR-AUC, F2-score
   - **Cell 10**: Listen to misclassified samples
   - **Cell 11**: Model exports (`.h5` + `.tflite`)

**Expected time**: 10-30 minutes (depending on dataset size and GPU)

### What to look for:
- **Good**: F2-score > 0.80, ROC-AUC > 0.85
- **Warning**: If accuracy ≈ 50%, data might be mislabeled
- **Warning**: If train accuracy >> test accuracy, possible data leakage (check GroupKFold)

---

## Step 3: Train 2D CNN (Production)

1. Open `02_2D_CNN/2d_cnn_gunshot_detector.ipynb`
2. In **Cell 3**, set `DATA_DIR` (same folder)
3. Choose `FEATURE_TYPE`:
   - `mel_spectrogram` for max accuracy (recommended first)
   - `mfcc` for smallest model (try after mel spectrogram)
4. Run all cells (Cell 1 through Cell 11)
5. Check results:
   - **Cell 9**: Includes head-to-head comparison with 1D CNN!
   - **Cell 10**: View misclassified spectrograms
   - **Cell 11**: Model exports

**Expected time**: 15-45 minutes

### What to look for:
- 2D CNN should **beat** 1D CNN on all metrics
- If it doesn't → try different feature parameters (increase `N_MELS`, decrease `HOP_LENGTH`)
- The comparison table in Cell 9 tells you exactly which model wins

---

## Step 4: Run with Both Feature Types

After running with `mel_spectrogram`, re-run the 2D CNN notebook with `FEATURE_TYPE = 'mfcc'` to compare.

This gives you three models to choose from:
1. 1D CNN (smallest, fastest, lowest accuracy)
2. 2D CNN + MFCC (medium size, good accuracy, fits on Arduino)
3. 2D CNN + Mel Spectrogram (largest, best accuracy)

---

## Step 5: Test Microphone

1. Open `03_Mic_Test/mic_test_and_inference.ipynb`
2. **First time only**: If Cell 3 shows "No input devices":
   - Open **Settings → Privacy → Microphone**
   - Turn ON "Let desktop apps access your microphone"
   - Restart your notebook kernel
3. Update `MODEL_PATH` in Cell 2 to your best model
4. Run cells:
   - **Cell 3**: Diagnostic check
   - **Cell 4**: Record and playback test
   - **Cell 5**: Noise floor measurement
   - **Cell 7**: Single-shot prediction (record 250ms → classify)
   - **Cell 8**: Continuous monitoring with ring buffer

---

## Step 6: Experiment & Compare

### Try different settings:

| Experiment | What to Change | Why |
|-----------|---------------|-----|
| Higher mel bins | `N_MELS = 128` | More frequency detail |
| Finer time resolution | `HOP_LENGTH = 64` | More time steps |
| Disable augmentation | `USE_AUGMENTATION = False` | See augmentation's impact |
| All data (not clean-only) | `USE_ONLY_CLEAN = False` | Use all 34k+ files |
| Arduino sample rate | `SAMPLE_RATE = 16000` | Test deployment-ready |

---

## Step 7: Deploy (Future)

Once you have a model you're happy with:

1. **TFLite INT8** is already exported by the notebooks
2. Convert to **C array** for Arduino:
   ```bash
   xxd -i model.tflite > model_data.cc
   ```
3. Include in your Arduino sketch with TensorFlow Lite Micro
4. Use the Arduino Nano 33 BLE Sense's built-in PDM microphone

---

## Troubleshooting Quick Reference

| Problem | Solution |
|---------|----------|
| GPU not detected | Install `tensorflow-gpu` with matching CUDA |
| Training is slow | Reduce `BATCH_SIZE` or use `FEATURE_TYPE = 'mfcc'` |
| ~50% accuracy | Check data labeling, sample rate match |
| High train, low test | Data leakage — verify GroupKFold |
| Mic not working | Windows Privacy Settings → Enable desktop mic access |
| Model too large for Arduino | Use MFCC (13×44) instead of Mel Spec (64×44) |
