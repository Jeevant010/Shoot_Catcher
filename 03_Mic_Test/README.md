# 🎤 03 — Microphone Test & Live 5-Model Intelligence Hub

Tests your microphone and runs real-time live gunshot detection or file benchmarks across all 5 trained models:
1. **Baseline 1D CNN** (`01_1D_CNN`)
2. **Baseline 2D CNN (Mel Spectrogram)** (`02_2D_CNN`)
3. **Robust CRNN (PCEN)** (`04_Robust_CRNN_PCEN`)
4. **Enhanced 1D CNN Dual-Head** (`Enhanced_Models/01_Enhanced_1D_CNN`)
5. **Enhanced 2D CNN Dual-Head** (`Enhanced_Models/02_Enhanced_2D_CNN`)

---

## Quick Start (Scripts & Live Dashboard)

Run the unified multi-model intelligence hub:
```bash
python scripts/live_demo.py
```

### Modes Available in `live_demo.py`:
- **`[1] 🚀 Multi-Model Live Real-Time Dashboard`**: Runs ALL 5 models in parallel on live microphone input. Displays side-by-side confidence bars and logs alerts.
- **`[2] 🎯 Single-Model Live Stream`**: Focus on 1 specific model for low-latency live monitoring.
- **`[3] 🎵 Benchmark on Audio File (.wav)`**: Evaluates any `.wav` recording across all 5 models side-by-side.
- **`[4] 🎙️ Quick Record 5s & Benchmark`**: Records 5s audio on the spot and runs all 5 models on it.
- **`[5] ⚡ Sensitivity Presets`**: Switch between High Sensitivity (phone testing), Standard Mode, and Ultra High.
- **`[6] 📋 Model Status Audit`**: Displays an audit table showing status (Trained/Untrained) for all 5 modules.

---

## Other Script Helpers in `scripts/`

- **`scripts/test_on_recording.py`**:
  ```bash
  # Interactive file picker or specify file directly:
  python scripts/test_on_recording.py --input path/to/gunshot.wav --threshold 0.50
  ```
- **`scripts/record_test.py`**:
  ```bash
  # Test your microphone and verify signal levels:
  python scripts/record_test.py
  ```

---

## Graceful Untrained Model Handling

If any model has not been trained yet:
- It is tagged as `⚠️ NOT TRAINED / FILE MISSING` in the audit table.
- The hub safely runs the remaining active models without crashing or errors.
- As soon as the `.h5` file is saved, it is automatically discovered and activated.
