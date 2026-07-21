# 🎤 03 — Microphone Test & Live Multi-Model Intelligence Hub

Tests your microphone and runs real-time live gunshot detection or file benchmarks across all 4 trained/untrained model modules:
1. **Baseline 1D CNN** (`01_1D_CNN`)
2. **Baseline 2D CNN** (`02_2D_CNN`)
3. **Enhanced 1D CNN Dual-Head** (`Enhanced_Models/01_Enhanced_1D_CNN`)
4. **Enhanced 2D CNN** (`Enhanced_Models/02_Enhanced_2D_CNN`)

---

## Quick Start (Scripts & Live Dashboard)

Run the unified multi-model intelligence hub:
```bash
python scripts/live_demo.py
```

### Modes Available in `live_demo.py`:
- **`[1] 🚀 Multi-Model Live Real-Time Dashboard`**: Runs ALL trained models in parallel on live microphone input. Displays side-by-side confidence progress bars and logs alerts.
- **`[2] 🎯 Single-Model Live Stream`**: Focus on 1 specific model for low-latency live monitoring.
- **`[3] 🎵 Multi-Model File Benchmark`**: Evaluates any `.wav` recording across all active models side-by-side.
- **`[4] 🎙️ Quick Record 5s & Benchmark`**: Records 5s audio on the spot and runs all models on it.
- **`[5] 📋 Model Status Audit`**: Displays an audit table showing status (Trained/Untrained) for all 4 modules.

---

## Jupyter Notebook Mode

Prefer working in Jupyter?
1. Open `mic_test_and_inference.ipynb`
2. Update `MODEL_PATH` in Cell 2
3. Run cells in order to test recording, measure noise floor, and run inference.

---

## Graceful Untrained Model Handling

If a model has not been trained yet (e.g. `Enhanced_Models/02_Enhanced_2D_CNN`), the system:
- Tags it as `⚠️ NOT TRAINED / FILE MISSING` in the audit table.
- Safely skips it during active inference without crashing, throwing errors, or interrupting other active models.
- Auto-discovers it as soon as the `.h5` file is created after training!

---

## Common Fixes

If microphone doesn't work:
1. Open **Settings → Privacy & Security → Microphone**
2. Turn ON "Microphone access"
3. Turn ON "Let desktop apps access your microphone"
4. Restart your terminal / Jupyter kernel

See [manual/README.md](manual/README.md) for full troubleshooting.
