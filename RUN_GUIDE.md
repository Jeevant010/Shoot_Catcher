# 📖 Run Guide — Shoot_Catcher CNN Models

Step-by-step instructions to train, test, and run gunshot detection models.

---

## 📚 Complete System Manual

For detailed architectural details, model specs, file storage locations, and troubleshooting, see:
* [DOCUMENTATION.md](file:///d:/Desktop/GunShot/Shoot_Catcher/DOCUMENTATION.md)

---

## 📂 Where Output Files Are Saved

| Output Type | Storage Path | Description |
|-------------|--------------|-------------|
| 📝 **Detections Text Log** | `03_Mic_Test/scripts/detections_log.txt` | Timestamped log of triggered gunshot events across models |
| 🔊 **Triggered Audio Snippets** | `03_Mic_Test/scripts/recordings/` | `.wav` clips saved when a gunshot is detected |
| 🎵 **Benchmark Audio Window Clips** | `03_Mic_Test/scripts/test_output/` | `.wav` sliding window slices from file benchmarks |
| 📦 **Trained Model Files** | `01_1D_CNN/output/`, `02_2D_CNN/output/`, etc. | `.h5` model weights, `.tflite` edge models, and `norm_stats.json` |

---

## 🚀 Step 1: Run Multi-Model Live Dashboard & Testing Suite

Launch the main operational hub script:
```bash
python 03_Mic_Test/scripts/live_demo.py
```

### Main Menu Options:
- **`[1] 🚀 Multi-Model Live Real-Time Dashboard`**: Evaluates **ALL trained models simultaneously** on live microphone stream.
- **`[2] 🎯 Single-Model Live Microphone Stream`**: Low-latency live monitoring focused on 1 chosen model.
- **`[3] 🎵 Multi-Model Audio File Benchmark`**: Slide window across `.wav` recording file and compare all active models side-by-side.
- **`[4] 🎙️ Quick Record 5s & Benchmark`**: Record 5 seconds from your mic on the spot and run multi-model benchmark.
- **`[5] 📋 View Model Status Audit`**: Displays module status table (Trained / Untrained, input shapes, model types).

---

## 🎤 Step 2: Choose the Right Microphone Option

When prompted by `live_demo.py`:
* **To test real-world room sound (speaking, clapping, phone sound):** Choose option **`2`** (`Microphone Array - Intel Smart Sound`).
* **To test audio playing directly on your computer (YouTube / VLC):** Choose option **`7`** (`Stereo Mix`).

---

## 🏋️ Step 3: Train / Retrain Models (Notebooks)

To train or re-train any of the 4 model modules:
1. Open the module's notebook in Jupyter or VS Code:
   - Module 1: `01_1D_CNN/1d_cnn_gunshot_detector.ipynb`
   - Module 2: `02_2D_CNN/2d_cnn_gunshot_detector.ipynb`
   - Module 3: `Enhanced_Models/01_Enhanced_1D_CNN/enhanced_1d_cnn.ipynb`
   - Module 4: `Enhanced_Models/02_Enhanced_2D_CNN/enhanced_2d_cnn.ipynb`
2. Run all cells to generate trained model `.h5` files into `output/`.
3. On next launch of `live_demo.py`, the new model will be automatically discovered and loaded!
