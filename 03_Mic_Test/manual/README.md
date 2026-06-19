# Microphone Test & Inference — Manual

## Table of Contents

1. [Windows Microphone Setup](#1-windows-microphone-setup)
2. [The Ring Buffer](#2-the-ring-buffer)
3. [Domain Shift in Practice](#3-domain-shift-in-practice)
4. [Troubleshooting](#4-troubleshooting)

---

## 1. Windows Microphone Setup

### Step-by-Step: Enable Microphone for Python/Jupyter

1. **Open Settings**: Press `Win + I` → go to **Privacy & Security** → **Microphone**
2. **Enable Microphone Access**: Toggle **"Microphone access"** to **ON**
3. **Enable Desktop Apps**: Scroll down to **"Let desktop apps access your microphone"** → toggle **ON**
4. **Restart**: Close your terminal/VS Code/Jupyter → reopen → run the notebook

### Why This Happens
Windows blocks desktop applications (Python, Jupyter, VS Code terminal) from accessing the microphone by default. This is a privacy protection. When you enable "Let desktop apps access your microphone," you're telling Windows that it's okay for Python to use your mic.

### Verifying It Works
After enabling, run Cell 3 in the notebook. You should see:
```
✅ Microphone access GRANTED
```

### If You Have Multiple Python Installations
If you installed Python from the Microsoft Store AND have Anaconda/manual Python:
1. Open Settings → **Apps** → **Advanced app settings** → **App execution aliases**
2. Turn OFF the "Python" aliases from Microsoft Store
3. This prevents conflicts between Python installations

---

## 2. The Ring Buffer

### The Guillotine Effect
Imagine a gunshot happens right at the boundary between two 250ms recording buffers:

```
Buffer 1: [────────── noise ──────│BANG]     ← Gets only the tail-end of the gunshot
Buffer 2: [──────── noise ─────── │────]     ← Gets only quiet noise after

Both buffers might be classified as "non-gunshot" — the gunshot was "guillotined" in half!
```

### The Solution: Overlapping Windows

By overlapping windows at 50%, every event is fully captured in at least one window:

```
Window 1: [0ms ──────────── 250ms]
Window 2:      [125ms ──────────── 375ms]     ← Catches the gunshot fully!
Window 3:            [250ms ──────────── 500ms]

Even if the gunshot falls at t=250ms, Window 2 captures it completely.
```

### How It Works in the Notebook
- We record the full duration at once
- Then process it in overlapping chunks
- If ANY window exceeds the confidence threshold, we flag a detection
- The `detections.json` file logs the exact timestamp of each detection

---

## 3. Domain Shift in Practice

### What to Expect
Your model was trained on YouTube audio, Freesound clips, and other internet sources. Your laptop microphone has:
- **Different frequency response**: Built-in mics roll off above ~10 kHz
- **Different noise floor**: Laptop fans, room acoustics, electrical noise
- **Different gain**: Lower sensitivity than studio mics

### Expected Performance Gap
| Metric | Test Set (clean data) | Laptop Mic (real-world) |
|--------|----------------------|------------------------|
| Accuracy | 85-95% | 60-80% |
| False positives | Low | Higher (claps, slams) |

### How to Improve Real-World Performance
1. **Record test data with YOUR mic**: Create 50-100 clips of background noise, claps, and slams
2. **Add to training set**: Include laptop-mic recordings in your non-gunshot training data
3. **Fine-tune**: Re-train the model with the mixed dataset
4. **Adjust threshold**: Increase `CONFIDENCE_THRESHOLD` (e.g., 0.8 or 0.9) to reduce false positives

---

## 4. Troubleshooting

### "No input devices found"
- **Windows**: Settings → Privacy → Microphone → Enable everything
- **Hardware**: Check if mic is physically connected or enabled in Device Manager
- **Restart**: After changing settings, restart the notebook kernel

### "PortAudioError: Invalid device"
- Your system may not support the requested sample rate
- Cell 3 checks supported rates. If 22050 Hz isn't supported, try 44100 Hz and resample:
  ```python
  SAMPLE_RATE = 44100  # Record at native rate
  # Then resample: audio_resampled = librosa.resample(audio, orig_sr=44100, target_sr=22050)
  ```

### "Recording is pure silence"
- Your mic might be muted in Windows Sound settings
- Right-click the speaker icon in taskbar → Sound settings → Input → check volume
- Try a different device index: `sd.default.device = [YOUR_INDEX, None]`

### "Model not found"
- Train the 1D CNN or 2D CNN first (run those notebooks)
- Update `MODEL_PATH` in Cell 2 to point to the correct `.h5` file

### Continuous monitoring shows all detections
- Lower `CONFIDENCE_THRESHOLD` — your background noise might sound similar to the training data
- Re-run the noise floor analysis (Cell 5) to check your environment
- Consider retraining with more diverse non-gunshot data

### Continuous monitoring shows no detections
- The model might need a louder sound than your mic can capture
- Try clapping loudly near the mic — if it doesn't trigger, check normalization
- Lower `CONFIDENCE_THRESHOLD` to 0.5 for testing

---

> **Remember**: The microphone test is for validation only. Real deployment would use the Arduino Nano 33 BLE Sense with its built-in PDM microphone, which has different characteristics than your laptop mic.
