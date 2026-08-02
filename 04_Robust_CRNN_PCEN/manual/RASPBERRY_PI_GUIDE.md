# 🍓 Phase 4: Raspberry Pi Deployment Guide — Robust CRNN-PCEN

This guide explains how to deploy the trained Module 04 Robust CRNN-PCEN gunshot detector on a **Raspberry Pi 4 / 5** or **Raspberry Pi Zero 2 W** with a USB or I2S microphone.

---

## 1. Prerequisites & Hardware Setup

### Hardware Required:
- Raspberry Pi 4B / 5 / Zero 2 W
- USB Microphone (e.g. USB PDM/MEMS mic or Mini USB Mic) OR I2S Microphone (INMP441)
- MicroSD Card with Raspberry Pi OS (64-bit Lite or Desktop)

---

## 2. Environment Installation on Raspberry Pi

Run the following commands in the Raspberry Pi terminal:

```bash
# Update System
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-numpy python3-scipy portaudio19-dev libsndfile1

# Install TFLite Runtime (Lightweight Tensorflow for Pi)
pip3 install tflite-runtime sounddevice soundfile
```

---

## 3. Copy Model Files to Raspberry Pi

Transfer the following files to a folder named `gunshot_detector/` on your Raspberry Pi:

```text
gunshot_detector/
├── pcen_mic_pipeline.py
├── run_pi_crnn.py
└── output/
    ├── crnn_pcen_int8.tflite
    └── pcen_stats.json
```

---

## 4. Raspberry Pi Deployment Script (`run_pi_crnn.py`)

Create `run_pi_crnn.py` on your Raspberry Pi:

```python
"""
Raspberry Pi Real-Time Gunshot Detector (TFLite Runtime)
"""
import time
import json
import numpy as np
import scipy.signal as signal
import tflite_runtime.interpreter as tflite
import sounddevice as sd
from pcen_mic_pipeline import compute_pcen

# Load Stats
stats = json.loads(open('output/pcen_stats.json').read())
norm_mean, norm_std = stats['mean'], stats['std']

# Load TFLite INT8 Model
interpreter = tflite.Interpreter(model_path="output/crnn_pcen_int8.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("✅ TFLite INT8 Model Loaded on Raspberry Pi")

def audio_callback(indata, frames, time_info, status):
    mono = indata.mean(axis=1) if indata.ndim > 1 else indata.flatten()
    rms = np.sqrt(np.mean(mono ** 2))
    
    if rms < 0.001:
        return

    # Extract PCEN
    pcen = compute_pcen(mono, sr=22050, n_mels=64, n_fft=512, hop_length=128)
    pcen_norm = (pcen - norm_mean) / max(norm_std, 1e-6)
    
    # Input tensor shape
    x = pcen_norm.reshape(1, 64, 130, 1).astype(np.float32)
    
    # Run TFLite inference
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prob = float(output_data[0][0])
    
    if prob >= 0.50:
        print(f"\n🚨 [ALERT] GUNSHOT DETECTED! Confidence: {prob*100:.1f}% | RMS: {rms:.5f}")

# Start Stream
with sd.InputStream(samplerate=22050, channels=1, blocksize=16537, callback=audio_callback):
    print("🔴 Raspberry Pi Listening for Gunshots... (Ctrl+C to exit)")
    while True:
        time.sleep(0.5)
```

---

## 5. Running as a Systemd Background Service

To make the Raspberry Pi automatically monitor for gunshots on boot:

1. Create service file:
   ```bash
   sudo nano /etc/systemd/system/gunshot_detector.service
   ```
2. Paste configuration:
   ```ini
   [Unit]
   Description=Gunshot Detection Service
   After=sound.target

   [Service]
   Type=simple
   User=pi
   WorkingDirectory=/home/pi/gunshot_detector
   ExecStart=/usr/bin/python3 run_pi_crnn.py
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```
3. Enable and start:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable gunshot_detector
   sudo systemctl start gunshot_detector
   ```
