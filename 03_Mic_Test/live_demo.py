import os
import sys
import time
import numpy as np
import sounddevice as sd
import librosa
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs

MODEL_PATH = r'..\01_1D_CNN\output\1d_cnn_best.h5'
TARGET_SR = 22050
CLIP_DURATION_MS = 250
TARGET_SAMPLES = int(TARGET_SR * CLIP_DURATION_MS / 1000)
CONFIDENCE_THRESHOLD = 0.7

def find_working_mic():
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            try:
                # Try to test with native settings
                native_sr = int(d['default_samplerate'])
                channels = min(d['max_input_channels'], 2)
                test = sd.rec(int(0.1 * native_sr), samplerate=native_sr, channels=channels, device=i, dtype='float32')
                sd.wait()
                return i, native_sr, channels
            except:
                continue
    return None, None, None

def process_audio(window, native_sr, channels):
    # Convert stereo to mono if needed
    if channels > 1:
        window = window.mean(axis=1)
    else:
        window = window.flatten()
        
    # Resample to the 22050Hz required by model
    if native_sr != TARGET_SR:
        window = librosa.resample(y=window, orig_sr=native_sr, target_sr=TARGET_SR)
        
    # Force exact length
    if len(window) >= TARGET_SAMPLES:
        window = window[:TARGET_SAMPLES]
    else:
        window = np.pad(window, (0, TARGET_SAMPLES - len(window)))
        
    # Normalize
    peak = np.max(np.abs(window))
    if peak > 1e-6:
        window = window / peak
        
    # Format for 1D CNN
    return window.reshape(1, -1, 1).astype(np.float32)

print("🔍 Initializing robust microphone test...")

device_id, native_sr, channels = find_working_mic()
if device_id is None:
    print("❌ Could not find a working microphone.")
    sys.exit(1)

device_name = sd.query_devices()[device_id]['name']
print(f"✅ Selected Device: [{device_id}] {device_name}")
print(f"   Native Rate: {native_sr} Hz | Channels: {channels}")
print(f"   Model Target: {TARGET_SR} Hz (will auto-resample)")

print("\n📦 Loading model...")
try:
    model = keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Could not load model at {MODEL_PATH}: {e}")
    sys.exit(1)

print("\n🔴 LISTENING FOR GUNSHOTS... (Press Ctrl+C to stop)")
print("-" * 50)

hop_ms = int(CLIP_DURATION_MS * 0.5) # 50% overlap
hop_samples_native = int(native_sr * hop_ms / 1000)
window_samples_native = int(native_sr * CLIP_DURATION_MS / 1000)

# Ring buffer for native audio
ring_buffer = np.zeros((window_samples_native, channels), dtype='float32')
loop_counter = 0

print("📝 Detections will be saved to: detections_log.txt")

def audio_callback(indata, frames, time_info, status):
    global ring_buffer, loop_counter
    if status:
        pass # Ignore overflows for now to keep console clean
        
    # Shift ring buffer and add new data
    ring_buffer = np.roll(ring_buffer, -frames, axis=0)
    ring_buffer[-frames:] = indata
    
    # Process the full window
    x = process_audio(ring_buffer, native_sr, channels)
    prob = model.predict(x, verbose=0).flatten()[0]
    
    timestamp = time.strftime("%H:%M:%S")
    loop_counter += 1
    
    if prob >= CONFIDENCE_THRESHOLD:
        msg = f"[{timestamp}] 🔫 GUNSHOT DETECTED! | Confidence: {prob:.4f}"
        print(f"\n{msg}")
        with open("detections_log.txt", "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d ") + msg + "\n")
    else:
        # Print a live update every ~1 second (8 hops)
        if loop_counter % 8 == 0:
            print(f"[{timestamp}] 🎧 Listening... (Background noise max: {prob:.4f})     ", end="\r", flush=True)

try:
    with sd.InputStream(device=device_id, samplerate=native_sr, channels=channels, 
                        blocksize=hop_samples_native, callback=audio_callback):
        while True:
            time.sleep(0.1)
except KeyboardInterrupt:
    print("\n\n🛑 Stopped listening.")
