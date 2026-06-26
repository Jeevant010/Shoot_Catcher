"""
Test the 1D CNN gunshot model against a .wav recording file.
Slides a 250ms window across the file and reports all detections.
Also saves each processed window as a .wav so you can hear exactly what the model "hears".
"""
import os
import sys
import io
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

# Fix Windows console encoding for emoji output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = r'..\..\01_1D_CNN\output\1d_cnn_best.h5'
TARGET_SR = 22050
CLIP_DURATION_MS = 250
TARGET_SAMPLES = int(TARGET_SR * CLIP_DURATION_MS / 1000)  # 5512
CONFIDENCE_THRESHOLD = 0.5  # Lower threshold to catch weak detections too
HOP_MS = 125  # 50% overlap — slide by 125ms each step

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = SCRIPT_DIR / 'test_output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# INTERACTIVE MENU
# ============================================================
print("\n🎧 --- SELECT AUDIO FILE TO TEST ---")
print("1. Use default 'test_recording.wav'")
print("2. Enter custom file path (e.g. downloaded clip)")
while True:
    choice = input("👉 Enter choice (1 or 2): ").strip()
    if choice == '1':
        RECORDING_PATH = str(SCRIPT_DIR.parent / 'test_recording.wav')
        break
    elif choice == '2':
        RECORDING_PATH = input("👉 Enter full path to .wav file: ").strip()
        # Remove quotes if user dragged-and-dropped
        RECORDING_PATH = RECORDING_PATH.strip('"').strip("'")
        break
    else:
        print("❌ Invalid choice.")

if not os.path.exists(RECORDING_PATH):
    print(f"❌ File not found: {RECORDING_PATH}")
    sys.exit(1)

# ============================================================
# LOAD MODEL
# ============================================================
print("📦 Loading model...")

class CompatBatchNormalization(keras.layers.BatchNormalization):
    def __init__(self, **kwargs):
        kwargs.pop('renorm', None)
        kwargs.pop('renorm_clipping', None)
        kwargs.pop('renorm_momentum', None)
        super().__init__(**kwargs)

try:
    model = keras.models.load_model(
        MODEL_PATH,
        custom_objects={'BatchNormalization': CompatBatchNormalization}
    )
    print("✅ Model loaded successfully!\n")
except Exception as e:
    print(f"❌ Could not load model: {e}")
    sys.exit(1)

# ============================================================
# LOAD RECORDING
# ============================================================
print(f"🎵 Loading recording: {RECORDING_PATH}")
raw_audio, file_sr = sf.read(RECORDING_PATH)
print(f"   Original: SR={file_sr} Hz, Shape={raw_audio.shape}, Duration={len(raw_audio)/file_sr:.2f}s")
print(f"   Peak amplitude: {np.max(np.abs(raw_audio)):.6f}")
print(f"   RMS energy: {np.sqrt(np.mean(raw_audio**2)):.6f}")

# Convert stereo to mono
if raw_audio.ndim > 1:
    raw_audio = raw_audio.mean(axis=1)
    print("   Converted stereo → mono")

# Resample to target SR
if file_sr != TARGET_SR:
    raw_audio = librosa.resample(y=raw_audio.astype(np.float32), orig_sr=file_sr, target_sr=TARGET_SR)
    print(f"   Resampled {file_sr} → {TARGET_SR} Hz")

total_samples = len(raw_audio)
print(f"   Final: {total_samples} samples, {total_samples/TARGET_SR:.2f}s\n")

# Save the full mono/resampled version for reference
sf.write(str(OUTPUT_DIR / 'full_resampled.wav'), raw_audio, TARGET_SR)
print(f"💾 Full resampled audio saved to: {OUTPUT_DIR / 'full_resampled.wav'}\n")

# ============================================================
# SLIDING WINDOW INFERENCE
# ============================================================
hop_samples = int(TARGET_SR * HOP_MS / 1000)
n_windows = (total_samples - TARGET_SAMPLES) // hop_samples + 1

print(f"🔍 Running inference on {n_windows} windows ({CLIP_DURATION_MS}ms each, {HOP_MS}ms hop)...")
print(f"   Confidence threshold for DETECTION: {CONFIDENCE_THRESHOLD}")
print("=" * 70)

detections = []
all_results = []

for i in range(n_windows):
    start = i * hop_samples
    end = start + TARGET_SAMPLES
    window = raw_audio[start:end].copy()

    # Compute raw energy
    rms = np.sqrt(np.mean(window ** 2))
    peak = np.max(np.abs(window))

    # Peak normalize (matching training preprocessing)
    if peak > 1e-6:
        window_norm = window / peak
    else:
        # Silent window
        window_norm = window

    # Format for model
    x = window_norm.reshape(1, -1, 1).astype(np.float32)

    # Predict
    prob = model.predict(x, verbose=0).flatten()[0]

    time_start_ms = start / TARGET_SR * 1000
    time_end_ms = end / TARGET_SR * 1000

    result = {
        'window': i,
        'time_start_ms': time_start_ms,
        'time_end_ms': time_end_ms,
        'prob': prob,
        'rms': rms,
        'peak': peak,
    }
    all_results.append(result)

    # Mark as detection?
    is_detection = prob >= CONFIDENCE_THRESHOLD
    marker = "🔫 GUNSHOT!" if is_detection else "   ambient"

    if is_detection:
        detections.append(result)

    # Print every window
    bar = "█" * int(prob * 30) + "░" * (30 - int(prob * 30))
    print(f"  [{time_start_ms:7.1f}-{time_end_ms:7.1f}ms] {marker}  conf={prob:.4f} |{bar}| rms={rms:.5f} peak={peak:.5f}")

    # Save the processed (normalized) window as .wav so you can listen
    sf.write(str(OUTPUT_DIR / f'window_{i:03d}_conf{prob:.3f}.wav'), window_norm, TARGET_SR)

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

if detections:
    print(f"\n🔫 {len(detections)} gunshot detection(s) found:\n")
    for d in detections:
        print(f"   [{d['time_start_ms']:.0f}-{d['time_end_ms']:.0f}ms] "
              f"Confidence: {d['prob']:.4f} | RMS: {d['rms']:.5f}")
else:
    print(f"\n❌ NO gunshot detections above threshold ({CONFIDENCE_THRESHOLD})")

# Top 5 highest confidence windows regardless of threshold
print(f"\n📊 Top 5 highest confidence windows (any threshold):")
sorted_results = sorted(all_results, key=lambda r: r['prob'], reverse=True)
for r in sorted_results[:5]:
    print(f"   [{r['time_start_ms']:.0f}-{r['time_end_ms']:.0f}ms] "
          f"conf={r['prob']:.4f} | rms={r['rms']:.5f} | peak={r['peak']:.5f}")

# Audio energy summary
rms_values = [r['rms'] for r in all_results]
peak_values = [r['peak'] for r in all_results]
print(f"\n📊 Audio energy stats:")
print(f"   RMS  — min: {min(rms_values):.6f}, max: {max(rms_values):.6f}, avg: {np.mean(rms_values):.6f}")
print(f"   Peak — min: {min(peak_values):.6f}, max: {max(peak_values):.6f}, avg: {np.mean(peak_values):.6f}")

if max(peak_values) < 0.01:
    print("\n⚠️  WARNING: The recording is EXTREMELY quiet.")
    print("   The peak amplitude is below 1% of full scale.")
    print("   The mic may not be picking up external audio properly.")
elif max(peak_values) < 0.1:
    print("\n⚠️  NOTE: The recording is quite quiet (peak < 10% of full scale).")
    print("   Peak normalization will amplify noise along with the signal.")

print(f"\n💾 All processed windows saved to: {OUTPUT_DIR}/")
print("   Listen to the high-confidence ones to hear what the model 'heard'.")
