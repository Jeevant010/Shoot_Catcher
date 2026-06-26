import os
import sys
import io
import time
import numpy as np
import sounddevice as sd

# Fix Windows console encoding for emoji output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import librosa
import soundfile as sf
import threading
import queue
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs

MODEL_PATH = r'..\..\01_1D_CNN\output\1d_cnn_best.h5'
TARGET_SR = 22050
CLIP_DURATION_MS = 250
TARGET_SAMPLES = int(TARGET_SR * CLIP_DURATION_MS / 1000)
CONFIDENCE_THRESHOLD = 0.7

# --- Tuning parameters ---
WARMUP_CALLBACKS = 16        # Skip first N callbacks to fill ring buffer with real audio
ENERGY_GATE_THRESHOLD = 0.005  # Minimum RMS energy to bother running inference (rejects silence)
COOLDOWN_SECONDS = 1.0       # Minimum time between detections (prevents duplicates)

# --- Audio recording ---
SAVE_AUDIO = True             # Save processed audio clips so you can hear what the model hears
# Using absolute or isolated paths to keep project root clean
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
RECORDINGS_DIR = SCRIPT_DIR / 'recordings'
if SAVE_AUDIO:
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = SCRIPT_DIR / "detections_log.txt"


def find_working_mic():
    devices = sd.query_devices()
    valid_mics = []
    
    print("\n🎤 --- AVAILABLE MICROPHONES ---")
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            try:
                # Try to test with native settings
                native_sr = int(d['default_samplerate'])
                channels = min(d['max_input_channels'], 2)
                test = sd.rec(int(0.1 * native_sr), samplerate=native_sr, channels=channels, device=i, dtype='float32')
                sd.wait()
                valid_mics.append((i, d['name'], native_sr, channels))
                print(f" [{len(valid_mics)}] {d['name']} (Rate: {native_sr}Hz, Channels: {channels})")
            except Exception:
                pass
                
    if not valid_mics:
        return None, None, None
        
    print("--------------------------------")
    while True:
        try:
            choice = input(f"👉 Select a microphone (1-{len(valid_mics)}): ")
            idx = int(choice) - 1
            if 0 <= idx < len(valid_mics):
                selected = valid_mics[idx]
                return selected[0], selected[2], selected[3]
            else:
                print("❌ Invalid choice. Try again.")
        except ValueError:
            print("❌ Please enter a number.")


def process_audio(window, native_sr, channels):
    """Process raw audio buffer into model-ready input.
    
    Returns:
        tuple: (processed_array, rms_energy) or (None, rms_energy) if energy too low
    """
    # Convert stereo to mono if needed
    if channels > 1:
        window = window.mean(axis=1)
    else:
        window = window.flatten()

    # Check energy BEFORE resampling (faster)
    rms_energy = np.sqrt(np.mean(window ** 2))

    # Resample to the 22050Hz required by model
    if native_sr != TARGET_SR:
        window = librosa.resample(y=window, orig_sr=native_sr, target_sr=TARGET_SR)

    # Force exact length
    if len(window) >= TARGET_SAMPLES:
        window = window[:TARGET_SAMPLES]
    else:
        window = np.pad(window, (0, TARGET_SAMPLES - len(window)))

    # Normalize to [-1, 1] by peak (matching training preprocessing)
    peak = np.max(np.abs(window))
    if peak > 1e-6:
        window = window / peak
    else:
        # Signal is effectively silent — return None to skip inference
        return None, rms_energy

    # Format for 1D CNN
    # Return the normalized 1D waveform too (for saving to .wav)
    return window.reshape(1, -1, 1).astype(np.float32), rms_energy, window.astype(np.float32)


# ============================================================
# INITIALIZATION
# ============================================================
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

# Fix for Keras 2 -> Keras 3 compatibility (BatchNormalization renorm args removed)
class CompatBatchNormalization(keras.layers.BatchNormalization):
    def __init__(self, **kwargs):
        # Strip args that existed in Keras 2 but were removed in Keras 3
        kwargs.pop('renorm', None)
        kwargs.pop('renorm_clipping', None)
        kwargs.pop('renorm_momentum', None)
        super().__init__(**kwargs)

try:
    model = keras.models.load_model(
        MODEL_PATH,
        custom_objects={'BatchNormalization': CompatBatchNormalization}
    )
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Could not load model at {MODEL_PATH}: {e}")
    sys.exit(1)

# ============================================================
# INFERENCE THREAD (keeps audio callback fast)
# ============================================================
audio_queue = queue.Queue(maxsize=4)  # Buffer up to 4 windows
last_detection_time = 0.0


def inference_thread_fn():
    """Run model inference on a separate thread so the audio callback stays fast."""
    global last_detection_time
    loop_counter = 0

    while True:
        try:
            x, rms, wav_data = audio_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        if x is None:
            # Poison pill — shut down
            break

        prob = model.predict(x, verbose=0).flatten()[0]
        timestamp = time.strftime("%H:%M:%S")
        timestamp_file = time.strftime("%H%M%S")
        loop_counter += 1
        now = time.time()

        if prob >= CONFIDENCE_THRESHOLD:
            # Cooldown: don't spam detections
            if (now - last_detection_time) >= COOLDOWN_SECONDS:
                last_detection_time = now
                msg = f"[{timestamp}] \U0001f52b GUNSHOT DETECTED! | Confidence: {prob:.4f} | RMS: {rms:.5f}"
                print(f"\n{msg}")
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(time.strftime("%Y-%m-%d ") + msg + "\n")

                # Save the audio the model processed for this detection
                if SAVE_AUDIO and wav_data is not None:
                    fname = RECORDINGS_DIR / f"GUNSHOT_{timestamp_file}_conf{prob:.3f}.wav"
                    sf.write(str(fname), wav_data, TARGET_SR)
                    print(f"   💾 Saved processed audio → {fname}")
        else:
            # Print a live update every ~1 second (8 hops)
            if loop_counter % 8 == 0:
                print(f"[{timestamp}] \U0001f3a7 Listening... (prob: {prob:.4f} | RMS: {rms:.5f})     ", end="\r", flush=True)

                # Periodically save a "what model hears" sample (every ~8 seconds)
                if SAVE_AUDIO and wav_data is not None and loop_counter % 64 == 0:
                    fname = RECORDINGS_DIR / f"ambient_{timestamp_file}_prob{prob:.3f}.wav"
                    sf.write(str(fname), wav_data, TARGET_SR)
                    print(f"\n   💾 Saved ambient sample → {fname}")


# Start inference thread
inf_thread = threading.Thread(target=inference_thread_fn, daemon=True)
inf_thread.start()

# ============================================================
# AUDIO CALLBACK (must be fast — no ML here!)
# ============================================================
hop_ms = int(CLIP_DURATION_MS * 0.5)  # 50% overlap
hop_samples_native = int(native_sr * hop_ms / 1000)
window_samples_native = int(native_sr * CLIP_DURATION_MS / 1000)

# Ring buffer for native audio
ring_buffer = np.zeros((window_samples_native, channels), dtype='float32')
callback_count = 0


def audio_callback(indata, frames, time_info, status):
    global ring_buffer, callback_count

    if status:
        pass  # Ignore overflows for now to keep console clean

    # Shift ring buffer and add new data
    ring_buffer = np.roll(ring_buffer, -frames, axis=0)
    ring_buffer[-frames:] = indata

    callback_count += 1

    # --- FIX 1: Skip warmup period (let buffer fill with real audio) ---
    if callback_count < WARMUP_CALLBACKS:
        if callback_count == 1:
            print("⏳ Warming up microphone buffer...")
        return

    if callback_count == WARMUP_CALLBACKS:
        print("✅ Warmup complete — now listening!\n")

    # --- FIX 2: Process audio and check energy gate ---
    result = process_audio(ring_buffer.copy(), native_sr, channels)
    if result[0] is None:
        # Silent — no audio to process
        return
    x, rms, wav_data = result

    # Skip if signal is silent (energy gate)
    if rms < ENERGY_GATE_THRESHOLD:
        return

    # --- FIX 3: Queue for inference thread (non-blocking) ---
    try:
        # Aggressive drop: If queue is full, remove oldest item to keep up with realtime
        if audio_queue.full():
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                pass
        audio_queue.put_nowait((x, rms, wav_data))
    except queue.Full:
        pass  # Drop frame if inference can't keep up (better than blocking audio)


# ============================================================
# MAIN LOOP
# ============================================================
print("\n🔴 LISTENING FOR GUNSHOTS... (Press Ctrl+C to stop)")
print("-" * 50)
print(f"📝 Detections will be saved to: {LOG_FILE}")
print(f"⚙️  Confidence threshold: {CONFIDENCE_THRESHOLD}")
print(f"⚙️  Energy gate: {ENERGY_GATE_THRESHOLD}")
print(f"⚙️  Cooldown: {COOLDOWN_SECONDS}s\n")

try:
    with sd.InputStream(device=device_id, samplerate=native_sr, channels=channels,
                        blocksize=hop_samples_native, callback=audio_callback):
        while True:
            time.sleep(0.1)
except KeyboardInterrupt:
    # Signal inference thread to stop
    audio_queue.put((None, 0))
    inf_thread.join(timeout=2.0)
    print("\n\n🛑 Stopped listening.")
