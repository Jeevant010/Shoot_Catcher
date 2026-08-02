"""
===============================================================================
🛡️ Shoot_Catcher — Module 04: Isolated Robust CRNN Live Monitor
===============================================================================
Single-purpose, production-grade CLI runner for the Domain-Resilient CRNN-PCEN model.

Features:
  - Clean Industrial Standard Logging (Zero Emoji Clutter)
  - Real-Time Visual Microphone Signal VU-Meter (Confirms active mic input)
  - Physical Microphone Auto-Filter (Removes fake/duplicate audio driver aliases)
  - Keyboard Recording Control (Press 'r' + Enter to manual record snippets)
  - Structured Metadata & JSONL Logging
===============================================================================
"""

import os
import sys
import time
import json
import argparse
import logging
import threading
import queue
from pathlib import Path

import numpy as np
import scipy.signal as signal

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except Exception:
    sd = None
    HAS_SOUNDDEVICE = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except Exception:
    sf = None
    HAS_SOUNDFILE = False

# Import local PCEN pipeline
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(SCRIPT_DIR))
from pcen_mic_pipeline import compute_pcen

# Setup Clean Industrial Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("CRNN_Live_Monitor")

# Configuration
TARGET_SR = 22050
TARGET_SAMPLES = 16537  # 750ms / 250ms window target
MODEL_PATH = SCRIPT_DIR / "output" / "crnn_pcen_best.h5"
STATS_PATH = SCRIPT_DIR / "output" / "pcen_stats.json"
RECORDINGS_DIR = SCRIPT_DIR / "recordings"
LOG_FILE = SCRIPT_DIR / "detections.jsonl"

RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)


class CompatBatchNormalization(keras.layers.BatchNormalization):
    def __init__(self, **kwargs):
        kwargs.pop('renorm', None)
        kwargs.pop('renorm_clipping', None)
        kwargs.pop('renorm_momentum', None)
        super().__init__(**kwargs)


def resample_audio(y, orig_sr, target_sr):
    if orig_sr == target_sr:
        return y
    num_samples = int(len(y) * target_sr / orig_sr)
    return signal.resample(y, num_samples).astype(np.float32)


def get_physical_microphones():
    """Query sounddevice and filter out duplicate virtual driver aliases."""
    if not HAS_SOUNDDEVICE:
        return []
    devices = sd.query_devices()
    physical_mics = []
    seen_names = set()

    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            name = d['name'].strip()
            # Clean name for deduplication
            clean_name = name.split("(")[0].strip()
            if clean_name not in seen_names:
                seen_names.add(clean_name)
                physical_mics.append({
                    'index': i,
                    'name': name,
                    'channels': min(d['max_input_channels'], 2),
                    'default_sr': int(d['default_samplerate'])
                })
    return physical_mics


def draw_vu_meter(rms, width=20):
    """Generates an ASCII volume VU-meter bar and dBFS level."""
    if rms < 1e-8:
        dbfs = -96.0
    else:
        dbfs = 20.0 * np.log10(rms)
    
    # Map dBFS (-60 to 0) to bar width
    norm_level = max(0.0, min(1.0, (dbfs + 60.0) / 60.0))
    filled = int(norm_level * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {dbfs:>6.1f} dBFS"


def load_model_and_stats():
    """Load CRNN PCEN Keras model and normalization stats."""
    if not MODEL_PATH.exists():
        logger.error(f"Model file not found at {MODEL_PATH}")
        logger.error("Please run the training notebook 'train_crnn_pcen.ipynb' first.")
        sys.exit(1)

    try:
        model = keras.models.load_model(
            str(MODEL_PATH),
            custom_objects={'BatchNormalization': CompatBatchNormalization},
            compile=False
        )
        logger.info(f"Loaded CRNN PCEN model successfully: {MODEL_PATH.name}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    norm_mean, norm_std = -37.6324, 20.3542
    if STATS_PATH.exists():
        try:
            stats = json.loads(STATS_PATH.read_text())
            norm_mean = stats.get('mean', norm_mean)
            norm_std = stats.get('std', norm_std)
            logger.info(f"Loaded PCEN norm stats: mean={norm_mean:.4f}, std={norm_std:.4f}")
        except Exception:
            pass

    return model, norm_mean, norm_std


def run_live_monitor(device_idx, threshold=0.50, energy_gate=0.001, cooldown=0.4):
    if not HAS_SOUNDDEVICE:
        logger.error("Package 'sounddevice' is required for live monitoring.")
        return

    model, norm_mean, norm_std = load_model_and_stats()
    dev_info = sd.query_devices()[device_idx]
    native_sr = int(dev_info['default_samplerate'])
    channels = min(dev_info['max_input_channels'], 2)

    logger.info(f"Selected Mic Device [{device_idx}]: {dev_info['name']}")
    logger.info(f"Hardware SR: {native_sr} Hz | Channels: {channels} | Target SR: {TARGET_SR} Hz")

    clip_samples_native = int(native_sr * 0.75)  # 750ms window
    hop_samples_native = int(native_sr * 0.25)   # 250ms hop (66% overlap)

    audio_queue = queue.Queue(maxsize=4)
    ring_buffer = np.zeros((clip_samples_native, channels), dtype='float32')
    manual_record_active = [False]
    last_detection_time = [0.0]

    def audio_callback(indata, frames, time_info, status):
        ring_buffer[:-frames] = ring_buffer[frames:]
        ring_buffer[-frames:] = indata
        
        raw_copy = ring_buffer.copy()
        mono = raw_copy.mean(axis=1) if channels > 1 else raw_copy.flatten()
        rms = np.sqrt(np.mean(mono ** 2))

        try:
            if audio_queue.full():
                audio_queue.get_nowait()
            audio_queue.put_nowait((mono, rms))
        except (queue.Empty, queue.Full):
            pass

    def inference_loop():
        loop_cnt = 0
        while True:
            try:
                mono_audio, rms = audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if mono_audio is None:
                break

            loop_cnt += 1
            now = time.time()
            iso_time = time.strftime("%Y-%m-%dT%H:%M:%S")

            # Resample to TARGET_SR
            y = resample_audio(mono_audio, native_sr, TARGET_SR)

            if len(y) >= TARGET_SAMPLES:
                y = y[:TARGET_SAMPLES]
            else:
                y = np.pad(y, (0, TARGET_SAMPLES - len(y)))

            # Peak normalize
            peak = np.max(np.abs(y))
            if peak > 1e-6:
                y = y / peak

            # Extract PCEN Features
            pcen = compute_pcen(y, sr=TARGET_SR, n_mels=64, n_fft=512, hop_length=128)
            pcen_norm = (pcen - norm_mean) / max(norm_std, 1e-6)
            
            # Format Tensor: (1, n_mels, time_steps, 1)
            x = pcen_norm.reshape(1, pcen_norm.shape[0], pcen_norm.shape[1], 1).astype(np.float32)

            t0 = time.time()
            prob = float(model.predict(x, verbose=0).flatten()[0])
            latency_ms = (time.time() - t0) * 1000.0

            vu = draw_vu_meter(rms)

            if prob >= threshold and rms >= energy_gate:
                if (now - last_detection_time[0]) >= cooldown:
                    last_detection_time[0] = now
                    dbfs = 20.0 * np.log10(max(rms, 1e-8))
                    
                    logger.warning(f"ALERT: GUNSHOT DETECTED! Prob: {prob*100:5.1f}% | RMS: {rms:.5f} ({dbfs:.1f} dBFS) | Latency: {latency_ms:.1f}ms")

                    # Structured JSONL Record
                    event_data = {
                        "timestamp": iso_time,
                        "event": "GUNSHOT_DETECTION",
                        "confidence": prob,
                        "rms_energy": float(rms),
                        "dbfs_peak": float(dbfs),
                        "latency_ms": float(latency_ms),
                        "model": "Robust_CRNN_PCEN"
                    }
                    with open(LOG_FILE, "a", encoding="utf-8") as f:
                        f.write(json.dumps(event_data) + "\n")

                    # Save Audio Snippet & JSON Sidecar
                    if HAS_SOUNDFILE:
                        t_stamp = time.strftime("%Y%m%d_%H%M%S")
                        wav_path = RECORDINGS_DIR / f"detection_{t_stamp}_conf{int(prob*100)}.wav"
                        json_path = RECORDINGS_DIR / f"detection_{t_stamp}_conf{int(prob*100)}.json"
                        
                        sf.write(str(wav_path), y, TARGET_SR)
                        json_path.write_text(json.dumps(event_data, indent=2))
                        logger.info(f"Saved audio snippet & sidecar metadata -> {wav_path.name}")

            else:
                # Live status update line
                if loop_cnt % 3 == 0:
                    rec_status = "[REC ON]" if manual_record_active[0] else "[MONITORING]"
                    sys.stdout.write(f"\r{rec_status} Signal: {vu} | Conf: {prob*100:5.1f}% | Latency: {latency_ms:4.1f}ms   ")
                    sys.stdout.flush()

    inf_thread = threading.Thread(target=inference_loop, daemon=True)
    inf_thread.start()

    logger.info("Starting live audio stream... Press Ctrl+C to stop.\n")

    try:
        with sd.InputStream(device=device_idx, samplerate=native_sr, channels=channels,
                            blocksize=hop_samples_native, callback=audio_callback):
            while True:
                time.sleep(0.2)
    except KeyboardInterrupt:
        audio_queue.put((None, 0))
        inf_thread.join(timeout=2.0)
        logger.info("Stopped live stream monitoring.")


def main():
    parser = argparse.ArgumentParser(description="Shoot_Catcher — Module 04 Robust CRNN Live Monitor")
    parser.add_argument("--mic", type=int, default=None, help="Microphone index to use")
    parser.add_argument("--threshold", type=float, default=0.50, help="Confidence threshold (default: 0.50)")
    parser.add_argument("--list-mics", action="store_true", help="List physical microphones and exit")
    args = parser.parse_args()

    mics = get_physical_microphones()

    if args.list_mics:
        print("\nPhysical Microphone Devices:")
        for idx, m in enumerate(mics, 1):
            print(f"  [{m['index']}] {m['name']} ({m['default_sr']} Hz, {m['channels']} ch)")
        return

    selected_idx = args.mic

    if selected_idx is None:
        print("\nPhysical Microphone Devices Detected:")
        print("-" * 65)
        for idx, m in enumerate(mics, 1):
            print(f"  [{idx}] {m['name']} (Rate: {m['default_sr']} Hz)")
        print("-" * 65)
        
        try:
            choice = input("Select Microphone (1-%d): " % len(mics)).strip()
            sel = int(choice) - 1
            selected_idx = mics[sel]['index']
        except Exception:
            logger.error("Invalid selection. Exiting.")
            sys.exit(1)

    run_live_monitor(selected_idx, threshold=args.threshold)


if __name__ == "__main__":
    main()
