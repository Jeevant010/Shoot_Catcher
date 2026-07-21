"""
===============================================================================
🎯 Shoot_Catcher — Multi-Model Gunshot Intelligence & Live Dashboard Hub
===============================================================================
Robust, real-time live monitoring and benchmark tool supporting all 4 model modules:
  1. Baseline 1D CNN           (01_1D_CNN)
  2. Baseline 2D CNN           (02_2D_CNN - Mel Spectrogram)
  3. Enhanced 1D CNN           (Enhanced_Models/01_Enhanced_1D_CNN - Dual Head)
  4. Enhanced 2D CNN           (Enhanced_Models/02_Enhanced_2D_CNN)

Gracefully handles untrained/missing models without crashing or errors.
Supports single-model live stream, multi-model live dashboard, and recording file benchmarks.
===============================================================================
"""

import os
import sys
import io
import time
import json
import numpy as np
import scipy.signal as signal
import threading
import queue
from pathlib import Path

# Fix Windows console encoding for emoji output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs

import tensorflow as tf
from tensorflow import keras

# Try importing sounddevice safely
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except Exception:
    sd = None
    HAS_SOUNDDEVICE = False

# Try importing soundfile safely
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except Exception:
    sf = None
    HAS_SOUNDFILE = False

# Try importing librosa safely (fallback to scipy/numpy if numba/numpy mismatch occurs)
try:
    import librosa
    HAS_LIBROSA = True
except Exception:
    HAS_LIBROSA = False

# ============================================================
# GLOBAL CONSTANTS & PATHS
# ============================================================
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = SCRIPT_DIR.parent.parent

RECORDINGS_DIR = SCRIPT_DIR / 'recordings'
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
TEST_OUTPUT_DIR = SCRIPT_DIR / 'test_output'
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = SCRIPT_DIR / "detections_log.txt"

# Audio & Threshold Parameters (Optimized defaults for phone speaker & live mic testing)
TARGET_SR = 22050
ENERGY_GATE_THRESHOLD = 0.001    # Low energy gate so quiet phone audio isn't ignored
CONFIDENCE_THRESHOLD = 0.50     # 50% threshold for high sensitivity during live testing
COOLDOWN_SECONDS = 0.4          # 0.4s cooldown so consecutive gunshots are caught quickly
WARMUP_CALLBACKS = 16


# Keras 2 -> Keras 3 compatibility helper
class CompatBatchNormalization(keras.layers.BatchNormalization):
    def __init__(self, **kwargs):
        kwargs.pop('renorm', None)
        kwargs.pop('renorm_clipping', None)
        kwargs.pop('renorm_momentum', None)
        super().__init__(**kwargs)


# ============================================================
# MODEL MODULE SPECIFICATIONS
# ============================================================
MODULE_SPECS = [
    {
        'id': '01_1d_cnn',
        'name': 'Baseline 1D CNN',
        'module': '01_1D_CNN',
        'relative_path': Path('01_1D_CNN/output/1d_cnn_best.h5'),
        'fallback_path': Path('01_1D_CNN/output/1d_cnn_gunshot_detector.h5'),
        'is_2d': False,
    },
    {
        'id': '02_2d_cnn',
        'name': 'Baseline 2D CNN (Mel Spectrogram)',
        'module': '02_2D_CNN',
        'relative_path': Path('02_2D_CNN/output/2d_cnn_mel_spectrogram_best.h5'),
        'norm_stats_path': Path('02_2D_CNN/output/2d_cnn_norm_stats.json'),
        'is_2d': True,
    },
    {
        'id': '01_enhanced_1d',
        'name': 'Enhanced 1D CNN (Dual-Head)',
        'module': 'Enhanced_Models/01_Enhanced_1D_CNN',
        'relative_path': Path('Enhanced_Models/01_Enhanced_1D_CNN/output/enhanced_1d_cnn_best.h5'),
        'is_2d': False,
    },
    {
        'id': '02_enhanced_2d',
        'name': 'Enhanced 2D CNN',
        'module': 'Enhanced_Models/02_Enhanced_2D_CNN',
        'relative_path': Path('Enhanced_Models/02_Enhanced_2D_CNN/output/enhanced_2d_cnn_best.h5'),
        'norm_stats_path': Path('Enhanced_Models/02_Enhanced_2D_CNN/output/2d_cnn_norm_stats.json'),
        'is_2d': True,
    },
]


# ============================================================
# HELPER: PURE SCIPY/NUMPY SPECTROGRAM COMPUTATION
# ============================================================
def compute_mel_spectrogram_scipy(y, sr=22050, n_mels=64, n_fft=512, hop_length=128):
    """Compute Log-Mel Spectrogram using pure Scipy/NumPy (Numba independent)."""
    f, t, Zxx = signal.stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length, boundary=None)
    power = np.abs(Zxx) ** 2

    low_freq, high_freq = 0, sr / 2.0
    mel_low = 2595 * np.log10(1 + low_freq / 700.0)
    mel_high = 2595 * np.log10(1 + high_freq / 700.0)
    mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = 700.0 * (10 ** (mel_points / 2595.0) - 1.0)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    num_bins = n_fft // 2 + 1
    fb = np.zeros((n_mels, num_bins), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]
        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                fb[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                fb[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    mel_spec = np.dot(fb, power)
    log_mel = 10.0 * np.log10(np.maximum(mel_spec, 1e-10))
    log_mel -= np.max(log_mel)
    return log_mel.astype(np.float32)


def resample_audio(y, orig_sr, target_sr):
    """Resample audio array using scipy.signal."""
    if orig_sr == target_sr:
        return y
    num_samples = int(len(y) * target_sr / orig_sr)
    return signal.resample(y, num_samples).astype(np.float32)


# ============================================================
# MODEL WRAPPER CLASS
# ============================================================
class ModelWrapper:
    def __init__(self, spec):
        self.spec = spec
        self.id = spec['id']
        self.name = spec['name']
        self.module = spec['module']
        self.is_2d = spec['is_2d']
        self.model_path = PROJECT_ROOT / spec['relative_path']

        # Fallback path if best model isn't found
        if not self.model_path.exists() and 'fallback_path' in spec:
            fallback = PROJECT_ROOT / spec['fallback_path']
            if fallback.exists():
                self.model_path = fallback

        self.norm_stats_path = PROJECT_ROOT / spec['norm_stats_path'] if 'norm_stats_path' in spec else None

        self.model = None
        self.status = "NOT TRAINED / FILE MISSING"
        self.is_dual_head = False
        self.input_samples = 16537
        self.input_shape = (None, 64, 130, 1) if self.is_2d else (None, 16537, 1)
        self.output_shape = (None, 1)
        self.norm_mean = -37.6324
        self.norm_std = 20.3542

        # Load normalization stats if available
        if self.norm_stats_path and self.norm_stats_path.exists():
            try:
                stats = json.loads(self.norm_stats_path.read_text())
                self.norm_mean = stats.get('mean', self.norm_mean)
                self.norm_std = stats.get('std', self.norm_std)
            except Exception:
                pass

        # Attempt to load model with compile=False for Keras 3 compatibility
        if self.model_path.exists():
            try:
                self.model = keras.models.load_model(
                    str(self.model_path),
                    custom_objects={'BatchNormalization': CompatBatchNormalization},
                    compile=False
                )
                self.status = "TRAINED & LOADED"

                # Safely inspect shape attributes without raising error
                try:
                    if hasattr(self.model, 'layers') and len(self.model.layers) > 0:
                        first_layer = self.model.layers[0]
                        if hasattr(first_layer, 'input_shape') and first_layer.input_shape is not None:
                            self.input_shape = first_layer.input_shape
                        elif hasattr(first_layer, 'batch_input_shape') and first_layer.batch_input_shape is not None:
                            self.input_shape = first_layer.batch_input_shape
                except Exception:
                    pass

                try:
                    if isinstance(self.model.output, list) and len(self.model.output) == 2:
                        self.is_dual_head = True
                except Exception:
                    pass

                # Determine expected samples/dimensions
                if not self.is_2d:
                    if self.input_shape and len(self.input_shape) >= 2 and self.input_shape[1] is not None:
                        self.input_samples = self.input_shape[1]
                else:
                    self.input_samples = 16537

            except Exception as e:
                self.status = f"ERROR LOADING ({str(e)[:35]}...)"
                self.model = None

    @property
    def is_available(self):
        return self.model is not None

    def predict(self, raw_audio, native_sr):
        """Run prediction on raw audio array. Returns (gunshot_prob, anomaly_score, processed_wav)."""
        if not self.is_available:
            return 0.0, None, None

        # Convert stereo to mono
        if raw_audio.ndim > 1:
            y = raw_audio.mean(axis=1)
        else:
            y = raw_audio.flatten()

        # Resample to TARGET_SR
        y = resample_audio(y, native_sr, TARGET_SR)

        # Force exact sample length
        if len(y) >= self.input_samples:
            y = y[:self.input_samples]
        else:
            y = np.pad(y, (0, self.input_samples - len(y)))

        # Peak normalize to [-1, 1]
        peak = np.max(np.abs(y))
        if peak > 1e-6:
            y = y / peak
        else:
            y = np.zeros_like(y)

        if not self.is_2d:
            # 1D Raw Audio tensor: (1, input_samples, 1)
            x = y.reshape(1, -1, 1).astype(np.float32)
        else:
            # 2D Spectrogram tensor: (1, n_mels, time_steps, 1)
            n_mels = 64
            target_time_steps = 130
            if self.input_shape and len(self.input_shape) >= 3 and self.input_shape[1] is not None:
                n_mels = self.input_shape[1]
            if self.input_shape and len(self.input_shape) >= 3 and self.input_shape[2] is not None:
                target_time_steps = self.input_shape[2]

            spec = compute_mel_spectrogram_scipy(y, sr=TARGET_SR, n_mels=n_mels, n_fft=512, hop_length=128)

            # Pad or truncate time_steps
            if spec.shape[1] < target_time_steps:
                spec = np.pad(spec, ((0, 0), (0, target_time_steps - spec.shape[1])))
            else:
                spec = spec[:, :target_time_steps]

            # Normalize using norm stats
            spec_norm = (spec - self.norm_mean) / max(self.norm_std, 1e-6)
            x = spec_norm.reshape(1, n_mels, target_time_steps, 1).astype(np.float32)

        # Execute prediction
        out = self.model.predict(x, verbose=0)
        if self.is_dual_head:
            gunshot_prob = float(out[0].flatten()[0])
            anomaly_score = float(out[1].flatten()[0])
        else:
            gunshot_prob = float(out.flatten()[0])
            anomaly_score = None

        return gunshot_prob, anomaly_score, y


# ============================================================
# MODEL MANAGER / DISCOVERY
# ============================================================
class ModelManager:
    def __init__(self):
        print("🔍 Scanning for trained models across all 4 modules...")
        self.models = [ModelWrapper(spec) for spec in MODULE_SPECS]
        self.trained_models = [m for m in self.models if m.is_available]

    def print_audit_table(self):
        print("\n" + "=" * 80)
        print("📋 MODULE & MODEL STATUS AUDIT")
        print("=" * 80)
        print(f" {'#':<3} {'Module Name':<35} {'Status':<24} {'Type':<12}")
        print("-" * 80)
        for i, m in enumerate(self.models, 1):
            type_str = "2D Spec" if m.is_2d else ("1D Dual" if m.is_dual_head else "1D Wave")
            status_color = "✅ " + m.status if m.is_available else "⚠️  " + m.status
            print(f" [{i}] {m.name:<35} {status_color:<24} {type_str:<12}")
        print("=" * 80)
        print(f" Total Modules: {len(self.models)} | Trained: {len(self.trained_models)} | Missing/Untrained: {len(self.models) - len(self.trained_models)}\n")


# ============================================================
# MICROPHONE HELPER
# ============================================================
def find_working_mic():
    if not HAS_SOUNDDEVICE:
        print("\n❌ Python package 'sounddevice' is not installed.")
        print("   Run: pip install sounddevice")
        return None, None, None

    devices = sd.query_devices()
    valid_mics = []

    print("\n🎤 --- AVAILABLE MICROPHONES ---")
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            try:
                native_sr = int(d['default_samplerate'])
                channels = min(d['max_input_channels'], 2)
                test = sd.rec(int(0.1 * native_sr), samplerate=native_sr, channels=channels, device=i, dtype='float32')
                sd.wait()
                valid_mics.append((i, d['name'], native_sr, channels))
                print(f" [{len(valid_mics)}] {d['name']} (Rate: {native_sr}Hz, Channels: {channels})")
            except Exception:
                pass

    if not valid_mics:
        print("❌ No working input microphones found.")
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


# ============================================================
# REAL-TIME LIVE MONITORING (SINGLE OR MULTI-MODEL)
# ============================================================
def run_live_monitoring(active_models, device_id, native_sr, channels, is_multi=True):
    """Run real-time live monitoring thread using single or multiple active models."""
    if not active_models:
        print("❌ No trained models available to run live monitoring.")
        return
    if not HAS_SOUNDDEVICE:
        print("❌ sounddevice package is required for live monitoring.")
        return

    # Calculate buffer size based on largest required sample window
    max_samples = max(m.input_samples for m in active_models)
    clip_duration_ms = int(max_samples / TARGET_SR * 1000)
    hop_ms = int(clip_duration_ms * 0.5)

    hop_samples_native = int(native_sr * hop_ms / 1000)
    window_samples_native = int(native_sr * clip_duration_ms / 1000)

    audio_queue = queue.Queue(maxsize=4)
    ring_buffer = np.zeros((window_samples_native, channels), dtype='float32')
    callback_count = [0]
    last_detection_time = [0.0]

    def audio_callback(indata, frames, time_info, status):
        ring_buffer[:-frames] = ring_buffer[frames:]
        ring_buffer[-frames:] = indata
        callback_count[0] += 1

        if callback_count[0] < WARMUP_CALLBACKS:
            if callback_count[0] == 1:
                print("⏳ Warming up microphone buffer...")
            return

        if callback_count[0] == WARMUP_CALLBACKS:
            print("✅ Warmup complete — now listening!\n")

        # Quick energy check before queueing
        raw_copy = ring_buffer.copy()
        if channels > 1:
            mono = raw_copy.mean(axis=1)
        else:
            mono = raw_copy.flatten()

        rms = np.sqrt(np.mean(mono ** 2))
        if rms < ENERGY_GATE_THRESHOLD:
            return

        try:
            if audio_queue.full():
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    pass
            audio_queue.put_nowait((raw_copy, rms))
        except queue.Full:
            pass

    def inference_loop():
        loop_cnt = 0
        while True:
            try:
                raw_audio, rms = audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if raw_audio is None:
                break

            loop_cnt += 1
            timestamp = time.strftime("%H:%M:%S")
            timestamp_file = time.strftime("%H%M%S")
            now = time.time()

            # Run inference across all active models
            results = {}
            any_gunshot = False
            processed_wav = None

            for m in active_models:
                g_prob, a_score, wav_data = m.predict(raw_audio, native_sr)
                results[m.id] = (g_prob, a_score)
                if g_prob >= CONFIDENCE_THRESHOLD:
                    any_gunshot = True
                if processed_wav is None:
                    processed_wav = wav_data

            if any_gunshot:
                if (now - last_detection_time[0]) >= COOLDOWN_SECONDS:
                    last_detection_time[0] = now
                    print("\n" + "🚨" * 35)
                    print(f"[{timestamp}] 🔫 GUNSHOT DETECTED! (RMS: {rms:.5f})")
                    print("-" * 70)
                    for m in active_models:
                        g_prob, a_score = results[m.id]
                        anom_str = f" | Anomaly: {a_score:.4f}" if a_score is not None else ""
                        bar_len = int(g_prob * 20)
                        bar = "█" * bar_len + "░" * (20 - bar_len)
                        print(f"  ├─ {m.name:<32} : {g_prob*100:>5.1f}% [{bar}]{anom_str}")
                    print("🚨" * 35 + "\n")

                    # Log detection
                    log_entry = f"[{timestamp}] GUNSHOT DETECTED | RMS: {rms:.5f} | " + " | ".join(
                        f"{m.name}: {results[m.id][0]:.4f}" for m in active_models
                    )
                    with open(LOG_FILE, "a", encoding="utf-8") as f:
                        f.write(time.strftime("%Y-%m-%d ") + log_entry + "\n")

                    # Save audio snippet
                    if HAS_SOUNDFILE and processed_wav is not None:
                        fname = RECORDINGS_DIR / f"GUNSHOT_{timestamp_file}_rms{rms:.3f}.wav"
                        sf.write(str(fname), processed_wav, TARGET_SR)
                        print(f"   💾 Audio saved to: {fname}")

            else:
                # Periodic listening status update
                if loop_cnt % 4 == 0:
                    summary_parts = []
                    for m in active_models:
                        g_prob, _ = results[m.id]
                        summary_parts.append(f"{m.name[:12]}: {g_prob:.3f}")
                    status_line = " | ".join(summary_parts)
                    print(f"[{timestamp}] 🎧 Listening... ({status_line} | RMS: {rms:.5f})      ", end="\r", flush=True)

    # Start inference thread
    inf_thread = threading.Thread(target=inference_loop, daemon=True)
    inf_thread.start()

    print("\n🔴 LISTENING FOR GUNSHOTS... (Press Ctrl+C to stop)")
    print("-" * 70)
    print(f"⚙️  Active Models       : {len(active_models)}")
    print(f"⚙️  Confidence Threshold: {CONFIDENCE_THRESHOLD*100:.0f}%")
    print(f"⚙️  Energy Gate        : {ENERGY_GATE_THRESHOLD}")
    print(f"⚙️  Cooldown           : {COOLDOWN_SECONDS}s")
    print(f"📝 Detections Logged   : {LOG_FILE}\n")

    try:
        with sd.InputStream(device=device_id, samplerate=native_sr, channels=channels,
                            blocksize=hop_samples_native, callback=audio_callback):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        audio_queue.put((None, 0))
        inf_thread.join(timeout=2.0)
        print("\n\n🛑 Stopped listening.")


# ============================================================
# MULTI-MODEL AUDIO FILE BENCHMARK (TEST ON RECORDING)
# ============================================================
def run_file_benchmark(active_models):
    """Run sliding window inference on a recording file across all active models."""
    if not active_models:
        print("❌ No trained models available for benchmark.")
        return

    print("\n🎧 --- MULTI-MODEL AUDIO FILE BENCHMARK ---")
    print(" 1. Use default 'test_recording.wav'")
    print(" 2. Enter custom .wav file path")
    choice = input("👉 Enter choice (1 or 2): ").strip()

    if choice == '1':
        wav_path = SCRIPT_DIR.parent / 'test_recording.wav'
    else:
        raw_in = input("👉 Enter full path to .wav file: ").strip().strip('"').strip("'")
        wav_path = Path(raw_in)

    if not wav_path.exists():
        print(f"❌ Audio file not found: {wav_path}")
        return

    print(f"\n🎵 Loading: {wav_path.name}")
    if HAS_SOUNDFILE:
        raw_audio, file_sr = sf.read(str(wav_path))
    else:
        file_sr, raw_audio = signal.io.wavfile.read(str(wav_path))
        raw_audio = raw_audio.astype(np.float32)

    if raw_audio.ndim > 1:
        raw_audio = raw_audio.mean(axis=1)

    # Resample audio to TARGET_SR
    audio_22k = resample_audio(raw_audio.astype(np.float32), file_sr, TARGET_SR)
    total_samples = len(audio_22k)
    duration_sec = total_samples / TARGET_SR

    print(f"   Original SR: {file_sr} Hz | Duration: {duration_sec:.2f}s | Samples: {total_samples}")

    # Sliding window settings
    max_samples = max(m.input_samples for m in active_models)
    clip_ms = int(max_samples / TARGET_SR * 1000)
    hop_ms = 125  # 125ms slide
    hop_samples = int(TARGET_SR * hop_ms / 1000)
    n_windows = max(1, (total_samples - max_samples) // hop_samples + 1)

    print(f"\n🔍 Running Benchmark across {len(active_models)} trained models...")
    print(f"   Windows: {n_windows} ({clip_ms}ms window, {hop_ms}ms hop)\n")

    # Header
    col_names = " | ".join(f"{m.name[:18]:<18}" for m in active_models)
    print(f" {'Time (ms)':<15} | {col_names} | {'RMS Energy':<10}")
    print("-" * (30 + len(active_models) * 21))

    model_detections = {m.id: 0 for m in active_models}

    for i in range(n_windows):
        start = i * hop_samples
        end = start + max_samples
        window = audio_22k[start:end].copy()

        rms = np.sqrt(np.mean(window ** 2))
        t_start = start / TARGET_SR * 1000
        t_end = end / TARGET_SR * 1000

        row_scores = []
        for m in active_models:
            g_prob, a_score, _ = m.predict(window, TARGET_SR)
            if g_prob >= CONFIDENCE_THRESHOLD:
                model_detections[m.id] += 1
                flag = "🔫"
            else:
                flag = "  "
            row_scores.append((g_prob, flag, m))

        score_str = " | ".join(f"{flag} {g_prob:.4f}           "[:18] for g_prob, flag, m in row_scores)
        print(f" [{t_start:6.0f}-{t_end:6.0f}ms] | {score_str} | {rms:.5f}")

        # Save processed window wav for reference
        if HAS_SOUNDFILE:
            sf.write(str(TEST_OUTPUT_DIR / f'window_{i:03d}.wav'), window, TARGET_SR)

    print("\n" + "=" * 70)
    print("📊 MULTI-MODEL BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    for m in active_models:
        dets = model_detections[m.id]
        print(f" ├─ {m.name:<32} : {dets:>3} window detection(s) >= {CONFIDENCE_THRESHOLD*100:.0f}%")
    print("=" * 70)
    if HAS_SOUNDFILE:
        print(f"💾 Processed window clips saved to: {TEST_OUTPUT_DIR}/\n")


# ============================================================
# QUICK RECORD & BENCHMARK
# ============================================================
def run_quick_record_benchmark(active_models, device_id, native_sr, channels):
    """Record 5 seconds on the spot and immediately run multi-model benchmark."""
    if not HAS_SOUNDDEVICE:
        print("❌ sounddevice package is required for recording.")
        return

    duration = 5
    print(f"\n🎤 RECORDING {duration} SECONDS... (Make a sound / play gunshot audio now!)")

    recording = sd.rec(int(duration * native_sr), samplerate=native_sr, channels=channels, device=device_id, dtype='float32')
    sd.wait()
    print("✅ Recording complete!")

    wav_path = SCRIPT_DIR.parent / 'test_recording.wav'
    if HAS_SOUNDFILE:
        sf.write(str(wav_path), recording, native_sr)
    print(f"💾 Saved recording to: {wav_path}")

    # Run benchmark on it
    run_file_benchmark(active_models)


# ============================================================
# SENSITIVITY CONFIGURATION MENU
# ============================================================
def configure_sensitivity():
    global CONFIDENCE_THRESHOLD, ENERGY_GATE_THRESHOLD, COOLDOWN_SECONDS
    print("\n⚡ --- SENSITIVITY PRESETS ---")
    print(f" Current Settings: Threshold={CONFIDENCE_THRESHOLD*100:.0f}%, EnergyGate={ENERGY_GATE_THRESHOLD}, Cooldown={COOLDOWN_SECONDS}s\n")
    print(" [1] ⚡ High Sensitivity (For Phone / Mic Testing: Threshold 50%, Gate 0.001, Cooldown 0.4s)")
    print(" [2] 🛡️ Standard Mode    (For Real Field Detection: Threshold 70%, Gate 0.005, Cooldown 1.0s)")
    print(" [3] 🎯 Ultra High       (For Quiet Phone Playback: Threshold 35%, Gate 0.0005, Cooldown 0.2s)")
    print(" [4] 🔧 Custom Threshold")
    
    choice = input("👉 Select preset (1-4): ").strip()
    if choice == '1':
        CONFIDENCE_THRESHOLD = 0.50
        ENERGY_GATE_THRESHOLD = 0.001
        COOLDOWN_SECONDS = 0.4
        print("✅ Configured for High Sensitivity Phone Testing!")
    elif choice == '2':
        CONFIDENCE_THRESHOLD = 0.70
        ENERGY_GATE_THRESHOLD = 0.005
        COOLDOWN_SECONDS = 1.0
        print("✅ Configured for Standard Field Detection!")
    elif choice == '3':
        CONFIDENCE_THRESHOLD = 0.35
        ENERGY_GATE_THRESHOLD = 0.0005
        COOLDOWN_SECONDS = 0.2
        print("✅ Configured for Ultra High Sensitivity!")
    elif choice == '4':
        try:
            val = float(input("👉 Enter threshold percentage (10-95): ").strip())
            CONFIDENCE_THRESHOLD = max(0.10, min(0.95, val / 100.0))
            print(f"✅ Confidence Threshold set to {CONFIDENCE_THRESHOLD*100:.0f}%")
        except ValueError:
            print("❌ Invalid number.")


# ============================================================
# MAIN INTERACTIVE MENU
# ============================================================
def main():
    print("\n" + "=" * 80)
    print("🎯 SHOOT_CATCHER — MULTI-MODEL GUNSHOT INTELLIGENCE HUB")
    print("=" * 80)

    # Initialize Model Manager & Audit
    manager = ModelManager()
    manager.print_audit_table()

    if not manager.trained_models:
        print("⚠️ WARNING: No trained model .h5 files were found!")
        print("   Please train models using notebooks in 01_1D_CNN, 02_2D_CNN, or Enhanced_Models.")
        print("   You can still inspect module paths or record test audio.\n")

    # Mic Setup
    device_id, native_sr, channels = None, None, None

    while True:
        print(f"\n⚙️  MAIN MENU (Sensitivity: {CONFIDENCE_THRESHOLD*100:.0f}% threshold, {COOLDOWN_SECONDS}s cooldown):")
        print(" ─────────────────────────────────────────────────────────────")
        print(" [1] 🚀 Run Multi-Model Live Real-Time Dashboard (All Trained Models)")
        print(" [2] 🎯 Run Single-Model Live Microphone Stream")
        print(" [3] 🎵 Run Multi-Model Benchmark on Audio File (.wav)")
        print(" [4] 🎙️ Quick Record 5s & Run Multi-Model Benchmark")
        print(" [5] ⚡ Change Sensitivity Preset (Phone Testing vs Standard)")
        print(" [6] 📋 View Model Status Audit & Architecture Info")
        print(" [0] 🚪 Exit")
        print(" ─────────────────────────────────────────────────────────────")

        choice = input("👉 Enter choice (0-6): ").strip()

        if choice == '1':
            if not manager.trained_models:
                print("❌ No trained models available.")
                continue
            if device_id is None:
                device_id, native_sr, channels = find_working_mic()
                if device_id is None:
                    continue
            run_live_monitoring(manager.trained_models, device_id, native_sr, channels, is_multi=True)

        elif choice == '2':
            if not manager.trained_models:
                print("❌ No trained models available.")
                continue
            print("\nSelect model to monitor:")
            for idx, m in enumerate(manager.trained_models, 1):
                print(f" [{idx}] {m.name}")
            c_idx = input("👉 Choice: ").strip()
            try:
                sel_model = manager.trained_models[int(c_idx) - 1]
                if device_id is None:
                    device_id, native_sr, channels = find_working_mic()
                    if device_id is None:
                        continue
                run_live_monitoring([sel_model], device_id, native_sr, channels, is_multi=False)
            except (ValueError, IndexError):
                print("❌ Invalid selection.")

        elif choice == '3':
            if not manager.trained_models:
                print("❌ No trained models available.")
                continue
            run_file_benchmark(manager.trained_models)

        elif choice == '4':
            if not manager.trained_models:
                print("❌ No trained models available.")
                continue
            if device_id is None:
                device_id, native_sr, channels = find_working_mic()
                if device_id is None:
                    continue
            run_quick_record_benchmark(manager.trained_models, device_id, native_sr, channels)

        elif choice == '5':
            configure_sensitivity()

        elif choice == '6':
            manager.print_audit_table()

        elif choice == '0':
            print("\n👋 Exiting Shoot_Catcher Hub. Goodbye!")
            sys.exit(0)
        else:
            print("❌ Invalid choice. Try again.")


if __name__ == "__main__":
    main()
