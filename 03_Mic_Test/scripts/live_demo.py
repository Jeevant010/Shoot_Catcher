"""
===============================================================================
🎯 Shoot_Catcher — Multi-Model Gunshot Intelligence & Live Dashboard Hub
===============================================================================
Robust, real-time live monitoring and benchmark tool supporting all 5 model modules:
  1. Baseline 1D CNN           (01_1D_CNN)
  2. Baseline 2D CNN           (02_2D_CNN - Mel Spectrogram)
  3. Robust CRNN PCEN          (04_Robust_CRNN_PCEN - Domain Resilient)
  4. Enhanced 1D CNN           (Enhanced_Models/01_Enhanced_1D_CNN - Dual Head)
  5. Enhanced 2D CNN           (Enhanced_Models/02_Enhanced_2D_CNN - Dual Head)

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
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs

import tensorflow as tf
from tensorflow import keras
tf.get_logger().setLevel('ERROR')
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

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

# Try importing librosa safely (fallback to scipy/numpy if unavailable)
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

# Audio & Threshold Parameters
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
# 5 MODEL MODULE SPECIFICATIONS
# ============================================================
MODULE_SPECS = [
    {
        'id': '01_1d_cnn',
        'name': 'Baseline 1D CNN',
        'module': '01_1D_CNN',
        'relative_path': Path('01_1D_CNN/output/1d_cnn_best.h5'),
        'fallback_path': Path('01_1D_CNN/output/1d_cnn_gunshot_detector.h5'),
        'feature_type': 'raw_1d',
        'is_dual_head': False,
        'input_samples': 16537,
    },
    {
        'id': '02_2d_cnn',
        'name': 'Baseline 2D CNN (Mel)',
        'module': '02_2D_CNN',
        'relative_path': Path('02_2D_CNN/output/2d_cnn_mel_spectrogram_best.h5'),
        'fallback_path': Path('02_2D_CNN/output/2d_cnn_mel_spectrogram_detector.h5'),
        'norm_stats_path': Path('02_2D_CNN/output/2d_cnn_norm_stats.json'),
        'feature_type': 'mel_2d',
        'mel_params': {'n_mels': 64, 'n_fft': 512, 'hop_length': 128, 'fmin': 0.0, 'fmax': None},
        'time_steps': 130,
        'is_dual_head': False,
        'input_samples': 16537,
    },
    {
        'id': '04_robust_crnn',
        'name': 'Robust CRNN (PCEN)',
        'module': '04_Robust_CRNN_PCEN',
        'relative_path': Path('04_Robust_CRNN_PCEN/output/crnn_pcen_best.h5'),
        'norm_stats_path': Path('04_Robust_CRNN_PCEN/output/pcen_stats.json'),
        'feature_type': 'pcen_2d',
        'mel_params': {'n_mels': 64, 'n_fft': 512, 'hop_length': 128},
        'time_steps': 130,
        'is_dual_head': False,
        'input_samples': 16537,
    },
    {
        'id': '01_enhanced_1d',
        'name': 'Enhanced 1D CNN (Dual)',
        'module': 'Enhanced_Models/01_Enhanced_1D_CNN',
        'relative_path': Path('Enhanced_Models/01_Enhanced_1D_CNN/output/enhanced_1d_cnn_best.h5'),
        'feature_type': 'raw_1d',
        'is_dual_head': True,
        'input_samples': 16537,
    },
    {
        'id': '02_enhanced_2d',
        'name': 'Enhanced 2D CNN (Dual)',
        'module': 'Enhanced_Models/02_Enhanced_2D_CNN',
        'relative_path': Path('Enhanced_Models/02_Enhanced_2D_CNN/output/enhanced_2d_cnn_best.h5'),
        'feature_type': 'mel_2d',
        'mel_params': {'n_mels': 64, 'n_fft': 2048, 'hop_length': 512, 'fmin': 20.0, 'fmax': 8000.0},
        'time_steps': 33,
        'is_dual_head': True,
        'input_samples': 16537,
    },
]


# ============================================================
# SPECTROGRAM & PCEN AUDIO FEATURE COMPUTATION
# ============================================================
def compute_mel_spectrogram_scipy(y, sr=22050, n_mels=64, n_fft=512, hop_length=128, fmin=0.0, fmax=None):
    """Compute normalized log-Mel Spectrogram [0.0, 1.0] using pure Scipy/NumPy."""
    if fmax is None:
        fmax = sr / 2.0

    f, t, Zxx = signal.stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length, boundary=None)
    power = np.abs(Zxx) ** 2

    mel_low = 2595 * np.log10(1 + max(0.0, fmin) / 700.0)
    mel_high = 2595 * np.log10(1 + fmax / 700.0)
    mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = 700.0 * (10 ** (mel_points / 2595.0) - 1.0)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bin_points = np.clip(bin_points, 0, n_fft // 2)

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
    spec_norm = (log_mel + 80.0) / 80.0
    return np.clip(spec_norm, 0.0, 1.0).astype(np.float32)


def compute_mel_spectrogram(y, sr=22050, n_mels=64, n_fft=512, hop_length=128, fmin=0.0, fmax=None):
    """Compute normalized log-Mel-spectrogram [0.0, 1.0] using librosa or scipy fallback."""
    if HAS_LIBROSA:
        try:
            S = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=n_mels, n_fft=n_fft,
                hop_length=hop_length, fmin=fmin, fmax=fmax
            )
            S_dB = librosa.power_to_db(S, ref=np.max)
            S_norm = (S_dB + 80.0) / 80.0
            return np.clip(S_norm, 0.0, 1.0).astype(np.float32)
        except Exception:
            pass
    return compute_mel_spectrogram_scipy(y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, fmin=fmin, fmax=fmax)


def compute_pcen_scipy(y, sr=22050, n_mels=64, n_fft=512, hop_length=128,
                       s=0.025, alpha=0.98, delta=2.0, r=0.5, eps=1e-6):
    """Pure NumPy / Scipy implementation of Per-Channel Energy Normalization (PCEN)."""
    f, t, Zxx = signal.stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length, boundary=None)
    power = np.abs(Zxx) ** 2

    low_freq, high_freq = 0.0, sr / 2.0
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

    S = np.dot(fb, power).astype(np.float32)
    M = signal.lfilter([s], [1.0, -(1.0 - s)], S, axis=-1)
    smooth = (eps + M) ** alpha
    normalized = S / smooth
    pcen = (normalized + delta) ** r - (delta ** r)
    return pcen.astype(np.float32)


def compute_pcen(y, sr=22050, n_mels=64, n_fft=512, hop_length=128):
    """Computes PCEN feature matrix using librosa or scipy fallback."""
    if HAS_LIBROSA:
        try:
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, power=1)
            pcen = librosa.pcen(S * (2**31), sr=sr, hop_length=hop_length, time_constant=0.025, gain=0.98, bias=2.0, power=0.5)
            return pcen.astype(np.float32)
        except Exception:
            pass
    return compute_pcen_scipy(y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)


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
        self.feature_type = spec.get('feature_type', 'raw_1d')
        self.is_dual_head = spec.get('is_dual_head', False)
        self.input_samples = spec.get('input_samples', 16537)
        self.mel_params = spec.get('mel_params', {})
        self.time_steps = spec.get('time_steps', 130)

        self.model_path = PROJECT_ROOT / spec['relative_path']
        if not self.model_path.exists() and 'fallback_path' in spec:
            fallback = PROJECT_ROOT / spec['fallback_path']
            if fallback.exists():
                self.model_path = fallback

        self.norm_stats_path = PROJECT_ROOT / spec['norm_stats_path'] if 'norm_stats_path' in spec else None

        self.model = None
        self.status = "NOT TRAINED / FILE MISSING"
        self.norm_mean = -37.6324 if self.feature_type == 'pcen_2d' else 0.0
        self.norm_std = 20.3542 if self.feature_type == 'pcen_2d' else 1.0

        # Load normalization stats if available
        if self.norm_stats_path and self.norm_stats_path.exists():
            try:
                stats = json.loads(self.norm_stats_path.read_text())
                self.norm_mean = stats.get('mean', self.norm_mean)
                self.norm_std = stats.get('std', self.norm_std)
            except Exception:
                pass

        # Attempt to load model with compile=False
        if self.model_path.exists():
            try:
                self.model = keras.models.load_model(
                    str(self.model_path),
                    custom_objects={'BatchNormalization': CompatBatchNormalization},
                    compile=False
                )
                self.status = "TRAINED & LOADED"

                # Check dual head output
                if hasattr(self.model, 'outputs') and len(self.model.outputs) == 2:
                    self.is_dual_head = True
            except Exception as e:
                self.status = f"ERROR LOADING ({str(e)[:30]}...)"
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

        # Remove DC offset (Hardware Mic Bias Removal)
        y = y - np.mean(y)

        # Resample to TARGET_SR
        y = resample_audio(y, native_sr, TARGET_SR)

        # Force exact sample length
        if len(y) >= self.input_samples:
            y = y[:self.input_samples]
        else:
            y = np.pad(y, (0, self.input_samples - len(y)))

        # Energy-Gated Peak Normalization
        rms = np.sqrt(np.mean(y ** 2))
        if rms < 0.0001:  # Near-zero digital silence gate
            return 0.0, (0.0 if self.is_dual_head else None), y

        if rms >= ENERGY_GATE_THRESHOLD:
            peak = np.max(np.abs(y))
            if peak > 1e-6:
                y = y / peak
        else:
            y = np.clip(y, -1.0, 1.0)

        # Feature preparation based on model type
        if self.feature_type == 'raw_1d':
            x = y.reshape(1, self.input_samples, 1).astype(np.float32)

        elif self.feature_type == 'mel_2d':
            n_mels = self.mel_params.get('n_mels', 64)
            n_fft = self.mel_params.get('n_fft', 512)
            hop_length = self.mel_params.get('hop_length', 128)
            fmin = self.mel_params.get('fmin', 0.0)
            fmax = self.mel_params.get('fmax', None)

            spec = compute_mel_spectrogram(y, sr=TARGET_SR, n_mels=n_mels, n_fft=n_fft,
                                           hop_length=hop_length, fmin=fmin, fmax=fmax)

            # Pad or truncate time_steps
            if spec.shape[1] < self.time_steps:
                spec = np.pad(spec, ((0, 0), (0, self.time_steps - spec.shape[1])))
            else:
                spec = spec[:, :self.time_steps]

            x = spec.reshape(1, n_mels, self.time_steps, 1).astype(np.float32)

        elif self.feature_type == 'pcen_2d':
            n_mels = self.mel_params.get('n_mels', 64)
            n_fft = self.mel_params.get('n_fft', 512)
            hop_length = self.mel_params.get('hop_length', 128)

            pcen = compute_pcen(y, sr=TARGET_SR, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)

            if pcen.shape[1] < self.time_steps:
                pcen = np.pad(pcen, ((0, 0), (0, self.time_steps - pcen.shape[1])))
            else:
                pcen = pcen[:, :self.time_steps]

            # PCEN Standardized Z-Score Normalization
            pcen_norm = (pcen - self.norm_mean) / max(self.norm_std, 1e-6)
            x = pcen_norm.reshape(1, n_mels, self.time_steps, 1).astype(np.float32)

        else:
            x = y.reshape(1, -1, 1).astype(np.float32)

        # Execute prediction
        out = self.model.predict(x, verbose=0)
        
        # Handle dict, list, or single array outputs cleanly
        if isinstance(out, dict):
            gunshot_prob = float(out.get('gunshot_output', list(out.values())[0]).flatten()[0])
            anomaly_score = float(out.get('anomaly_output', list(out.values())[1]).flatten()[0]) if len(out) > 1 else None
        elif isinstance(out, (list, tuple)) and len(out) == 2:
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
        print("🔍 Scanning for trained models across all 5 modules...")
        self.models = [ModelWrapper(spec) for spec in MODULE_SPECS]
        self.trained_models = [m for m in self.models if m.is_available]

    def print_audit_table(self):
        print("\n" + "=" * 85)
        print("📋 MODULE & MODEL STATUS AUDIT (ALL 5 MODELS)")
        print("=" * 85)
        print(f" {'#':<3} {'Model / Module Name':<35} {'Status':<25} {'Architecture / Input':<18}")
        print("-" * 85)
        for i, m in enumerate(self.models, 1):
            if m.feature_type == 'pcen_2d':
                type_str = "2D PCEN (CRNN)"
            elif m.feature_type == 'mel_2d':
                type_str = "2D Mel (Dual)" if m.is_dual_head else "2D Mel Spec"
            else:
                type_str = "1D Wave (Dual)" if m.is_dual_head else "1D Raw Wave"

            status_color = "✅ " + m.status if m.is_available else "⚠️  " + m.status
            print(f" [{i}] {m.name:<35} {status_color:<25} {type_str:<18}")
        print("=" * 85)
        print(f" Total Modules: {len(self.models)} | Trained & Active: {len(self.trained_models)} | Untrained: {len(self.models) - len(self.trained_models)}\n")


# ============================================================
# MICROPHONE HELPER WITH LIVE SIGNAL PROBING
# ============================================================
def find_working_mic():
    if not HAS_SOUNDDEVICE:
        print("\n❌ Python package 'sounddevice' is not installed.")
        print("   Run: pip install sounddevice")
        return None, None, None

    devices = sd.query_devices()
    valid_mics = []

    print("\n🎤 --- SCANNING & PROBING AUDIO INPUT DEVICES ---")
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            name = d['name'].strip()
            try:
                native_sr = int(d['default_samplerate'])
                channels = min(d['max_input_channels'], 2)
                # Quick 150ms test to check live signal activity
                test = sd.rec(int(0.15 * native_sr), samplerate=native_sr, channels=channels, device=i, dtype='float32')
                sd.wait()
                peak = float(np.max(np.abs(test)))
                rms = float(np.sqrt(np.mean(test ** 2)))
                status = "🔊 ACTIVE SIGNAL" if rms > 0.0001 else "⚠️  SILENT / MUTED"
                valid_mics.append((i, name, native_sr, channels, rms, peak))
                print(f" [{len(valid_mics)}] {name:<40} ({native_sr}Hz) | {status} (RMS: {rms:.5f})")
            except Exception:
                pass

    if not valid_mics:
        print("❌ No working input microphones found.")
        return None, None, None

    # Pick recommended index (highest active signal)
    best_idx = 0
    max_rms = -1.0
    for idx, (_, _, _, _, rms, _) in enumerate(valid_mics):
        if rms > max_rms:
            max_rms = rms
            best_idx = idx

    print("--------------------------------------------------------------------------------")
    print(f"💡 TIP: Select device [{best_idx+1}] which has the strongest detected signal.")
    print("   If testing PC YouTube playback directly on laptop, select 'Stereo Mix' if available.")
    print("--------------------------------------------------------------------------------")

    while True:
        try:
            choice = input(f"👉 Select a microphone (1-{len(valid_mics)}, default {best_idx+1}): ").strip()
            if not choice:
                idx = best_idx
            else:
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

    max_samples = max(m.input_samples for m in active_models)
    clip_duration_ms = int(max_samples / TARGET_SR * 1000)
    hop_ms = int(clip_duration_ms * 0.25)  # 75% overlap

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
                    print("\n" + "🚨" * 38)
                    print(f"[{timestamp}] 🔫 GUNSHOT DETECTED! (Signal RMS: {rms:.5f})")
                    print("-" * 76)
                    for m in active_models:
                        g_prob, a_score = results[m.id]
                        anom_str = f" | Anomaly: {a_score:.4f}" if a_score is not None else ""
                        bar_len = int(g_prob * 20)
                        bar = "█" * bar_len + "░" * (20 - bar_len)
                        flag = "🔥" if g_prob >= CONFIDENCE_THRESHOLD else "  "
                        print(f"  {flag} {m.name:<30} : {g_prob*100:>5.1f}% [{bar}]{anom_str}")
                    print("🚨" * 38 + "\n")

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
                        print(f"   💾 Audio clip saved to: {fname}")

            else:
                # Periodic listening status update
                if loop_cnt % 4 == 0:
                    summary_parts = []
                    for m in active_models:
                        g_prob, _ = results[m.id]
                        summary_parts.append(f"{m.name[:10]}:{g_prob*100:.0f}%")
                    status_line = " | ".join(summary_parts)
                    print(f"[{timestamp}] 🎧 Listening... ({status_line} | RMS: {rms:.5f})      ", end="\r", flush=True)

    inf_thread = threading.Thread(target=inference_loop, daemon=True)
    inf_thread.start()

    print("\n🔴 LIVE MONITORING RUNNING... (Press Ctrl+C to return to menu)")
    print("-" * 76)
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
def run_file_benchmark(active_models, wav_path=None):
    """Run sliding window inference on a recording file across all active models."""
    if not active_models:
        print("❌ No trained models available for benchmark.")
        return

    if wav_path is None:
        print("\n🎧 --- MULTI-MODEL AUDIO FILE BENCHMARK (ALL 5 MODELS) ---")
        print(" [1] Use default 'test_recording.wav'")
        print(" [2] Enter custom .wav file path")
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

    max_samples = max(m.input_samples for m in active_models)
    clip_ms = int(max_samples / TARGET_SR * 1000)
    hop_ms = 125   # Sliding window hop
    hop_samples = int(TARGET_SR * hop_ms / 1000)
    n_windows = max(1, (total_samples - max_samples) // hop_samples + 1)

    print(f"\n🔍 Running Benchmark across {len(active_models)} active models...")
    print(f"   Total Windows: {n_windows} ({clip_ms}ms window, {hop_ms}ms hop)\n")

    # Dynamic Column Formatter
    col_width = 16
    col_headers = " | ".join(f"{m.name[:col_width]:<{col_width}}" for m in active_models)
    print(f" {'Time Range (ms)':<16} | {col_headers} | {'RMS Energy':<10}")
    print("-" * (30 + len(active_models) * (col_width + 3)))

    # Track model detection counts and detection event timestamps
    model_detections = {m.id: 0 for m in active_models}
    detected_event_times = []

    for i in range(n_windows):
        start = i * hop_samples
        end = start + max_samples
        window = audio_22k[start:end].copy()

        rms = np.sqrt(np.mean(window ** 2))
        t_start = start / TARGET_SR * 1000
        t_end = end / TARGET_SR * 1000

        row_scores = []
        any_model_triggered = False
        for m in active_models:
            g_prob, a_score, _ = m.predict(window, TARGET_SR)
            if g_prob >= CONFIDENCE_THRESHOLD:
                model_detections[m.id] += 1
                flag = "🔫"
                # Exclude enhanced 1D from alone triggering event saving if it's always firing
                if m.id != '01_enhanced_1d':
                    any_model_triggered = True
            else:
                flag = "  "
            row_scores.append((g_prob, flag, m))

        if any_model_triggered:
            detected_event_times.append((start + end) / 2.0)

        score_cells = [f"{flag}{g_prob*100:>5.1f}%".center(col_width) for g_prob, flag, m in row_scores]
        score_str = " | ".join(score_cells)
        print(f" [{t_start:5.0f}-{t_end:5.0f}ms] | {score_str} | {rms:.5f}")

    print("\n" + "=" * 80)
    print("📊 5-MODEL BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    models_triggered = 0
    for m in active_models:
        dets = model_detections[m.id]
        if dets > 0:
            models_triggered += 1
            status_icon = "🔥 DETECTED"
        else:
            status_icon = "⚪ NO TRIGGER"
        print(f" ├─ {m.name:<30} : {dets:>3} detections | {status_icon}")
    print("-" * 80)
    if models_triggered >= 2:
        print(f" 🚨 OVERALL VERDICT: 🔫 CONFIRMED GUNSHOT DETECTED! ({models_triggered}/{len(active_models)} models triggered)")
    elif models_triggered == 1:
        print(f" ⚠️  OVERALL VERDICT: ⚠️ POSSIBLE GUNSHOT / ANOMALY (1/{len(active_models)} model triggered)")
    else:
        print(" ⚪ OVERALL VERDICT: ⚪ NO GUNSHOT DETECTED")
    print("=" * 80)

    # Save amplified full audio & detected event clips
    if HAS_SOUNDFILE:
        # Boost volume to 90% peak so user can clearly hear it
        peak = np.max(np.abs(audio_22k))
        if peak > 1e-6:
            amplified_audio = (audio_22k / peak) * 0.90
        else:
            amplified_audio = audio_22k

        full_save_path = TEST_OUTPUT_DIR / "full_recording_amplified.wav"
        sf.write(str(full_save_path), amplified_audio, TARGET_SR)
        print(f"💾 Full Amplified Audio saved to: {full_save_path}")

        # Extract and save 2.5s audio clips around detected events
        if detected_event_times:
            # Cluster nearby detection times
            clusters = []
            curr_cluster = [detected_event_times[0]]
            for t in detected_event_times[1:]:
                if t - curr_cluster[-1] < TARGET_SR * 1.5:
                    curr_cluster.append(t)
                else:
                    clusters.append(int(np.mean(curr_cluster)))
                    curr_cluster = [t]
            if curr_cluster:
                clusters.append(int(np.mean(curr_cluster)))

            for idx, center_sample in enumerate(clusters, 1):
                c_start = max(0, center_sample - int(TARGET_SR * 1.0))
                c_end = min(total_samples, center_sample + int(TARGET_SR * 1.5))
                clip = audio_22k[c_start:c_end]
                c_peak = np.max(np.abs(clip))
                if c_peak > 1e-6:
                    clip = (clip / c_peak) * 0.90
                c_path = TEST_OUTPUT_DIR / f"gunshot_event_{idx:02d}_{center_sample/TARGET_SR:.1f}s.wav"
                sf.write(str(c_path), clip, TARGET_SR)
                print(f"💾 Gunshot Event Clip [{center_sample/TARGET_SR:.1f}s] saved to: {c_path}")

    # Offer to play back in speakers
    if HAS_SOUNDDEVICE and sys.stdin.isatty():
        try:
            play_choice = input("\n🔊 Would you like to PLAY BACK the recorded audio now? (Y/n): ").strip().lower()
            if play_choice != 'n':
                peak = np.max(np.abs(audio_22k))
                play_data = (audio_22k / peak * 0.90) if peak > 1e-6 else audio_22k
                print("🔊 Playing back audio in speakers...")
                sd.play(play_data, TARGET_SR)
                sd.wait()
                print("✅ Playback finished.")
        except (EOFError, KeyboardInterrupt, Exception):
            pass


# ============================================================
# INTERACTIVE START / STOP GUNSHOT TEST
# ============================================================
def run_manual_start_stop_test(active_models, device_id, native_sr, channels):
    """
    User presses Enter to START recording/listening, plays gunshot audio,
    then presses Enter to STOP. Immediately evaluates all 5 models.
    """
    if not HAS_SOUNDDEVICE:
        print("❌ sounddevice package is required for recording.")
        return

    print("\n" + "=" * 80)
    print("🎙️ MANUAL START / STOP GUNSHOT DETECTION TEST (ALL 5 MODELS)")
    print("=" * 80)
    print(" 1. Press ENTER to START listening & recording.")
    print(" 2. Play your gunshot sound effect (from phone / YouTube / speakers).")
    print(" 3. Press ENTER when finished to STOP and view 5-model detection results.")
    print("=" * 80)

    input("\n👉 Press [ENTER] to START listening...")

    audio_chunks = []
    stop_event = threading.Event()
    start_time = time.time()

    def record_callback(indata, frames, time_info, status):
        if not stop_event.is_set():
            audio_chunks.append(indata.copy())

    stream = sd.InputStream(
        device=device_id,
        samplerate=native_sr,
        channels=channels,
        dtype='float32',
        callback=record_callback
    )

    print("\n🔴 LISTENING & RECORDING STARTED!")
    print("   👉 Play your gunshot sound now...")
    print("   👉 Press [ENTER] on your keyboard when finished to STOP...")

    stream.start()

    # Background thread to wait for user to press ENTER
    def wait_for_enter():
        try:
            sys.stdin.readline()
        except Exception:
            pass
        stop_event.set()

    enter_thread = threading.Thread(target=wait_for_enter, daemon=True)
    enter_thread.start()

    try:
        while not stop_event.is_set():
            elapsed = time.time() - start_time
            if audio_chunks:
                latest = audio_chunks[-1]
                mono = latest.mean(axis=1) if latest.ndim > 1 else latest.flatten()
                rms = np.sqrt(np.mean(mono ** 2))
                bar_level = min(15, int(rms * 250))
                meter = "█" * bar_level + "░" * (15 - bar_level)
            else:
                rms = 0.0
                meter = "░" * 15

            print(f"\r 🔴 RECORDING [{elapsed:4.1f}s] [{meter}] Signal RMS: {rms:.5f} | Press ENTER to stop... ", end="", flush=True)
            time.sleep(0.08)
    except KeyboardInterrupt:
        stop_event.set()

    stream.stop()
    stream.close()

    print("\n\n⏹️ RECORDING STOPPED! Processing audio...")

    if not audio_chunks:
        print("❌ No audio data captured.")
        return

    full_recording = np.concatenate(audio_chunks, axis=0)
    duration = len(full_recording) / native_sr
    print(f"✅ Captured {duration:.2f} seconds of audio ({len(full_recording)} samples).")

    # Save to test_recording.wav with normalization
    wav_path = SCRIPT_DIR.parent / 'test_recording.wav'
    if HAS_SOUNDFILE:
        mono_rec = full_recording.mean(axis=1) if full_recording.ndim > 1 else full_recording.flatten()
        peak = np.max(np.abs(mono_rec))
        norm_rec = (mono_rec / peak * 0.90) if peak > 1e-6 else mono_rec
        sf.write(str(wav_path), norm_rec, native_sr)
        print(f"💾 Saved normalized test recording to: {wav_path}")

    # Immediately run 5-model benchmark on captured audio
    run_file_benchmark(active_models, wav_path=wav_path)


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
    run_file_benchmark(active_models, wav_path=wav_path)


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
    print("\n" + "=" * 85)
    print("🎯 SHOOT_CATCHER — 5-MODEL GUNSHOT INTELLIGENCE & BENCHMARK HUB")
    print("=" * 85)

    # Initialize Model Manager & Audit
    manager = ModelManager()
    manager.print_audit_table()

    if not manager.trained_models:
        print("⚠️ WARNING: No trained model .h5 files were found!")
        print("   Please train models in 01_1D_CNN, 02_2D_CNN, 04_Robust_CRNN_PCEN, or Enhanced_Models.\n")

    # Mic Setup
    device_id, native_sr, channels = None, None, None

    while True:
        print(f"\n⚙️  MAIN MENU (Sensitivity: {CONFIDENCE_THRESHOLD*100:.0f}% threshold, {COOLDOWN_SECONDS}s cooldown):")
        print(" ─────────────────────────────────────────────────────────────────")
        print(" [1] 🎯 Start / Stop Gunshot Test (Press Enter to Start -> Play Sound -> Press Enter to Stop & Evaluate)")
        print(" [2] 🚀 Continuous Real-Time Live Monitoring Stream (All 5 Models)")
        print(" [3] 🎯 Single-Model Continuous Live Stream")
        print(" [4] 🎵 Run Benchmark on Audio File (.wav)")
        print(" [5] 🎙️ Fixed 5s Quick Record & Benchmark")
        print(" [6] ⚡ Change Sensitivity Preset (Phone Testing vs Standard)")
        print(" [7] 📋 View Model Status Audit (All 5 Modules)")
        print(" [0] 🚪 Exit")
        print(" ─────────────────────────────────────────────────────────────────")

        choice = input("👉 Enter choice (0-7): ").strip()

        if choice == '1':
            if not manager.trained_models:
                print("❌ No trained models available.")
                continue
            if device_id is None:
                device_id, native_sr, channels = find_working_mic()
                if device_id is None:
                    continue
            run_manual_start_stop_test(manager.trained_models, device_id, native_sr, channels)

        elif choice == '2':
            if not manager.trained_models:
                print("❌ No trained models available.")
                continue
            if device_id is None:
                device_id, native_sr, channels = find_working_mic()
                if device_id is None:
                    continue
            run_live_monitoring(manager.trained_models, device_id, native_sr, channels, is_multi=True)

        elif choice == '3':
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

        elif choice == '4':
            if not manager.trained_models:
                print("❌ No trained models available.")
                continue
            run_file_benchmark(manager.trained_models)

        elif choice == '5':
            if not manager.trained_models:
                print("❌ No trained models available.")
                continue
            if device_id is None:
                device_id, native_sr, channels = find_working_mic()
                if device_id is None:
                    continue
            run_quick_record_benchmark(manager.trained_models, device_id, native_sr, channels)

        elif choice == '6':
            configure_sensitivity()

        elif choice == '7':
            manager.print_audit_table()

        elif choice == '0':
            print("\n👋 Exiting Shoot_Catcher Hub. Goodbye!")
            sys.exit(0)
        else:
            print("❌ Invalid choice. Try again.")


if __name__ == "__main__":
    main()
