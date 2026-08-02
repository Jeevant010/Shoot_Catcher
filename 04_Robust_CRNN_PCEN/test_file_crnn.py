"""
===============================================================================
🛡️ Shoot_Catcher — Module 04: Isolated CRNN File Benchmark Tester
===============================================================================
Single-purpose offline benchmark runner for the Robust CRNN-PCEN model.
Evaluates audio recording files frame-by-frame and exports CSV reports.
===============================================================================
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
import numpy as np
import scipy.signal as signal
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except Exception:
    sf = None
    HAS_SOUNDFILE = False

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(SCRIPT_DIR))
from pcen_mic_pipeline import compute_pcen

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("CRNN_File_Benchmark")

TARGET_SR = 22050
TARGET_SAMPLES = 16537
MODEL_PATH = SCRIPT_DIR / "output" / "crnn_pcen_best.h5"
STATS_PATH = SCRIPT_DIR / "output" / "pcen_stats.json"
TEST_OUTPUT_DIR = SCRIPT_DIR / "output" / "benchmark_results"
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class CompatBatchNormalization(keras.layers.BatchNormalization):
    def __init__(self, **kwargs):
        kwargs.pop('renorm', None)
        kwargs.pop('renorm_clipping', None)
        kwargs.pop('renorm_momentum', None)
        super().__init__(**kwargs)


def run_benchmark(wav_path, threshold=0.50, hop_ms=125):
    if not MODEL_PATH.exists():
        logger.error(f"Model not found at {MODEL_PATH}")
        sys.exit(1)

    model = keras.models.load_model(
        str(MODEL_PATH),
        custom_objects={'BatchNormalization': CompatBatchNormalization},
        compile=False
    )
    norm_mean, norm_std = -37.6324, 20.3542
    if STATS_PATH.exists():
        try:
            stats = json.loads(STATS_PATH.read_text())
            norm_mean, norm_std = stats.get('mean', norm_mean), stats.get('std', norm_std)
        except Exception:
            pass

    logger.info(f"Loaded Robust CRNN PCEN Model")
    logger.info(f"Processing File: {wav_path.name}")

    if HAS_SOUNDFILE:
        raw_audio, file_sr = sf.read(str(wav_path))
    else:
        file_sr, raw_audio = signal.io.wavfile.read(str(wav_path))
        raw_audio = raw_audio.astype(np.float32)

    if raw_audio.ndim > 1:
        raw_audio = raw_audio.mean(axis=1)

    # Resample
    if file_sr != TARGET_SR:
        num_samples = int(len(raw_audio) * TARGET_SR / file_sr)
        y_all = signal.resample(raw_audio, num_samples).astype(np.float32)
    else:
        y_all = raw_audio.astype(np.float32)

    total_samples = len(y_all)
    hop_samples = int(TARGET_SR * hop_ms / 1000.0)
    n_windows = max(1, (total_samples - TARGET_SAMPLES) // hop_samples + 1)

    logger.info(f"Running sliding window inference across {n_windows} windows (750ms window, {hop_ms}ms hop)...")

    results = []
    detections = 0

    for i in range(n_windows):
        start = i * hop_samples
        end = start + TARGET_SAMPLES
        window = y_all[start:end].copy()

        if len(window) < TARGET_SAMPLES:
            window = np.pad(window, (0, TARGET_SAMPLES - len(window)))

        rms = np.sqrt(np.mean(window ** 2))
        dbfs = 20.0 * np.log10(max(rms, 1e-8))

        peak = np.max(np.abs(window))
        if peak > 1e-6:
            window = window / peak

        pcen = compute_pcen(window, sr=TARGET_SR, n_mels=64, n_fft=512, hop_length=128)
        pcen_norm = (pcen - norm_mean) / max(norm_std, 1e-6)
        x = pcen_norm.reshape(1, pcen_norm.shape[0], pcen_norm.shape[1], 1).astype(np.float32)

        prob = float(model.predict(x, verbose=0).flatten()[0])
        is_det = prob >= threshold
        if is_det: detections += 1

        t_start = start / TARGET_SR * 1000.0
        t_end = end / TARGET_SR * 1000.0

        results.append({
            'window_idx': i,
            'time_start_ms': round(t_start, 1),
            'time_end_ms': round(t_end, 1),
            'confidence': round(prob, 4),
            'detected': is_det,
            'rms_energy': round(rms, 6),
            'dbfs_peak': round(dbfs, 2)
        })

    df = pd.DataFrame(results)
    csv_out = TEST_OUTPUT_DIR / f"{wav_path.stem}_benchmark.csv"
    df.to_csv(csv_out, index=False)

    logger.info("=" * 60)
    logger.info(f"BENCHMARK SUMMARY FOR {wav_path.name}")
    logger.info("=" * 60)
    logger.info(f"Total Windows Tested : {n_windows}")
    logger.info(f"Gunshot Detections   : {detections} (>= {threshold*100:.0f}%)")
    logger.info(f"Detailed CSV Report  : {csv_out}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Shoot_Catcher — Module 04 CRNN File Benchmark")
    parser.add_argument("--input", type=str, default=None, help="Path to input .wav file")
    parser.add_argument("--threshold", type=float, default=0.50, help="Confidence threshold")
    args = parser.parse_args()

    if args.input is None:
        default_file = SCRIPT_DIR.parent / "test_recording.wav"
        if default_file.exists():
            wav_path = default_file
        else:
            logger.error("No input file provided. Use --input path/to/file.wav")
            sys.exit(1)
    else:
        wav_path = Path(args.input)

    run_benchmark(wav_path, threshold=args.threshold)


if __name__ == "__main__":
    main()
