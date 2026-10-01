"""
===============================================================================
🎯 Shoot_Catcher — 5-Model Benchmark on My_Test_Audio (Option A)
===============================================================================
Evaluates all 5 models on your external recorded audio:
  - Actual_Gunshots/ (Label: 1)
  - Like_Gunshots/   (Label: 0 - Hard Imposters)
  - Not_Gunshots/    (Label: 0 - Ambient Background)

Applies a sliding 750ms window (with 75% overlap) across each audio file.
A file is detected as a Gunshot if ANY window exceeds threshold (0.50).
===============================================================================
"""

import os
import sys
import io
import time
from pathlib import Path
import numpy as np
import soundfile as sf

# sys.stdout / sys.stderr handled by live_demo

PROJECT_ROOT = Path(r"c:\Users\aadit\Desktop\Shoot_Catcher")
SCRIPTS_DIR = PROJECT_ROOT / "03_Mic_Test" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import live_demo

TARGET_SR = 22050
WINDOW_SAMPLES = 16537  # 750ms at 22,050Hz
HOP_SAMPLES = int(WINDOW_SAMPLES * 0.25)  # 75% overlap = 25% hop (187.5ms)
THRESHOLD = 0.50

def evaluate_file_with_sliding_window(model, audio_data, sr):
    """Slide 750ms window across file. Returns max confidence across all windows."""
    # Convert to mono
    if audio_data.ndim > 1:
        y = audio_data.mean(axis=1)
    else:
        y = audio_data.flatten()

    # Resample to 22050
    y = live_demo.resample_audio(y, sr, TARGET_SR)

    # If audio is shorter than 750ms, pad it
    if len(y) < WINDOW_SAMPLES:
        y = np.pad(y, (0, WINDOW_SAMPLES - len(y)))

    max_prob = 0.0
    # Slide window
    num_windows = max(1, int(np.ceil((len(y) - WINDOW_SAMPLES) / HOP_SAMPLES)) + 1)
    
    for i in range(num_windows):
        start = i * HOP_SAMPLES
        end = start + WINDOW_SAMPLES
        if end > len(y):
            window = y[-WINDOW_SAMPLES:]
        else:
            window = y[start:end]

        prob, _, _ = model.predict(window, native_sr=TARGET_SR)
        if prob > max_prob:
            max_prob = prob

    return float(max_prob)

def main():
    base_dir = PROJECT_ROOT / "My_Test_Audio"
    dir_actual = base_dir / "Actual_Gunshots"
    dir_like = base_dir / "Like_Gunshots"
    dir_not = base_dir / "Not_Gunshots"

    actual_files = sorted(list(dir_actual.glob("*.wav")))
    like_files = sorted(list(dir_like.glob("*.wav")))
    not_files = sorted(list(dir_not.glob("*.wav")))

    print("=" * 85)
    print("🎯 SHOOT_CATCHER — 5-MODEL BENCHMARK ON EXTERNAL TEST AUDIO (OPTION A)")
    print("=" * 85)
    print(f" ├─ [Actual_Gunshots] : {len(actual_files)} files (Expected: Gunshot 🔫)")
    print(f" ├─ [Like_Gunshots]   : {len(like_files)} files (Expected: Non-Gunshot 💥 [Imposters])")
    print(f" └─ [Not_Gunshots]    : {len(not_files)} files (Expected: Non-Gunshot 🌿 [Ambient])")
    print("-" * 85)

    manager = live_demo.ModelManager()
    manager.print_audit_table()
    models = manager.trained_models
    print(f"Active trained models: {len(models)}\n")

    # Structure data: (filepath, label, category_name)
    test_suite = (
        [(f, 1, "Actual_Gunshots") for f in actual_files] +
        [(f, 0, "Like_Gunshots") for f in like_files] +
        [(f, 0, "Not_Gunshots") for f in not_files]
    )

    results = {m.id: {"name": m.name, "y_true": [], "y_prob": [], "y_pred": [], "by_cat": {}} for m in models}
    for m in models:
        for cat in ["Actual_Gunshots", "Like_Gunshots", "Not_Gunshots"]:
            results[m.id]["by_cat"][cat] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "total": 0}

    start_time = time.time()
    for idx, (fpath, true_label, cat_name) in enumerate(test_suite, 1):
        try:
            audio, sr = sf.read(str(fpath))
        except Exception as e:
            print(f"⚠️ Error reading {fpath.name}: {e}")
            continue

        print(f"[{idx:>2}/{len(test_suite)}] Evaluating: {fpath.name[:35]:<35} ({cat_name})")

        for m in models:
            max_p = evaluate_file_with_sliding_window(m, audio, sr)
            pred = 1 if max_p >= THRESHOLD else 0

            results[m.id]["y_true"].append(true_label)
            results[m.id]["y_prob"].append(max_p)
            results[m.id]["y_pred"].append(pred)

            c_stats = results[m.id]["by_cat"][cat_name]
            c_stats["total"] += 1
            if true_label == 1:
                if pred == 1:
                    c_stats["tp"] += 1
                else:
                    c_stats["fn"] += 1
            else:
                if pred == 1:
                    c_stats["fp"] += 1
                else:
                    c_stats["tn"] += 1

    elapsed = time.time() - start_time
    print(f"\n✅ Finished evaluation in {elapsed:.1f} seconds.\n")

    # Overall Metrics Table
    print("=" * 100)
    print("📊 OVERALL PERFORMANCE SCORECARD (ALL 75 EXTERNAL FILES)")
    print("=" * 100)
    print(f"{'Model Name':<28} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'F2-Score':<8} | {'TP':<4} {'FP':<4} {'TN':<4} {'FN':<4}")
    print("-" * 100)

    for m in models:
        y_t = np.array(results[m.id]["y_true"])
        y_p = np.array(results[m.id]["y_pred"])

        tp = int(np.sum((y_t == 1) & (y_p == 1)))
        fp = int(np.sum((y_t == 0) & (y_p == 1)))
        tn = int(np.sum((y_t == 0) & (y_p == 0)))
        fn = int(np.sum((y_t == 1) & (y_p == 0)))

        acc = (tp + tn) / len(y_t)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f2 = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0

        print(f"{m.name:<28} | {acc*100:>7.2f}% | {prec*100:>8.2f}% | {rec*100:>7.2f}% | {f1*100:>7.2f}% | {f2*100:>7.2f}% | {tp:>4} {fp:>4} {tn:>4} {fn:>4}")

    print("=" * 100)

    # Detailed Category Breakdown Table
    print("\n" + "=" * 100)
    print("🔍 DETAILED PERFORMANCE BREAKDOWN BY SOUND CATEGORY")
    print("=" * 100)
    for m in models:
        c_act = results[m.id]["by_cat"]["Actual_Gunshots"]
        c_like = results[m.id]["by_cat"]["Like_Gunshots"]
        c_not = results[m.id]["by_cat"]["Not_Gunshots"]

        print(f"📌 {m.name}:")
        print(f"   ├─ Real Gunshots Caught (Recall)      : {c_act['tp']}/{c_act['total']} ({c_act['tp']/max(1, c_act['total'])*100:.1f}%)")
        print(f"   ├─ Imposters Rejected (Claps/Knocks)  : {c_like['tn']}/{c_like['total']} ({c_like['tn']/max(1, c_like['total'])*100:.1f}%) [False Alarms: {c_like['fp']}]")
        print(f"   └─ Ambient Noise Ignored (Rain/Motor) : {c_not['tn']}/{c_not['total']} ({c_not['tn']/max(1, c_not['total'])*100:.1f}%) [False Alarms: {c_not['fp']}]")
    print("=" * 100)

if __name__ == "__main__":
    main()
