"""
===============================================================================
📊 Shoot_Catcher — Benchmark Runner: Held-Out Dataset (Option B)
===============================================================================
Evaluates all 5 trained models on the curated 1:1 balanced test dataset:
  - Data/SPLIT_DATASET_750MS/test/class_1_gunshot/ (721 files)
  - Data/SPLIT_DATASET_750MS/test/class_0_nongunshot/ (721 files)

Computes Accuracy, Precision, Recall, F1, F2, and Confusion Matrix (TP, FP, TN, FN).
===============================================================================
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
import soundfile as sf

# Path configuration
PROJECT_ROOT = Path(os.path.abspath(__file__)).parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "03_Mic_Test" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import live_demo

def main():
    test_dir = PROJECT_ROOT / "Data" / "SPLIT_DATASET_750MS" / "test"
    gunshot_dir = test_dir / "class_1_gunshot"
    nongunshot_dir = test_dir / "class_0_nongunshot"

    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        return

    gunshot_files = sorted(list(gunshot_dir.glob("*.wav")))
    nongunshot_files = sorted(list(nongunshot_dir.glob("*.wav")))
    total_files = len(gunshot_files) + len(nongunshot_files)

    print("=" * 85)
    print("🎯 SHOOT_CATCHER — 5-MODEL BENCHMARK: HELD-OUT SPLIT DATASET")
    print("=" * 85)
    print(f"Dataset Path   : {test_dir}")
    print(f"Gunshot Clips  : {len(gunshot_files)}")
    print(f"Non-Gunshot    : {len(nongunshot_files)}")
    print(f"Total Clips    : {total_files}")
    print("-" * 85)

    manager = live_demo.ModelManager()
    manager.print_audit_table()
    models = manager.trained_models
    print(f"Active trained models for evaluation: {len(models)}\n")

    results = {m.id: {"name": m.name, "y_true": [], "y_prob": [], "y_pred": []} for m in models}
    all_items = [(f, 1) for f in gunshot_files] + [(f, 0) for f in nongunshot_files]

    start_time = time.time()
    for idx, (filepath, label) in enumerate(all_items, 1):
        if idx % 100 == 0 or idx == len(all_items):
            print(f"⏳ Evaluating clip [{idx:>4}/{total_files}]...", flush=True)

        try:
            audio, sr = sf.read(str(filepath))
        except Exception as e:
            print(f"⚠️ Error reading {filepath.name}: {e}")
            continue

        for m in models:
            prob, anom, _ = m.predict(audio, native_sr=sr)
            pred = 1 if prob >= 0.50 else 0
            results[m.id]["y_true"].append(label)
            results[m.id]["y_prob"].append(prob)
            results[m.id]["y_pred"].append(pred)

    elapsed = time.time() - start_time
    print(f"\n✅ Completed evaluation in {elapsed:.2f} seconds ({elapsed/total_files*1000:.1f} ms/file across all models).\n")

    print("=" * 100)
    print("📊 SCORECARD: HELD-OUT DATASET TEST RESULTS")
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

if __name__ == "__main__":
    main()
