"""
Evaluation of all 5 Shoot_Catcher models on the 1:1 balanced test set in Data/SPLIT_DATASET_750MS/test/
"""
import sys
import os
import time
from pathlib import Path
import numpy as np
import soundfile as sf
# from tqdm import tqdm

# Add 03_Mic_Test/scripts to path to reuse ModelManager
PROJECT_ROOT = Path(r"c:\Users\aadit\Desktop\Shoot_Catcher")
SCRIPTS_DIR = PROJECT_ROOT / "03_Mic_Test" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import live_demo

def evaluate():
    test_dir = PROJECT_ROOT / "Data" / "SPLIT_DATASET_750MS" / "test"
    gunshot_dir = test_dir / "class_1_gunshot"
    nongunshot_dir = test_dir / "class_0_nongunshot"

    gunshot_files = sorted(list(gunshot_dir.glob("*.wav")))
    nongunshot_files = sorted(list(nongunshot_dir.glob("*.wav")))

    print(f"Found {len(gunshot_files)} gunshot test files.")
    print(f"Found {len(nongunshot_files)} non-gunshot test files.")
    total_files = len(gunshot_files) + len(nongunshot_files)
    print(f"Total test set size: {total_files}")

    manager = live_demo.ModelManager()
    manager.print_audit_table()
    models = manager.trained_models
    print(f"Active trained models: {len(models)}")
    for m in models:
        print(f" - {m.name} ({m.module})")

    # Metrics storage per model:
    # y_true, y_pred, y_prob
    results = {m.id: {"name": m.name, "y_true": [], "y_prob": [], "y_pred": []} for m in models}

    all_items = [(f, 1) for f in gunshot_files] + [(f, 0) for f in nongunshot_files]

    start_time = time.time()
    for idx, (filepath, label) in enumerate(all_items):
        if (idx + 1) % 100 == 0 or idx == len(all_items) - 1:
            print(f"Processing clip {idx + 1}/{len(all_items)}...", flush=True)

        audio, sr = sf.read(str(filepath))

        for m in models:
            prob, anom, _ = m.predict(audio, native_sr=sr)
            pred = 1 if prob >= 0.50 else 0
            results[m.id]["y_true"].append(label)
            results[m.id]["y_prob"].append(prob)
            results[m.id]["y_pred"].append(pred)

    elapsed = time.time() - start_time
    print(f"\nCompleted evaluation in {elapsed:.2f} seconds ({elapsed/total_files*1000:.1f} ms per file across all models).\n")

    # Calculate metrics
    print("=" * 95)
    print(f"{'Model Name':<28} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'F2-Score':<8} | {'TP':<4} {'FP':<4} {'TN':<4} {'FN':<4}")
    print("-" * 95)

    summary_stats = {}
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
        
        summary_stats[m.id] = {
            "name": m.name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "f2": f2,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn
        }

    print("=" * 95)

if __name__ == "__main__":
    evaluate()
