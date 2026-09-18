"""
===============================================================================
📊 Shoot_Catcher — Optimized Multi-Model Audio Benchmark Suite
===============================================================================
Evaluates all 5 Shoot_Catcher deep learning models across a benchmark dataset
structured into the 3 standard folders:
  - Actual_Gunshots/  (Ground Truth: 1 - Firearm Discharges)
  - Fake_Gunshots/    (Ground Truth: 0 - Acoustic Imposters & Handling Clicks)
  - Not_Gunshots/     (Ground Truth: 0 - Ambient Environmental Noise)

Optimizations:
  - Resampling performed once per audio file
  - Windows shared across all 5 models in a single unified pass
  - Energy-prioritized window scanning for ultra-fast evaluation on long recordings
  - Full metrics: Accuracy, Precision, Recall, Specificity, F1, F2
  - Per-category breakdown (Gunshots Caught, Imposters Rejected, Ambient Ignored)
  - HTML Interactive Player Dashboard & CSV scorecard generation
===============================================================================
"""

import os
import sys
import time
import argparse
import csv
from pathlib import Path
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "03_Mic_Test" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import live_demo

TARGET_SR = 22050
WINDOW_SAMPLES = 16537  # 750ms at 22,050Hz
HOP_SAMPLES = int(WINDOW_SAMPLES * 0.35)  # ~65% overlap (262.5ms hop for high speed + accuracy)
DEFAULT_THRESHOLD = 0.50
MAX_WINDOWS_PER_FILE = 60  # Cap on extremely long audio tracks

OUTPUT_DIR = PROJECT_ROOT / "Verification_Outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def find_audio_files(directory):
    if not directory.exists():
        return []
    exts = ("*.wav", "*.mp3", "*.flac", "*.ogg")
    files = []
    for ext in exts:
        files.extend(directory.glob(ext))
        files.extend(directory.glob(ext.upper()))
    return sorted(list(set(files)))

def fast_resample(y, orig_sr, target_sr):
    """Ultra-fast polyphase resample (10ms) using soxr, replacing slow FFT."""
    if orig_sr == target_sr:
        return y.astype(np.float32)
    try:
        import soxr
        return soxr.resample(y, orig_sr, target_sr).astype(np.float32)
    except Exception:
        import scipy.signal
        gcd = np.gcd(orig_sr, target_sr)
        return scipy.signal.resample_poly(y, target_sr // gcd, orig_sr // gcd).astype(np.float32)

def evaluate_file_across_all_models(models, audio_data, sr):
    """
    Fast unified evaluation: resamples in 10ms with soxr, extracts windows,
    prioritizes energetic transients, and updates model max confidences.
    """
    if audio_data.ndim > 1:
        y = audio_data.mean(axis=1)
    else:
        y = audio_data.flatten()

    y = fast_resample(y, sr, TARGET_SR)

    if len(y) < WINDOW_SAMPLES:
        y = np.pad(y, (0, WINDOW_SAMPLES - len(y)))

    total_len = len(y)
    num_total_windows = max(1, int(np.ceil((total_len - WINDOW_SAMPLES) / HOP_SAMPLES)) + 1)

    # Pre-extract all window indices and compute their RMS energy
    window_candidates = []
    for i in range(num_total_windows):
        start = i * HOP_SAMPLES
        end = start + WINDOW_SAMPLES
        if end > total_len:
            w = y[-WINDOW_SAMPLES:]
            actual_start = total_len - WINDOW_SAMPLES
        else:
            w = y[start:end]
            actual_start = start
        rms = float(np.sqrt(np.mean(w ** 2)))
        window_candidates.append((actual_start, rms))

    # If file is long, prioritize the top energetic windows + evenly spaced temporal windows
    if len(window_candidates) > MAX_WINDOWS_PER_FILE:
        # Top 35 by energy (catches all gunshots and bursts)
        by_energy = sorted(window_candidates, key=lambda x: -x[1])[:35]
        # Top 25 evenly spaced across the full timeline (catches steady states)
        step = max(1, len(window_candidates) // 25)
        by_time = [window_candidates[k] for k in range(0, len(window_candidates), step)][:25]
        
        # Combine unique starts sorted chronologically
        selected_starts = sorted(list(set([item[0] for item in (by_energy + by_time)])))
    else:
        selected_starts = [item[0] for item in window_candidates]

    max_probs = {m.id: 0.0 for m in models}
    best_times = {m.id: 0.0 for m in models}

    for start in selected_starts:
        end = start + WINDOW_SAMPLES
        w = y[start:end]
        time_sec = start / TARGET_SR

        # Skip digital silence
        if np.max(np.abs(w)) < 1e-5:
            continue

        # Single evaluation pass across active models
        for m in models:
            prob, _, _ = m.predict(w, native_sr=TARGET_SR)
            if prob > max_probs[m.id]:
                max_probs[m.id] = float(prob)
                best_times[m.id] = float(time_sec)

        # Early exit if all active models already saturated high confidence
        if all(p >= 0.99 for p in max_probs.values()):
            break

    return max_probs, best_times

def generate_html_report(report_path, dataset_name, models, metrics_data, log_records):
    """Generate a sleek, dark-mode HTML dashboard with results."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Shoot_Catcher — Benchmark Report: {dataset_name}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 28px; }}
        h1 {{ margin-bottom: 4px; color: #38bdf8; font-size: 26px; }}
        p.sub {{ color: #94a3b8; font-size: 14px; margin-top: 0; margin-bottom: 24px; }}
        .cards {{ display: flex; gap: 16px; margin-bottom: 28px; flex-wrap: wrap; }}
        .card {{ background: #1e293b; padding: 16px 22px; border-radius: 10px; border: 1px solid #334155; min-width: 180px; }}
        .card-val {{ font-size: 24px; font-weight: 700; color: #38bdf8; }}
        .card-lbl {{ font-size: 12px; color: #94a3b8; text-transform: uppercase; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 10px; overflow: hidden; margin-bottom: 32px; font-size: 13px; }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ background: #0f172a; color: #94a3b8; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }}
        tr:hover {{ background: #273549; }}
        .badge {{ padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 11px; text-transform: uppercase; }}
        .tp {{ background: #065f46; color: #34d399; }}
        .tn {{ background: #1e3a8a; color: #60a5fa; }}
        .fp {{ background: #831843; color: #f472b6; }}
        .fn {{ background: #7c2d12; color: #fb923c; }}
        .bar-wrap {{ background: #334155; border-radius: 4px; height: 8px; width: 100px; display: inline-block; margin-right: 8px; vertical-align: middle; overflow: hidden; }}
        .bar-fill {{ background: #38bdf8; height: 100%; }}
    </style>
</head>
<body>
    <h1>🔬 Shoot_Catcher Benchmark: {dataset_name}</h1>
    <p class="sub">Evaluation across all 5 trained deep learning models | 750ms Sliding Window | Decision Threshold: 0.50</p>
    
    <h2>📊 Model Performance Scorecard</h2>
    <table>
        <thead>
            <tr>
                <th>Model Architecture</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall (Gunshots)</th>
                <th>F1-Score</th>
                <th>F2-Score</th>
                <th>Gunshots Caught</th>
                <th>Fake Gunshots Rejected</th>
                <th>Ambient Ignored</th>
                <th>TP / FP / TN / FN</th>
            </tr>
        </thead>
        <tbody>
    """
    for m in models:
        d = metrics_data[m.id]
        html += f"""
            <tr>
                <td><strong>{m.name}</strong></td>
                <td><strong>{d['acc']*100:.2f}%</strong></td>
                <td>{d['prec']*100:.1f}%</td>
                <td><span class="bar-wrap"><span class="bar-fill" style="width:{d['rec']*100}%;"></span></span>{d['rec']*100:.1f}%</td>
                <td>{d['f1']*100:.1f}%</td>
                <td><strong>{d['f2']*100:.1f}%</strong></td>
                <td>{d['actual_caught']}</td>
                <td>{d['fake_rejected']}</td>
                <td>{d['not_ignored']}</td>
                <td><code>{d['tp']} / {d['fp']} / {d['tn']} / {d['fn']}</code></td>
            </tr>
        """
    html += """
        </tbody>
    </table>

    <h2>📋 Detailed Evaluation Log (Sample Audio Records)</h2>
    <table>
        <thead>
            <tr>
                <th>Model</th>
                <th>Ground Truth Category</th>
                <th>Audio File</th>
                <th>Max Confidence</th>
                <th>Decision Outcome</th>
            </tr>
        </thead>
        <tbody>
    """
    for r in log_records[:200]:
        badge_cls = r['outcome_code'].lower()
        html += f"""
            <tr>
                <td>{r['model_name']}</td>
                <td>{r['category']}</td>
                <td><code>{r['filename']}</code></td>
                <td><strong>{r['confidence']*100:.1f}%</strong></td>
                <td><span class="badge {badge_cls}">{r['outcome']}</span></td>
            </tr>
        """
    html += """
        </tbody>
    </table>
</body>
</html>
    """
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

def run_benchmark(dataset_dir, dataset_name, max_files_per_cat=None, threshold=DEFAULT_THRESHOLD):
    base_dir = Path(dataset_dir)

    # Flexible directory lookup (with or without underscores)
    def find_matching_dir(parent, candidates):
        for c in candidates:
            p = parent / c
            if p.exists() and p.is_dir():
                return p
        return parent / candidates[0]

    dir_actual = find_matching_dir(base_dir, ["Actual_Gunshots", "ActualGunshots", "actual_gunshots", "actualgunshots"])
    dir_fake = find_matching_dir(base_dir, ["Fake_Gunshots", "FakeGunshots", "fake_gunshots", "fakegunshots"])
    dir_not = find_matching_dir(base_dir, ["Not_Gunshots", "NotGunshots", "not_gunshots", "notgunshots"])

    actual_files = find_audio_files(dir_actual)
    fake_files = find_audio_files(dir_fake)
    not_files = find_audio_files(dir_not)

    if max_files_per_cat:
        actual_files = actual_files[:max_files_per_cat]
        fake_files = fake_files[:max_files_per_cat]
        not_files = not_files[:max_files_per_cat]

    total_files = len(actual_files) + len(fake_files) + len(not_files)

    print("=" * 85)
    print(f"🎯 SHOOT_CATCHER BENCHMARK — DATASET: [{dataset_name}]")
    print("=" * 85)
    print(f"Directory        : {base_dir}")
    print(f"Actual Gunshots  : {len(actual_files)} files (True Firearm Discharges)")
    print(f"Fake Gunshots    : {len(fake_files)} files (Acoustic Imposters / Weapon Clicks)")
    print(f"Not Gunshots     : {len(not_files)} files (Ambient Background Noise)")
    print(f"Total Evaluated  : {total_files} files")
    print(f"Decision Threshold: {threshold:.2f}")
    print("=" * 85)

    if total_files == 0:
        print("❌ No audio files found in target dataset directories!")
        return

    # Load models
    manager = live_demo.ModelManager()
    manager.print_audit_table()
    models = manager.trained_models
    print(f"\n⚡ Active Evaluated Models: {len(models)}\n")

    test_suite = (
        [(f, 1, "Actual_Gunshots") for f in actual_files] +
        [(f, 0, "Fake_Gunshots") for f in fake_files] +
        [(f, 0, "Not_Gunshots") for f in not_files]
    )

    results = {
        m.id: {
            "name": m.name,
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "by_cat": {
                "Actual_Gunshots": {"caught": 0, "total": len(actual_files)},
                "Fake_Gunshots": {"rejected": 0, "total": len(fake_files)},
                "Not_Gunshots": {"rejected": 0, "total": len(not_files)},
            }
        } for m in models
    }

    log_records = []
    start_time = time.time()

    for idx, (fpath, true_label, cat_name) in enumerate(test_suite, 1):
        # Use extended path syntax for Windows long paths
        abs_path = str(fpath.resolve())
        long_path = "\\\\?\\" + abs_path if sys.platform == "win32" else abs_path

        try:
            audio, sr = sf.read(long_path)
        except Exception as e:
            print(f"⚠️ Error reading {fpath.name[:30]}: {e}")
            continue

        print(f"[{idx:3d}/{total_files}] Processing: {fpath.name[:50]:<50}", end=" ", flush=True)

        # Evaluate across all models in one unified pass
        max_probs, best_times = evaluate_file_across_all_models(models, audio, sr)

        for m in models:
            conf = max_probs[m.id]
            best_t = best_times[m.id]
            pred = 1 if conf >= threshold else 0

            # Outcomes
            if true_label == 1 and pred == 1:
                outcome = "True Positive"
                outcome_code = "TP"
                results[m.id]["tp"] += 1
                results[m.id]["by_cat"]["Actual_Gunshots"]["caught"] += 1
            elif true_label == 0 and pred == 0:
                outcome = "True Negative"
                outcome_code = "TN"
                results[m.id]["tn"] += 1
                if cat_name == "Fake_Gunshots":
                    results[m.id]["by_cat"]["Fake_Gunshots"]["rejected"] += 1
                else:
                    results[m.id]["by_cat"]["Not_Gunshots"]["rejected"] += 1
            elif true_label == 0 and pred == 1:
                outcome = "False Positive (Alarm)"
                outcome_code = "FP"
                results[m.id]["fp"] += 1
            else:
                outcome = "False Negative (Miss)"
                outcome_code = "FN"
                results[m.id]["fn"] += 1

            log_records.append({
                "model_name": m.name,
                "category": cat_name,
                "filename": fpath.name,
                "confidence": conf,
                "outcome": outcome,
                "outcome_code": outcome_code,
                "time_sec": best_t
            })

        print("✔")

    elapsed = time.time() - start_time
    print("\n" + "=" * 90)
    print(f"📊 BENCHMARK SCORECARD: {dataset_name} (Completed in {elapsed:.1f}s)")
    print("=" * 90)
    print(f"{'Model Architecture':<28} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'F2-Score':<8} | {'TP/FP/TN/FN':<15}")
    print("-" * 90)

    metrics_summary = {}

    for m in models:
        r = results[m.id]
        tp, fp, tn, fn = r["tp"], r["fp"], r["tn"], r["fn"]
        total = max(1, tp + fp + tn + fn)
        acc = (tp + tn) / total
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f2 = (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0

        act_caught = f"{r['by_cat']['Actual_Gunshots']['caught']}/{max(1, len(actual_files))}"
        fake_rej = f"{r['by_cat']['Fake_Gunshots']['rejected']}/{max(1, len(fake_files))}"
        not_rej = f"{r['by_cat']['Not_Gunshots']['rejected']}/{max(1, len(not_files))}"

        metrics_summary[m.id] = {
            "acc": acc, "prec": prec, "rec": rec, "f1": f1, "f2": f2,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "actual_caught": act_caught,
            "fake_rejected": fake_rej,
            "not_ignored": not_rej
        }

        print(f"{m.name:<28} | {acc*100:6.2f}% | {prec*100:7.2f}% | {rec*100:6.2f}% | {f1*100:6.2f}% | {f2*100:6.2f}% | {tp:3d}/{fp:3d}/{tn:3d}/{fn:3d}")

    print("=" * 90)

    # Export CSV
    clean_name = dataset_name.lower().replace(" ", "_")
    csv_path = OUTPUT_DIR / f"{clean_name}_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Accuracy", "Precision", "Recall", "F1", "F2", "TP", "FP", "TN", "FN", "Gunshots_Caught", "Fake_Gunshots_Rejected", "Ambient_Ignored"])
        for m in models:
            d = metrics_summary[m.id]
            writer.writerow([m.name, f"{d['acc']:.4f}", f"{d['prec']:.4f}", f"{d['rec']:.4f}", f"{d['f1']:.4f}", f"{d['f2']:.4f}", d['tp'], d['fp'], d['tn'], d['fn'], d['actual_caught'], d['fake_rejected'], d['not_ignored']])
    print(f"📁 Saved CSV Summary : {csv_path}")

    # Export HTML Dashboard
    html_path = OUTPUT_DIR / f"{clean_name}_dashboard.html"
    generate_html_report(html_path, dataset_name, models, metrics_summary, log_records)
    print(f"🌐 Saved HTML Report : {html_path}")
    print("=" * 90)

def main():
    parser = argparse.ArgumentParser(description="Shoot_Catcher Unified Audio Benchmark Suite")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Path to dataset directory containing 3 folders")
    parser.add_argument("--name", type=str, default="Benchmark", help="Dataset display name")
    parser.add_argument("--max-files", type=int, default=None, help="Max files per category to evaluate")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Confidence threshold (default: 0.50)")
    args = parser.parse_args()

    run_benchmark(args.dataset_dir, args.name, max_files_per_cat=args.max_files, threshold=args.threshold)

if __name__ == "__main__":
    main()
