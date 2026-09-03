"""
===============================================================================
📊 Shoot_Catcher — Benchmark Runner with Human Audio Verification Export
===============================================================================
Evaluates models on audio files and organizes detections for human listening:
  - My_Test_Audio/Actual_Gunshots/ (Label: 1)
  - My_Test_Audio/Like_Gunshots/   (Label: 0 - Hard Imposters)
  - My_Test_Audio/Not_Gunshots/    (Label: 0 - Ambient Background)

Outputs:
  - Verification_Outputs/By_Model/<Model_Name>/Detected_Gunshots/
  - Verification_Outputs/By_Model/<Model_Name>/Ignored_NonGunshots/
  - Verification_Outputs/By_Model/<Model_Name>/Trigger_Slices_750ms/
  - Verification_Outputs/verification_summary.csv
  - Verification_Outputs/verification_dashboard.html (Interactive Player)
===============================================================================
"""

import os
import sys
import time
import shutil
import csv
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(os.path.abspath(__file__)).parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "03_Mic_Test" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import live_demo

TARGET_SR = 22050
WINDOW_SAMPLES = 16537  # 750ms at 22,050Hz
HOP_SAMPLES = int(WINDOW_SAMPLES * 0.25)  # 75% overlap (187.5ms hop)
DEFAULT_THRESHOLD = 0.50

VERIFICATION_DIR = PROJECT_ROOT / "Verification_Outputs"

def sanitize_name(name):
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")

def evaluate_file_with_sliding_window(model, audio_data, sr, threshold=0.50):
    """Slide 750ms window across file. Returns (max_prob, best_window, best_time_sec)."""
    if audio_data.ndim > 1:
        y = audio_data.mean(axis=1)
    else:
        y = audio_data.flatten()

    y = live_demo.resample_audio(y, sr, TARGET_SR)

    if len(y) < WINDOW_SAMPLES:
        y = np.pad(y, (0, WINDOW_SAMPLES - len(y)))

    max_prob = 0.0
    best_window = None
    best_time = 0.0

    num_windows = max(1, int(np.ceil((len(y) - WINDOW_SAMPLES) / HOP_SAMPLES)) + 1)

    for i in range(num_windows):
        start = i * HOP_SAMPLES
        end = start + WINDOW_SAMPLES
        time_sec = start / TARGET_SR

        if end > len(y):
            window = y[-WINDOW_SAMPLES:]
        else:
            window = y[start:end]

        prob, _, _ = model.predict(window, native_sr=TARGET_SR)
        if prob > max_prob:
            max_prob = prob
            best_window = window.copy()
            best_time = time_sec

    return float(max_prob), best_window, best_time

def generate_html_dashboard(dashboard_path, log_records, threshold=0.50):
    """Generate an interactive HTML dashboard with embedded audio players for human audit."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Shoot_Catcher — Human Audio Verification Dashboard</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #2b2c2e; color: #f8fafc; margin: 0; padding: 24px; }}
        h1 {{ margin-bottom: 4px; color: #38bdf8; font-size: 24px; }}
        p.subtitle {{ color: #94a3b8; font-size: 14px; margin-top: 0; margin-bottom: 20px; }}
        .stats-bar {{ display: flex; gap: 16px; margin-bottom: 24px; }}
        .stat-card {{ background: #1e293b; padding: 14px 20px; border-radius: 8px; border: 1px solid #334155; }}
        .stat-val {{ font-size: 20px; font-weight: bold; color: #38bdf8; }}
        .stat-lbl {{ font-size: 12px; color: #94a3b8; text-transform: uppercase; }}
        table {{ width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 8px; overflow: hidden; font-size: 13px; }}
        th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ background: #2b2c2e; color: #94a3b8; font-weight: 600; text-transform: uppercase; font-size: 11px; }}
        tr:hover {{ background: #444f64; }}
        .tag {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 11px; }}
        .tag-tp {{ background: #065f46; color: #34d399; }}
        .tag-fp {{ background: #991b1b; color: #f87171; }}
        .tag-tn {{ background: #334155; color: #94a3b8; }}
        .tag-fn {{ background: #854d0e; color: #facc15; }}
        audio {{ height: 32px; width: 220px; }}
        .filter-controls {{ margin-bottom: 16px; display: flex; gap: 10px; }}
        select, input {{ background: #1e293b; color: #f8fafc; border: 1px solid #334155; padding: 6px 12px; border-radius: 6px; font-size: 13px; }}
    </style>
</head>
<body>
    <h1>🎯 Shoot_Catcher — Human Audio Verification Dashboard</h1>
    <p class="subtitle">Listen to and verify audio files differentiated by each model</p>
    
    <div class="stats-bar">
        <div class="stat-card"><div class="stat-val">{len(log_records)}</div><div class="stat-lbl">Evaluations Logged</div></div>
        <div class="stat-card"><div class="stat-val">22,050 Hz</div><div class="stat-lbl">Target Sample Rate</div></div>
        <div class="stat-card"><div class="stat-val">750 ms</div><div class="stat-lbl">Sliding Window</div></div>
        <div class="stat-card"><div class="stat-val">{threshold:.2f}</div><div class="stat-lbl">Trigger Threshold</div></div>
    </div>

    <table id="verifTable">
        <thead>
            <tr>
                <th>Model</th>
                <th>Sound Category</th>
                <th>Original File</th>
                <th>Trigger Time</th>
                <th>Confidence</th>
                <th>Outcome</th>
                <th>Listen to Full File</th>
                <th>Listen to 750ms Trigger Slice</th>
            </tr>
        </thead>
        <tbody>
    """

    for r in log_records:
        outcome = r["outcome"]
        tag_class = "tag-tn"
        if outcome == "True Positive":
            tag_class = "tag-tp"
        elif outcome == "False Positive (Alarm)":
            tag_class = "tag-fp"
        elif outcome == "False Negative (Missed)":
            tag_class = "tag-fn"

        full_rel = os.path.relpath(r["full_audio_path"], dashboard_path.parent).replace("\\", "/") if r.get("full_audio_path") else ""
        slice_rel = os.path.relpath(r["slice_audio_path"], dashboard_path.parent).replace("\\", "/") if r.get("slice_audio_path") else None

        html += f"""
            <tr>
                <td><strong>{r['model_name']}</strong></td>
                <td>{r['category']}</td>
                <td><code>{r['filename']}</code></td>
                <td>{r['time_sec']:.2f}s</td>
                <td><strong>{r['confidence']*100:.1f}%</strong></td>
                <td><span class="tag {tag_class}">{outcome}</span></td>
                <td>{f'<audio controls preload="none" src="{full_rel}"></audio>' if full_rel else '<span style="color:#64748b;">N/A</span>'}</td>
                <td>{f'<audio controls preload="none" src="{slice_rel}"></audio>' if slice_rel else '<span style="color:#64748b;">N/A</span>'}</td>
            </tr>
        """

    html += """
        </tbody>
    </table>
</body>
</html>
    """
    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(html)

def main():
    parser = argparse.ArgumentParser(description="Shoot_Catcher — Benchmark Runner with Dual Flow Support (Normal & Human Verification)")
    parser.add_argument("--no-export", action="store_true", help="Run in FAST mode (metrics scorecard only, skip copying audio files)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Confidence trigger threshold (default: 0.50)")
    parser.add_argument("--input-dir", type=str, default=None, help="Custom test audio directory (default: My_Test_Audio)")
    args = parser.parse_args()

    export_audio = not args.no_export
    threshold = args.threshold

    base_dir = Path(args.input_dir) if args.input_dir else (PROJECT_ROOT / "My_Test_Audio")
    dir_actual = base_dir / "Actual_Gunshots"
    dir_like = base_dir / "Like_Gunshots"
    dir_not = base_dir / "Not_Gunshots"

    actual_files = sorted(list(dir_actual.glob("*.wav"))) if dir_actual.exists() else []
    like_files = sorted(list(dir_like.glob("*.wav"))) if dir_like.exists() else []
    not_files = sorted(list(dir_not.glob("*.wav"))) if dir_not.exists() else []
    total_files = len(actual_files) + len(like_files) + len(not_files)

    print("=" * 85)
    print(f"🎯 SHOOT_CATCHER — 5-MODEL BENCHMARK [{ 'HUMAN VERIFICATION FLOW' if export_audio else 'NORMAL FAST FLOW' }]")
    print("=" * 85)
    print(f"Input Directory    : {base_dir}")
    print(f"Actual Gunshots    : {len(actual_files)} files (True Gunshots)")
    print(f"Like Gunshots      : {len(like_files)} files (Hard Imposters)")
    print(f"Not Gunshots       : {len(not_files)} files (Everyday Ambient)")
    print(f"Total Audio Files  : {total_files}")
    print(f"Detection Threshold: {threshold:.2f}")
    print(f"Audio Export Mode  : {'ENABLED -> ' + str(VERIFICATION_DIR) if export_audio else 'DISABLED (Fast Metrics Only)'}")
    print("-" * 85)

    if export_audio:
        VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)

    manager = live_demo.ModelManager()
    manager.print_audit_table()
    models = manager.trained_models
    print(f"Active trained models: {len(models)}\n")

    if export_audio:
        for m in models:
            m_dir = VERIFICATION_DIR / "By_Model" / sanitize_name(m.name)
            (m_dir / "Detected_Gunshots").mkdir(parents=True, exist_ok=True)
            (m_dir / "Ignored_NonGunshots").mkdir(parents=True, exist_ok=True)
            (m_dir / "Trigger_Slices_750ms").mkdir(parents=True, exist_ok=True)

    test_suite = (
        [(f, 1, "Actual_Gunshots") for f in actual_files] +
        [(f, 0, "Like_Gunshots") for f in like_files] +
        [(f, 0, "Not_Gunshots") for f in not_files]
    )

    results = {m.id: {"name": m.name, "y_true": [], "y_prob": [], "y_pred": [], "by_cat": {}} for m in models}
    for m in models:
        for cat in ["Actual_Gunshots", "Like_Gunshots", "Not_Gunshots"]:
            results[m.id]["by_cat"][cat] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "total": 0}

    log_records = []
    start_time = time.time()

    for idx, (fpath, true_label, cat_name) in enumerate(test_suite, 1):
        try:
            audio, sr = sf.read(str(fpath))
        except Exception as e:
            print(f"⚠️ Error reading {fpath.name}: {e}")
            continue

        print(f"[{idx:>2}/{total_files}] Evaluating & Exporting: {fpath.name[:35]:<35} ({cat_name})")

        for m in models:
            m_slug = sanitize_name(m.name)
            m_dir = VERIFICATION_DIR / "By_Model" / m_slug

            max_p, best_win, best_time = evaluate_file_with_sliding_window(m, audio, sr, threshold=threshold)
            pred = 1 if max_p >= threshold else 0

            results[m.id]["y_true"].append(true_label)
            results[m.id]["y_prob"].append(max_p)
            results[m.id]["y_pred"].append(pred)

            c_stats = results[m.id]["by_cat"][cat_name]
            c_stats["total"] += 1

            outcome = ""
            if true_label == 1:
                if pred == 1:
                    c_stats["tp"] += 1
                    outcome = "True Positive"
                else:
                    c_stats["fn"] += 1
                    outcome = "False Negative (Missed)"
            else:
                if pred == 1:
                    c_stats["fp"] += 1
                    outcome = "False Positive (Alarm)"
                else:
                    c_stats["tn"] += 1
                    outcome = "True Negative"

            dest_file_path = None
            slice_path = None

            if export_audio:
                # Determine destination folder
                dest_folder = m_dir / "Detected_Gunshots" if pred == 1 else m_dir / "Ignored_NonGunshots"
                dest_file_name = f"[{cat_name}]_{fpath.stem}_conf_{max_p:.3f}.wav"
                dest_file_path = dest_folder / dest_file_name

                # Copy full audio file for human listening
                if not dest_file_path.exists():
                    shutil.copy2(str(fpath), str(dest_file_path))

                # Save the exact 750ms trigger slice if detected
                if pred == 1 and best_win is not None:
                    slice_name = f"[{cat_name}]_{fpath.stem}_at_{best_time:.2f}s_conf_{max_p:.3f}.wav"
                    slice_path = m_dir / "Trigger_Slices_750ms" / slice_name
                    if not slice_path.exists():
                        sf.write(str(slice_path), best_win, TARGET_SR)

                log_records.append({
                    "model_name": m.name,
                    "category": cat_name,
                    "filename": fpath.name,
                    "true_label": true_label,
                    "prediction": pred,
                    "confidence": max_p,
                    "time_sec": best_time,
                    "outcome": outcome,
                    "full_audio_path": str(dest_file_path),
                    "slice_audio_path": str(slice_path) if slice_path else ""
                })

    elapsed = time.time() - start_time
    print(f"\n✅ Finished evaluation in {elapsed:.1f} seconds.\n")

    if export_audio:
        # Write CSV Log
        csv_path = VERIFICATION_DIR / "verification_summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "model_name", "category", "filename", "true_label", "prediction",
                "confidence", "time_sec", "outcome", "full_audio_path", "slice_audio_path"
            ])
            writer.writeheader()
            writer.writerows(log_records)
        print(f"📄 Saved CSV Audit Log: {csv_path}")

        # Generate HTML Audio Dashboard
        dashboard_path = VERIFICATION_DIR / "verification_dashboard.html"
        generate_html_dashboard(dashboard_path, log_records, threshold=threshold)
        print(f"🌐 Generated Interactive Audio Dashboard: {dashboard_path}")

    # Print Scorecard
    print("\n" + "=" * 100)
    print("📊 SCORECARD (ALL 75 EXTERNAL AUDIO FILES)")
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

    # Print category breakdown
    print("\n" + "=" * 100)
    print("🔍 DETAILED BREAKDOWN BY SOUND CATEGORY")
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

    print(f"\n📂 Audio files ready for human verification in: {VERIFICATION_DIR}")

if __name__ == "__main__":
    main()
