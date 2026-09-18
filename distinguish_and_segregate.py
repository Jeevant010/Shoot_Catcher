"""
===============================================================================
🎯 Shoot_Catcher — Multi-Model Audio Distinction & Segregation Pipeline
===============================================================================
Evaluates all audio files in test_dataset against trained Shoot_Catcher models,
distinguishes sounds into 3 separate classes:
  1. Real Gunshots   (Authentic firearm discharges)
  2. Fake Gunshots   (Acoustic imposters: fireworks, claps, door slams, clicks)
  3. Not Gunshots    (Ambient environmental noise: speech, engines, rain, sirens)

Physically segregates files into target destination folders:
  - Verification_Outputs/Classified_Audio/Real_Gunshots/
  - Verification_Outputs/Classified_Audio/Fake_Gunshots/
  - Verification_Outputs/Classified_Audio/Not_Gunshots/
  - Per-model folders under Verification_Outputs/By_Model/...

Generates:
  - Verification_Outputs/test_dataset_classification_scorecard.csv
  - Verification_Outputs/test_dataset_dashboard.html
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
import scipy.signal as signal

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "03_Mic_Test" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import live_demo

TARGET_SR = 22050
WINDOW_SAMPLES = 16537  # 750ms at 22,050Hz
HOP_SAMPLES = int(WINDOW_SAMPLES * 0.25)  # 75% overlap (187.5ms hop)
DEFAULT_THRESHOLD = 0.50

OUTPUT_BASE = PROJECT_ROOT / "Verification_Outputs"
CLASSIFIED_DIR = OUTPUT_BASE / "Classified_Audio"
DIR_REAL = CLASSIFIED_DIR / "Real_Gunshots"
DIR_FAKE = CLASSIFIED_DIR / "Fake_Gunshots"
DIR_NOT = CLASSIFIED_DIR / "Not_Gunshots"

def setup_output_dirs(models):
    for d in [DIR_REAL, DIR_FAKE, DIR_NOT]:
        d.mkdir(parents=True, exist_ok=True)
    for m in models:
        m_slug = m.name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        (OUTPUT_BASE / "By_Model" / m_slug / "Predicted_Gunshots").mkdir(parents=True, exist_ok=True)
        (OUTPUT_BASE / "By_Model" / m_slug / "Predicted_NonGunshots").mkdir(parents=True, exist_ok=True)

def find_audio_files(directory):
    if not directory or not directory.exists():
        return []
    exts = ("*.wav", "*.mp3", "*.flac", "*.ogg")
    files = []
    for ext in exts:
        files.extend(directory.glob(ext))
        files.extend(directory.glob(ext.upper()))
    return sorted(list(set(files)))

def locate_dataset_folders(base_dir):
    p = Path(base_dir)
    if (p / "test_dataset").exists() and (p / "test_dataset").is_dir():
        cand = p / "test_dataset"
        sub = [x.name.lower() for x in cand.iterdir() if x.is_dir()]
        if any("gunshot" in s or "imposter" in s for s in sub):
            p = cand

    def find_dir(candidates):
        for c in candidates:
            cand_p = p / c
            if cand_p.exists() and cand_p.is_dir():
                return cand_p
        for item in p.iterdir():
            if item.is_dir():
                for c in candidates:
                    if c.lower() in item.name.lower():
                        return item
        return None

    dir_gunshots = find_dir(["1_gunshots", "gunshots", "actual_gunshots", "actualgunshots", "class_1_gunshot"])
    dir_nongunshots = find_dir(["2_nongunshots", "nongunshots", "not_gunshots", "notgunshots", "class_0_nongunshot", "ambient"])
    dir_imposters = find_dir(["3_imposters", "imposters", "imposter", "fake_gunshots", "fakegunshots", "like_gunshots"])

    return dir_gunshots, dir_nongunshots, dir_imposters

def fast_resample(y, orig_sr, target_sr):
    if orig_sr == target_sr:
        return y.astype(np.float32)
    try:
        import soxr
        return soxr.resample(y, orig_sr, target_sr).astype(np.float32)
    except Exception:
        gcd = np.gcd(orig_sr, target_sr)
        return signal.resample_poly(y, target_sr // gcd, orig_sr // gcd).astype(np.float32)

def extract_acoustic_features(w, sr=TARGET_SR):
    """Extract physical acoustic transient features from a 750ms window."""
    w = w.astype(np.float32)
    w = w - np.mean(w)
    peak = float(np.max(np.abs(w)))
    rms = float(np.sqrt(np.mean(w ** 2)))
    crest_factor_db = float(20 * np.log10(peak / (rms + 1e-9))) if rms > 1e-9 else 0.0

    fft_len = min(len(w), sr)
    fft_vals = np.abs(np.fft.rfft(w[:fft_len])) ** 2
    freqs = np.fft.rfftfreq(fft_len, 1.0 / sr)
    total_power = float(np.sum(fft_vals)) + 1e-12

    low_freq_ratio = float(np.sum(fft_vals[freqs < 200.0]) / total_power)
    mid_freq_ratio = float(np.sum(fft_vals[(freqs >= 200.0) & (freqs <= 2500.0)]) / total_power)
    high_freq_crack_ratio = float(np.sum(fft_vals[freqs > 4000.0]) / total_power)
    spectral_centroid = float(np.sum(freqs * fft_vals) / total_power)

    # Multi-burst count (detecting cluster bursts like fireworks)
    hop_env = int(sr * 0.01)
    env = [np.max(np.abs(w[i:i + hop_env])) for i in range(0, len(w) - hop_env, hop_env)]
    burst_thresh = peak * 0.25
    burst_count = 0
    in_burst = False
    for v in env:
        if v >= burst_thresh:
            if not in_burst:
                burst_count += 1
                in_burst = True
        else:
            in_burst = False

    return {
        "peak": peak,
        "rms": rms,
        "crest_factor_db": crest_factor_db,
        "low_freq_ratio": low_freq_ratio,
        "mid_freq_ratio": mid_freq_ratio,
        "high_freq_crack_ratio": high_freq_crack_ratio,
        "spectral_centroid": spectral_centroid,
        "burst_count": burst_count
    }

def classify_sound_3way(max_probs, acoustic_feats):
    """
    Classify sound into Real_Gunshot, Fake_Gunshot, or Not_Gunshot.
    Combines primary CNN/CRNN detections with acoustic physics discrimination.
    """
    p_enh2d = max_probs.get("02_enhanced_2d", 0.0)
    p_crnn = max_probs.get("04_robust_crnn", 0.0)
    p_1d = max_probs.get("01_1d_cnn", 0.0)

    peak = acoustic_feats["peak"]
    crest = acoustic_feats["crest_factor_db"]
    low = acoustic_feats["low_freq_ratio"]
    high = acoustic_feats["high_freq_crack_ratio"]
    centroid = acoustic_feats["spectral_centroid"]

    # Model says gunshot
    is_gun_model = (
        (p_enh2d >= 0.45) or
        (p_crnn >= 0.50) or
        (p_enh2d >= 0.35 and p_crnn >= 0.35) or
        (p_enh2d >= 0.35 and p_1d >= 0.70)
    )

    if is_gun_model:
        # Check acoustic imposter signatures that fool CNNs
        # 1. Hand Clap signature: high centroid (>3000 Hz) with 0% low-end blast (<2%)
        if low < 0.02 and centroid > 3000.0:
            return "Fake_Gunshot", "Acoustic Imposter (Clap signature: High centroid >3kHz, 0% low blast)"
        # 2. Heavy Door Slam signature: pure low thud (>88% low) with no supersonic crack (<1% high)
        if low > 0.88 and high < 0.01 and centroid < 250.0:
            return "Fake_Gunshot", "Acoustic Imposter (Door slam signature: Low thud >88%, no bullet crack)"
        
        return "Real_Gunshot", f"Authentic Gunshot (CNN: Enh2D={p_enh2d*100:.1f}%, CRNN={p_crnn*100:.1f}%)"

    else:
        # Model rejected gunshot. Is it an acoustic imposter or ordinary ambient?
        # Imposters have sharp transients (high crest factor or peak burst)
        if (crest > 13.5 and peak > 0.08) or (crest > 18.0 and peak > 0.04) or (low > 0.75 and peak > 0.20):
            return "Fake_Gunshot", f"Acoustic Imposter (Transient impulse [Crest: {crest:.1f}dB], rejected by CNN)"
        else:
            return "Not_Gunshot", f"Not Gunshot / Ambient (Continuous background, CNN={max(p_enh2d, p_crnn)*100:.1f}%)"

def evaluate_single_file(models, audio_data, sr):
    if audio_data.ndim > 1:
        y = audio_data.mean(axis=1)
    else:
        y = audio_data.flatten()

    y = fast_resample(y, sr, TARGET_SR)

    if len(y) < WINDOW_SAMPLES:
        y = np.pad(y, (0, WINDOW_SAMPLES - len(y)))

    total_len = len(y)
    num_total_windows = max(1, int(np.ceil((total_len - WINDOW_SAMPLES) / HOP_SAMPLES)) + 1)

    # Compute RMS for all windows
    window_candidates = []
    for i in range(num_total_windows):
        start = i * HOP_SAMPLES
        end = start + WINDOW_SAMPLES
        w = y[total_len - WINDOW_SAMPLES:] if end > total_len else y[start:end]
        actual_start = total_len - WINDOW_SAMPLES if end > total_len else start
        rms = float(np.sqrt(np.mean(w ** 2)))
        window_candidates.append((actual_start, rms))

    # Pick top 20 by energy + up to 6 by time
    if len(window_candidates) > 25:
        by_energy = sorted(window_candidates, key=lambda x: -x[1])[:20]
        step = max(1, len(window_candidates) // 6)
        by_time = [window_candidates[k] for k in range(0, len(window_candidates), step)][:6]
        # Keep energy sorted so loudest windows are evaluated first!
        combined_starts = [item[0] for item in by_energy]
        for item in by_time:
            if item[0] not in combined_starts:
                combined_starts.append(item[0])
    else:
        by_energy = sorted(window_candidates, key=lambda x: -x[1])
        combined_starts = [item[0] for item in by_energy]

    max_probs = {m.id: 0.0 for m in models}
    best_transient_window = y[combined_starts[0]:combined_starts[0] + WINDOW_SAMPLES]
    max_rms = -1.0

    for start in combined_starts:
        w = y[start:start + WINDOW_SAMPLES]
        if len(w) < WINDOW_SAMPLES:
            w = np.pad(w, (0, WINDOW_SAMPLES - len(w)))

        rms = float(np.sqrt(np.mean(w ** 2)))
        if rms > max_rms:
            max_rms = rms
            best_transient_window = w

        if np.max(np.abs(w)) < 1e-4:
            continue

        for m in models:
            prob, _, _ = m.predict(w, native_sr=TARGET_SR)
            if prob > max_probs[m.id]:
                max_probs[m.id] = float(prob)

        # Early exit if top models already have high confidence
        if max_probs["02_enhanced_2d"] >= 0.95 and max_probs["04_robust_crnn"] >= 0.95:
            break

    acoustic_feats = extract_acoustic_features(best_transient_window, sr=TARGET_SR)
    return max_probs, acoustic_feats

def generate_html_report(dashboard_path, summary_data, cm, file_records, models):
    total_files = summary_data["total_files"]
    real_caught = summary_data["real_caught"]
    real_total = summary_data["real_total"]
    fake_rejected = summary_data["fake_rejected"]
    fake_total = summary_data["fake_total"]
    not_ignored = summary_data["not_ignored"]
    not_total = summary_data["not_total"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Shoot_Catcher — Audio Distinction & Segregation Report</title>
    <style>
        :root {{
            --bg: #090d16;
            --card-bg: #131b2e;
            --border: #1e293b;
            --text: #f1f5f9;
            --muted: #94a3b8;
            --primary: #38bdf8;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --purple: #a855f7;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 28px;
            line-height: 1.5;
        }}
        h1 {{ font-size: 28px; margin-bottom: 6px; color: var(--primary); display: flex; align-items: center; gap: 10px; }}
        p.subtitle {{ color: var(--muted); font-size: 14px; margin-top: 0; margin-bottom: 24px; }}
        .badge-bar {{ display: flex; gap: 10px; margin-bottom: 24px; flex-wrap: wrap; }}
        .badge-item {{ background: #1e293b; padding: 6px 14px; border-radius: 9999px; font-size: 13px; font-weight: 500; border: 1px solid #334155; }}
        
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 28px; }}
        .kpi-card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3); }}
        .kpi-val {{ font-size: 32px; font-weight: 800; color: var(--primary); }}
        .kpi-lbl {{ font-size: 12px; font-weight: 600; text-transform: uppercase; color: var(--muted); letter-spacing: 0.5px; margin-top: 4px; }}
        .kpi-sub {{ font-size: 13px; color: var(--muted); margin-top: 6px; }}

        .section-title {{ font-size: 20px; font-weight: 700; margin-top: 36px; margin-bottom: 16px; color: #e2e8f0; }}
        
        table {{ width: 100%; border-collapse: collapse; background: var(--card-bg); border-radius: 12px; overflow: hidden; margin-bottom: 32px; font-size: 13px; border: 1px solid var(--border); }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid var(--border); }}
        th {{ background: #0f172a; color: var(--muted); text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }}
        tr:hover {{ background: #1a253c; }}

        .tag {{ display: inline-block; padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 700; text-transform: uppercase; }}
        .tag-real {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }}
        .tag-fake {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }}
        .tag-not {{ background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }}
        .tag-match {{ color: #10b981; font-weight: 700; }}
        .tag-mismatch {{ color: #ef4444; font-weight: 700; }}

        .cm-table {{ max-width: 650px; margin-bottom: 32px; }}
        .cm-table th, .cm-table td {{ text-align: center; }}
        .cm-diag {{ background: rgba(16, 185, 129, 0.15); font-weight: 800; color: #34d399; }}

        .search-bar {{ width: 100%; max-width: 400px; padding: 10px 14px; background: #1e293b; border: 1px solid #334155; border-radius: 8px; color: #fff; font-size: 13px; margin-bottom: 16px; }}
        .search-bar:focus {{ outline: none; border-color: var(--primary); }}
    </style>
</head>
<body>
    <h1>🔫 Shoot_Catcher — Audio Distinction & Segregation Dashboard</h1>
    <p class="subtitle">Complete 3-Way Classification & Evaluation across 165 Audio Files in <code>test_dataset</code></p>

    <div class="badge-bar">
        <span class="badge-item">📁 Segregated Audio Folders: <code>Verification_Outputs/Classified_Audio/</code></span>
        <span class="badge-item">⚡ 750ms Window (75% Overlap)</span>
        <span class="badge-item">🎯 Total Evaluated: {total_files} Files</span>
    </div>

    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-val" style="color: #34d399;">{real_caught} / {real_total}</div>
            <div class="kpi-lbl">Real Gunshots Caught</div>
            <div class="kpi-sub">{real_caught / max(1, real_total) * 100:.1f}% Sensitivity / Recall</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-val" style="color: #fbbf24;">{fake_rejected} / {fake_total}</div>
            <div class="kpi-lbl">Fake Gunshots Separated</div>
            <div class="kpi-sub">{fake_rejected / max(1, fake_total) * 100:.1f}% Imposter Separation</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-val" style="color: #60a5fa;">{not_ignored} / {not_total}</div>
            <div class="kpi-lbl">Ambient Noise Ignored</div>
            <div class="kpi-sub">{not_ignored / max(1, not_total) * 100:.1f}% Ambient Specificity</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-val" style="color: #a855f7;">{summary_data['overall_accuracy']:.1f}%</div>
            <div class="kpi-lbl">Overall 3-Way Accuracy</div>
            <div class="kpi-sub">{summary_data['total_correct']} of {total_files} correctly segregated</div>
        </div>
    </div>

    <h2 class="section-title">📊 3x3 Multi-Class Confusion Matrix</h2>
    <table class="cm-table">
        <thead>
            <tr>
                <th>True Ground Truth \\ Classified As</th>
                <th>Pred: Real Gunshot</th>
                <th>Pred: Fake Gunshot</th>
                <th>Pred: Not Gunshot</th>
                <th>Total Files</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>1_gunshots (Real Firearms)</strong></td>
                <td class="cm-diag">{cm['Real_Gunshot']['Real_Gunshot']}</td>
                <td>{cm['Real_Gunshot']['Fake_Gunshot']}</td>
                <td>{cm['Real_Gunshot']['Not_Gunshot']}</td>
                <td><strong>{real_total}</strong></td>
            </tr>
            <tr>
                <td><strong>3_imposters (Fake Gunshots)</strong></td>
                <td>{cm['Fake_Gunshot']['Real_Gunshot']}</td>
                <td class="cm-diag">{cm['Fake_Gunshot']['Fake_Gunshot']}</td>
                <td>{cm['Fake_Gunshot']['Not_Gunshot']}</td>
                <td><strong>{fake_total}</strong></td>
            </tr>
            <tr>
                <td><strong>2_nongunshots (Ambient Noise)</strong></td>
                <td>{cm['Not_Gunshot']['Real_Gunshot']}</td>
                <td>{cm['Not_Gunshot']['Fake_Gunshot']}</td>
                <td class="cm-diag">{cm['Not_Gunshot']['Not_Gunshot']}</td>
                <td><strong>{not_total}</strong></td>
            </tr>
        </tbody>
    </table>

    <h2 class="section-title">🤖 Per-Model Deep Learning Benchmarks</h2>
    <table>
        <thead>
            <tr>
                <th>Model Architecture</th>
                <th>Overall Acc</th>
                <th>Gunshots Caught</th>
                <th>Fake Gunshots Rejected</th>
                <th>Ambient Ignored</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
            </tr>
        </thead>
        <tbody>
    """
    for m in models:
        stats = summary_data["model_benchmarks"][m.id]
        html += f"""
            <tr>
                <td><strong>{m.name}</strong></td>
                <td><strong>{stats['acc']*100:.1f}%</strong></td>
                <td>{stats['real_caught_str']}</td>
                <td>{stats['fake_rejected_str']}</td>
                <td>{stats['not_ignored_str']}</td>
                <td>{stats['prec']*100:.1f}%</td>
                <td>{stats['rec']*100:.1f}%</td>
                <td><strong>{stats['f1']*100:.1f}%</strong></td>
            </tr>
        """
    html += """
        </tbody>
    </table>

    <h2 class="section-title">📂 Segregated Audio File Records (All 165 Files)</h2>
    <input type="text" id="filterInput" class="search-bar" placeholder="🔍 Search by file name, category, or classification..." onkeyup="filterTable()">
    <table id="fileTable">
        <thead>
            <tr>
                <th>File Name</th>
                <th>Ground Truth</th>
                <th>Final Classified Category</th>
                <th>Enhanced 2D Prob</th>
                <th>CRNN Prob</th>
                <th>Crest Factor</th>
                <th>Low Freq (Blast)</th>
                <th>High Freq (Crack)</th>
                <th>Segregated Folder Destination</th>
                <th>Rationale / Physics Signature</th>
            </tr>
        </thead>
        <tbody>
    """
    for r in file_records:
        tag_true = "tag-real" if "gunshot" in r["true_cat"].lower() and "non" not in r["true_cat"].lower() else ("tag-fake" if "imposter" in r["true_cat"].lower() else "tag-not")
        tag_pred = "tag-real" if r["pred_class"] == "Real_Gunshot" else ("tag-fake" if r["pred_class"] == "Fake_Gunshot" else "tag-not")
        match_icon = "✅ Match" if r["is_match"] else "⚠️ Divergence"
        match_cls = "tag-match" if r["is_match"] else "tag-mismatch"

        html += f"""
            <tr>
                <td><code>{r['filename']}</code></td>
                <td><span class="tag {tag_true}">{r['true_cat']}</span></td>
                <td><span class="tag {tag_pred}">{r['pred_class']}</span> <span class="{match_cls}">{match_icon}</span></td>
                <td><strong>{r['prob_enh2d']*100:.1f}%</strong></td>
                <td>{r['prob_crnn']*100:.1f}%</td>
                <td>{r['crest_db']:.1f} dB</td>
                <td>{r['low_ratio']*100:.1f}%</td>
                <td>{r['high_ratio']*100:.1f}%</td>
                <td><code>{r['dest_folder']}</code></td>
                <td style="color: var(--muted); font-size: 12px;">{r['rationale']}</td>
            </tr>
        """
    html += """
        </tbody>
    </table>

    <script>
        function filterTable() {
            var input = document.getElementById("filterInput");
            var filter = input.value.toLowerCase();
            var table = document.getElementById("fileTable");
            var tr = table.getElementsByTagName("tr");
            for (var i = 1; i < tr.length; i++) {
                var text = tr[i].textContent || tr[i].innerText;
                if (text.toLowerCase().indexOf(filter) > -1) {
                    tr[i].style.display = "";
                } else {
                    tr[i].style.display = "none";
                }
            }
        }
    </script>
</body>
</html>
    """
    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(html)

def run_distinction_pipeline(dataset_dir, threshold=DEFAULT_THRESHOLD, copy_files=True):
    dir_gunshots, dir_nongunshots, dir_imposters = locate_dataset_folders(dataset_dir)

    print("=" * 85)
    print("🎯 SHOOT_CATCHER — AUDIO DISTINCTION & SEGREGATION PIPELINE")
    print("=" * 85)
    print(f"Dataset Root Directory : {dataset_dir}")
    print(f"1. Real Gunshots Dir   : {dir_gunshots} ({len(find_audio_files(dir_gunshots)) if dir_gunshots else 0} files)")
    print(f"2. Non-Gunshots Dir    : {dir_nongunshots} ({len(find_audio_files(dir_nongunshots)) if dir_nongunshots else 0} files)")
    print(f"3. Imposters Dir       : {dir_imposters} ({len(find_audio_files(dir_imposters)) if dir_imposters else 0} files)")
    print(f"Decision Threshold     : {threshold:.2f}")
    print("=" * 85)

    if not dir_gunshots or not dir_nongunshots or not dir_imposters:
        print("❌ Could not locate all 3 target subdirectories!")
        return

    gunshot_files = find_audio_files(dir_gunshots)
    nongunshot_files = find_audio_files(dir_nongunshots)
    imposter_files = find_audio_files(dir_imposters)

    total_files = len(gunshot_files) + len(nongunshot_files) + len(imposter_files)
    if total_files == 0:
        print("❌ No audio files found in target directories!")
        return

    # Load all models
    mgr = live_demo.ModelManager()
    mgr.print_audit_table()
    models = mgr.trained_models
    print(f"\n⚡ Active Loaded Models: {len(models)}\n")

    setup_output_dirs(models)

    items = []
    for f in gunshot_files:
        items.append((f, "Real_Gunshot", "1_gunshots", 1))
    for f in imposter_files:
        items.append((f, "Fake_Gunshot", "3_imposters", 0))
    for f in nongunshot_files:
        items.append((f, "Not_Gunshot", "2_nongunshots", 0))

    model_stats = {
        m.id: {
            "name": m.name,
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "real_caught": 0, "fake_rejected": 0, "not_ignored": 0
        } for m in models
    }

    confusion_matrix = {
        "Real_Gunshot": {"Real_Gunshot": 0, "Fake_Gunshot": 0, "Not_Gunshot": 0},
        "Fake_Gunshot": {"Real_Gunshot": 0, "Fake_Gunshot": 0, "Not_Gunshot": 0},
        "Not_Gunshot": {"Real_Gunshot": 0, "Fake_Gunshot": 0, "Not_Gunshot": 0},
    }

    file_records = []
    start_time = time.time()

    print(f"🚀 Processing {total_files} audio files across all models...\n")

    for idx, (fpath, true_class, true_folder_name, binary_label) in enumerate(items, 1):
        abs_path = str(fpath.resolve())
        long_path = "\\\\?\\" + abs_path if sys.platform == "win32" else abs_path

        try:
            audio, sr = sf.read(long_path)
        except Exception as e:
            print(f"[{idx:3d}/{total_files}] ⚠️ Read error {fpath.name}: {e}")
            continue

        max_probs, acoustic_feats = evaluate_single_file(models, audio, sr)

        # 3-Way Classification
        pred_class, rationale = classify_sound_3way(max_probs, acoustic_feats)
        confusion_matrix[true_class][pred_class] += 1
        is_match = (pred_class == true_class)

        # Binary evaluations per model
        for m in models:
            prob = max_probs[m.id]
            pred_bin = 1 if prob >= threshold else 0
            if binary_label == 1 and pred_bin == 1:
                model_stats[m.id]["tp"] += 1
                model_stats[m.id]["real_caught"] += 1
            elif binary_label == 0 and pred_bin == 0:
                model_stats[m.id]["tn"] += 1
                if true_class == "Fake_Gunshot":
                    model_stats[m.id]["fake_rejected"] += 1
                else:
                    model_stats[m.id]["not_ignored"] += 1
            elif binary_label == 0 and pred_bin == 1:
                model_stats[m.id]["fp"] += 1
            else:
                model_stats[m.id]["fn"] += 1

            if copy_files:
                m_slug = m.name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                sub_folder = "Predicted_Gunshots" if pred_bin == 1 else "Predicted_NonGunshots"
                dest_m = OUTPUT_BASE / "By_Model" / m_slug / sub_folder / fpath.name
                if not dest_m.exists():
                    try:
                        shutil.copy2(long_path, str(dest_m))
                    except Exception:
                        pass

        dest_folder = DIR_REAL if pred_class == "Real_Gunshot" else (DIR_FAKE if pred_class == "Fake_Gunshot" else DIR_NOT)
        if copy_files:
            target_dest = dest_folder / fpath.name
            try:
                shutil.copy2(long_path, str(target_dest))
            except Exception:
                pass

        file_records.append({
            "filename": fpath.name,
            "true_cat": true_folder_name,
            "true_class": true_class,
            "pred_class": pred_class,
            "is_match": is_match,
            "dest_folder": f"Classified_Audio/{dest_folder.name}/",
            "prob_enh2d": max_probs.get("02_enhanced_2d", 0.0),
            "prob_crnn": max_probs.get("04_robust_crnn", 0.0),
            "prob_1d": max_probs.get("01_1d_cnn", 0.0),
            "crest_db": acoustic_feats["crest_factor_db"],
            "low_ratio": acoustic_feats["low_freq_ratio"],
            "high_ratio": acoustic_feats["high_freq_crack_ratio"],
            "centroid": acoustic_feats["spectral_centroid"],
            "rationale": rationale
        })

        icon = "✅" if is_match else "⚠️"
        p_enh = max_probs.get('02_enhanced_2d', 0.0) * 100
        p_crnn = max_probs.get('04_robust_crnn', 0.0) * 100
        print(f"[{idx:3d}/{total_files}] {icon} {fpath.name[:25]:<25} | True: {true_class:<12} -> Pred: {pred_class:<12} (Enh2D: {p_enh:5.1f}%, CRNN: {p_crnn:5.1f}%)")

    elapsed = time.time() - start_time
    print("\n" + "=" * 90)
    print(f"🏁 PROCESSING COMPLETE in {elapsed:.1f}s ({elapsed/max(1, total_files):.2f}s / file)")
    print("=" * 90)

    # Summary Metrics
    real_caught = confusion_matrix["Real_Gunshot"]["Real_Gunshot"]
    real_total = len(gunshot_files)
    fake_as_fake = confusion_matrix["Fake_Gunshot"]["Fake_Gunshot"]
    fake_rejected = confusion_matrix["Fake_Gunshot"]["Fake_Gunshot"] + confusion_matrix["Fake_Gunshot"]["Not_Gunshot"]
    fake_total = len(imposter_files)
    not_ignored = confusion_matrix["Not_Gunshot"]["Not_Gunshot"]
    not_total = len(nongunshot_files)
    total_correct = sum(confusion_matrix[c][c] for c in ["Real_Gunshot", "Fake_Gunshot", "Not_Gunshot"])
    overall_acc = (total_correct / max(1, total_files)) * 100.0

    print("\n📊 3-WAY SEGREGATION SUMMARY:")
    print(f"   • Real Gunshots Caught (1_gunshots)    : {real_caught}/{real_total} ({real_caught/max(1, real_total)*100:.1f}%)")
    print(f"   • Fake Gunshots Classified as Fake     : {fake_as_fake}/{fake_total} ({fake_as_fake/max(1, fake_total)*100:.1f}%)")
    print(f"   • Fake Gunshots Rejected as Gunshots   : {fake_rejected}/{fake_total} ({fake_rejected/max(1, fake_total)*100:.1f}%)")
    print(f"   • Ambient Noise Ignored (2_nongunshots): {not_ignored}/{not_total} ({not_ignored/max(1, not_total)*100:.1f}%)")
    print(f"   • Overall 3-Way Accuracy               : {total_correct}/{total_files} ({overall_acc:.2f}%)")

    print("\n📊 3x3 CONFUSION MATRIX:")
    print(f"   {'True Ground Truth':<25} | {'Pred: Real':<12} | {'Pred: Fake':<12} | {'Pred: Not':<12}")
    print("   " + "-" * 65)
    for true_c in ["Real_Gunshot", "Fake_Gunshot", "Not_Gunshot"]:
        print(f"   {true_c:<25} | {confusion_matrix[true_c]['Real_Gunshot']:<12} | {confusion_matrix[true_c]['Fake_Gunshot']:<12} | {confusion_matrix[true_c]['Not_Gunshot']:<12}")

    # Compile Model Benchmarks
    model_benchmarks = {}
    print("\n" + "=" * 90)
    print("🤖 PER-MODEL BINARY EVALUATION BENCHMARK:")
    print(f"{'Model Name':<28} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1':<7} | {'Gunshots':<10} | {'Fake Rej':<10} | {'Ambient':<10}")
    print("-" * 90)
    for m in models:
        s = model_stats[m.id]
        tp, fp, tn, fn = s["tp"], s["fp"], s["tn"], s["fn"]
        acc = (tp + tn) / max(1, (tp + tn + fp + fn))
        prec = tp / max(1, (tp + fp))
        rec = tp / max(1, (tp + fn))
        f1 = (2 * prec * rec) / max(1e-6, (prec + rec))

        act_caught_str = f"{s['real_caught']}/{real_total}"
        fake_rej_str = f"{s['fake_rejected']}/{fake_total}"
        not_ign_str = f"{s['not_ignored']}/{not_total}"

        model_benchmarks[m.id] = {
            "name": m.name, "acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "real_caught_str": act_caught_str,
            "fake_rejected_str": fake_rej_str,
            "not_ignored_str": not_ign_str
        }
        print(f"{m.name:<28} | {acc*100:6.2f}% | {prec*100:7.2f}% | {rec*100:6.2f}% | {f1*100:5.2f}% | {act_caught_str:<10} | {fake_rej_str:<10} | {not_ign_str:<10}")
    print("=" * 90)

    summary_data = {
        "total_files": total_files,
        "real_caught": real_caught,
        "real_total": real_total,
        "fake_rejected": fake_rejected,
        "fake_total": fake_total,
        "not_ignored": not_ignored,
        "not_total": not_total,
        "total_correct": total_correct,
        "overall_accuracy": overall_acc,
        "model_benchmarks": model_benchmarks
    }

    # Save CSV
    csv_path = OUTPUT_BASE / "test_dataset_classification_scorecard.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Filename", "True_Folder", "True_Class", "Predicted_Class", "Is_Match",
            "Dest_Folder", "Prob_Enhanced_2D", "Prob_Robust_CRNN", "Prob_Baseline_1D",
            "Crest_Factor_dB", "Low_Freq_Blast_Ratio", "High_Freq_Crack_Ratio",
            "Spectral_Centroid_Hz", "Rationale"
        ])
        for r in file_records:
            writer.writerow([
                r["filename"], r["true_cat"], r["true_class"], r["pred_class"], r["is_match"],
                r["dest_folder"], f"{r['prob_enh2d']:.4f}", f"{r['prob_crnn']:.4f}", f"{r['prob_1d']:.4f}",
                f"{r['crest_db']:.2f}", f"{r['low_ratio']:.4f}", f"{r['high_ratio']:.4f}",
                f"{r['centroid']:.1f}", r["rationale"]
            ])
    print(f"\n📁 Saved Scorecard CSV    : {csv_path}")

    # Save HTML Dashboard
    html_path = OUTPUT_BASE / "test_dataset_dashboard.html"
    generate_html_report(html_path, summary_data, confusion_matrix, file_records, models)
    print(f"🌐 Saved HTML Dashboard   : {html_path}")
    print(f"📂 Audio Segregated Into  : {CLASSIFIED_DIR}")

    return summary_data, confusion_matrix, file_records

def main():
    parser = argparse.ArgumentParser(description="Shoot_Catcher Test Dataset Distinction & Segregation")
    parser.add_argument("--dataset-dir", type=str, default=str(PROJECT_ROOT / "test_dataset"), help="Path to test_dataset folder")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--no-copy", action="store_true", help="Disable physical file copying")
    args = parser.parse_args()

    run_distinction_pipeline(args.dataset_dir, threshold=args.threshold, copy_files=not args.no_copy)

if __name__ == "__main__":
    main()
