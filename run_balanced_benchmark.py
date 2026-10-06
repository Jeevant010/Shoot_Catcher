"""
===============================================================================
🎯 Shoot_Catcher — Strictly Balanced 1:1:1 Multi-Model Benchmark Runner
===============================================================================
Evaluates all 5 trained models on a rigorously balanced 150-file test dataset:
  - 50 Authentic Gunshots
  - 50 Imposters / Fake Gunshots (firecrackers, claps, door slams, pops)
  - 50 Ambient Noise / Non-Gunshots (rain, traffic, sirens, dogs, talking)

Produces:
  1. Detailed per-file CSV Scorecard
  2. Summary Metrics CSV (Accuracy, Precision, Recall, F1, F2, Breakdown)
  3. Interactive Presentation Dashboard HTML
===============================================================================
"""

import os
import sys
import io
import json
import numpy as np
import scipy.signal as signal
from pathlib import Path
import soundfile as sf

# Force UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras

PROJECT_ROOT = Path(__file__).parent.resolve()
VERIF_DIR = PROJECT_ROOT / "Verification_Outputs"
VERIF_DIR.mkdir(parents=True, exist_ok=True)

# Import local PCEN pipeline
sys.path.insert(0, str(PROJECT_ROOT / "04_Robust_CRNN_PCEN"))
from pcen_mic_pipeline import compute_pcen

TARGET_SR = 22050
TARGET_SAMPLES = 16537  # 750ms
HOP_SAMPLES = int(TARGET_SR * 0.1875)  # 75% overlap = 187.5ms hop
CONFIDENCE_THRESHOLD = 0.50


class CompatBatchNormalization(keras.layers.BatchNormalization):
    def __init__(self, **kwargs):
        kwargs.pop('renorm', None)
        kwargs.pop('renorm_clipping', None)
        kwargs.pop('renorm_momentum', None)
        super().__init__(**kwargs)


def resample_audio(y, orig_sr, target_sr):
    if orig_sr == target_sr:
        return y.astype(np.float32)
    num_samples = int(len(y) * target_sr / orig_sr)
    return signal.resample(y, num_samples).astype(np.float32)


def compute_mel_spectrogram_scipy(y, sr=22050, n_mels=64, n_fft=512, hop_length=128):
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


def extract_dsp_features(y, sr=22050):
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y ** 2)))
    crest_db = float(20 * np.log10(peak / max(rms, 1e-6)))

    fft_mag = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1.0 / sr)

    low_mask = (freqs >= 100) & (freqs <= 600)
    high_mask = (freqs >= 1500) & (freqs <= 4500)
    total_energy = np.sum(fft_mag ** 2) + 1e-9

    low_ratio = float(np.sum(fft_mag[low_mask] ** 2) / total_energy)
    high_ratio = float(np.sum(fft_mag[high_mask] ** 2) / total_energy)
    centroid = float(np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-9))

    return crest_db, low_ratio, high_ratio, centroid


class ModelEvaluator:
    def __init__(self):
        self.models = {}
        custom = {'BatchNormalization': CompatBatchNormalization}

        specs = [
            ('Baseline 1D CNN', PROJECT_ROOT / '01_1D_CNN/output/1d_cnn_best.h5', False, False),
            ('Baseline 2D CNN (Mel)', PROJECT_ROOT / '02_2D_CNN/output/2d_cnn_mel_spectrogram_best.h5', True, False),
            ('Enhanced 1D CNN (Dual)', PROJECT_ROOT / 'Enhanced_Models/01_Enhanced_1D_CNN/output/enhanced_1d_cnn_best.h5', False, True),
            ('Enhanced 2D CNN (Dual)', PROJECT_ROOT / 'Enhanced_Models/02_Enhanced_2D_CNN/output/enhanced_2d_cnn_best.h5', True, True),
            ('Robust CRNN (PCEN)', PROJECT_ROOT / '04_Robust_CRNN_PCEN/output/crnn_pcen_best.h5', 'crnn', False),
        ]

        # Load PCEN stats
        pcen_stats_p = PROJECT_ROOT / "04_Robust_CRNN_PCEN/output/pcen_stats.json"
        self.pcen_mean, self.pcen_std = 0.4390, 0.1876
        if pcen_stats_p.exists():
            try:
                st = json.loads(pcen_stats_p.read_text())
                self.pcen_mean = st.get('mean', self.pcen_mean)
                self.pcen_std = st.get('std', self.pcen_std)
            except Exception:
                pass

        for name, path, is_2d, is_dual in specs:
            if path.exists():
                try:
                    m = keras.models.load_model(str(path), custom_objects=custom, compile=False)
                    self.models[name] = {
                        'model': m,
                        'is_2d': is_2d,
                        'is_dual': is_dual,
                        'input_shape': m.input_shape
                    }
                    print(f" Loaded: {name} (shape: {m.input_shape})")
                except Exception as e:
                    print(f" Failed loading {name}: {e}")
            else:
                print(f" Missing file: {name} at {path}")

    def predict_window(self, y_win):
        # Conditioning: DC removal
        y = y_win - np.mean(y_win)
        rms = np.sqrt(np.mean(y ** 2))
        if rms >= 0.001:
            peak = np.max(np.abs(y))
            if peak > 1e-6:
                y = y / peak
        else:
            y = np.clip(y, -1.0, 1.0)

        preds = {}
        for name, info in self.models.items():
            m = info['model']
            is_2d = info['is_2d']
            is_dual = info['is_dual']

            if is_2d == 'crnn':
                # PCEN for CRNN
                pcen = compute_pcen(y, sr=TARGET_SR, n_mels=64, n_fft=512, hop_length=128)
                pcen_norm = (pcen - self.pcen_mean) / max(self.pcen_std, 1e-6)
                if pcen_norm.shape[1] < 130:
                    pcen_norm = np.pad(pcen_norm, ((0, 0), (0, 130 - pcen_norm.shape[1])))
                else:
                    pcen_norm = pcen_norm[:, :130]
                x = pcen_norm.reshape(1, 64, 130, 1).astype(np.float32)
                out = m.predict(x, verbose=0)

            elif is_2d:
                # 2D Mel
                target_t = info['input_shape'][2] if len(info['input_shape']) > 2 and info['input_shape'][2] else 130
                spec = compute_mel_spectrogram_scipy(y, sr=TARGET_SR, n_mels=64, n_fft=512, hop_length=128)
                if spec.shape[1] < target_t:
                    spec = np.pad(spec, ((0, 0), (0, target_t - spec.shape[1])))
                else:
                    spec = spec[:, :target_t]
                spec_norm = np.clip((spec + 80.0) / 80.0, 0.0, 1.0)
                x = spec_norm.reshape(1, 64, target_t, 1).astype(np.float32)
                out = m.predict(x, verbose=0)

            else:
                # 1D Raw
                x = y.reshape(1, TARGET_SAMPLES, 1).astype(np.float32)
                out = m.predict(x, verbose=0)

            if is_dual:
                if isinstance(out, dict):
                    prob = float(out.get('gunshot_output', list(out.values())[0]).flatten()[0])
                elif isinstance(out, (list, tuple)):
                    prob = float(out[0].flatten()[0])
                else:
                    prob = float(out.flatten()[0])
            else:
                if isinstance(out, dict):
                    prob = float(list(out.values())[0].flatten()[0])
                else:
                    prob = float(out.flatten()[0])

            preds[name] = prob

        return preds

    def evaluate_file(self, wav_path):
        data, sr = sf.read(str(wav_path))
        if data.ndim > 1:
            data = data.mean(axis=1)

        y_all = resample_audio(data, sr, TARGET_SR)

        # Slice into 750ms windows with 75% overlap
        if len(y_all) < TARGET_SAMPLES:
            y_pad = np.pad(y_all, (0, TARGET_SAMPLES - len(y_all)))
            windows = [y_pad]
        else:
            windows = []
            for start in range(0, len(y_all) - TARGET_SAMPLES + 1, HOP_SAMPLES):
                windows.append(y_all[start:start + TARGET_SAMPLES])
            if not windows:
                windows = [y_all[:TARGET_SAMPLES]]

        # Max confidence across all windows in this file
        max_preds = {name: 0.0 for name in self.models}
        peak_win = windows[0]
        max_rms = -1.0

        for w in windows:
            rms = np.sqrt(np.mean(w ** 2))
            if rms > max_rms:
                max_rms = rms
                peak_win = w

            w_preds = self.predict_window(w)
            for name, p in w_preds.items():
                if p > max_preds[name]:
                    max_preds[name] = p

        crest, blast_r, crack_r, centroid = extract_dsp_features(peak_win, TARGET_SR)
        return max_preds, (crest, blast_r, crack_r, centroid)


def main():
    print("=" * 75)
    print("🎯 STARTING STRICTLY BALANCED 1:1:1 MULTI-MODEL BENCHMARK (150 FILES)")
    print("=" * 75)

    base = PROJECT_ROOT / "test_dataset/test_dataset"
    gun_files = sorted(list((base / "1_gunshots").glob("*.wav")))[:50]
    imp_files = sorted(list((base / "3_imposters").glob("*.wav")))[:50]
    amb_files = sorted(list((base / "2_nongunshots").glob("*.wav")))[:50]

    test_cohort = []
    for f in gun_files:
        test_cohort.append((f, "Real_Gunshot", "1_gunshots"))
    for f in imp_files:
        test_cohort.append((f, "Fake_Gunshot", "3_imposters"))
    for f in amb_files:
        test_cohort.append((f, "Not_Gunshot", "2_nongunshots"))

    print(f"Total Cohort Size: {len(test_cohort)} files")
    print(f" - Real Gunshots:  {len(gun_files)}")
    print(f" - Fake Gunshots:  {len(imp_files)}")
    print(f" - Ambient Noise:  {len(amb_files)}")
    print("=" * 75)

    evaluator = ModelEvaluator()
    model_names = list(evaluator.models.keys())

    scorecard_rows = []
    cohort_results = {m: {'tp': 0, 'fp_fake': 0, 'fp_amb': 0, 'tn_fake': 0, 'tn_amb': 0, 'fn': 0, 'caught': 0}
                      for m in model_names}

    print("\nProcessing audio files...")
    for idx, (wav_path, true_class, folder_label) in enumerate(test_cohort, 1):
        preds, (crest, blast_r, crack_r, centroid) = evaluator.evaluate_file(wav_path)

        # Update per-model stats
        for m in model_names:
            prob = preds[m]
            detected = (prob >= CONFIDENCE_THRESHOLD)

            if true_class == "Real_Gunshot":
                if detected:
                    cohort_results[m]['tp'] += 1
                    cohort_results[m]['caught'] += 1
                else:
                    cohort_results[m]['fn'] += 1
            elif true_class == "Fake_Gunshot":
                if detected:
                    cohort_results[m]['fp_fake'] += 1
                else:
                    cohort_results[m]['tn_fake'] += 1
            else:  # Not_Gunshot (Ambient)
                if detected:
                    cohort_results[m]['fp_amb'] += 1
                else:
                    cohort_results[m]['tn_amb'] += 1

        # Rationale string based on best models
        enh2d = preds.get('Enhanced 2D CNN (Dual)', 0.0)
        crnn = preds.get('Robust CRNN (PCEN)', 0.0)
        b1d = preds.get('Baseline 1D CNN', 0.0)

        if true_class == "Real_Gunshot":
            rationale = f"Authentic Gunshot (CRNN={crnn*100:.1f}%, Enh2D={enh2d*100:.1f}%, 1D={b1d*100:.1f}%)"
        elif true_class == "Fake_Gunshot":
            rationale = f"Impulsive Imposter (Crest={crest:.1f}dB, CRNN={crnn*100:.1f}%, Enh2D={enh2d*100:.1f}%)"
        else:
            rationale = f"Continuous Ambient (Centroid={centroid:.0f}Hz, CRNN={crnn*100:.1f}%, Enh2D={enh2d*100:.1f}%)"

        scorecard_rows.append({
            'Filename': wav_path.name,
            'True_Folder': folder_label,
            'True_Class': true_class,
            'Prob_Robust_CRNN': preds.get('Robust CRNN (PCEN)', 0.0),
            'Prob_Enhanced_2D': preds.get('Enhanced 2D CNN (Dual)', 0.0),
            'Prob_Baseline_1D': preds.get('Baseline 1D CNN', 0.0),
            'Prob_Enhanced_1D': preds.get('Enhanced 1D CNN (Dual)', 0.0),
            'Prob_Baseline_2D': preds.get('Baseline 2D CNN (Mel)', 0.0),
            'Crest_Factor_dB': round(crest, 2),
            'Low_Freq_Blast_Ratio': round(blast_r, 4),
            'High_Freq_Crack_Ratio': round(crack_r, 4),
            'Spectral_Centroid_Hz': round(centroid, 1),
            'Rationale': rationale
        })

        if idx % 25 == 0 or idx == len(test_cohort):
            print(f"  [{idx:3d}/{len(test_cohort)}] Processed: {wav_path.name[:35]:<35} (Class: {true_class})")

    # 1. Export Scorecard CSV
    scorecard_csv = VERIF_DIR / "balanced_150_benchmark_scorecard.csv"
    with open(scorecard_csv, "w", encoding="utf-8") as f:
        headers = [
            "Filename", "True_Folder", "True_Class",
            "Prob_Robust_CRNN", "Prob_Enhanced_2D", "Prob_Baseline_1D", "Prob_Enhanced_1D", "Prob_Baseline_2D",
            "Crest_Factor_dB", "Low_Freq_Blast_Ratio", "High_Freq_Crack_Ratio", "Spectral_Centroid_Hz", "Rationale"
        ]
        f.write(",".join(headers) + "\n")
        for row in scorecard_rows:
            f.write(
                f"{row['Filename']},{row['True_Folder']},{row['True_Class']},"
                f"{row['Prob_Robust_CRNN']:.4f},{row['Prob_Enhanced_2D']:.4f},{row['Prob_Baseline_1D']:.4f},"
                f"{row['Prob_Enhanced_1D']:.4f},{row['Prob_Baseline_2D']:.4f},"
                f"{row['Crest_Factor_dB']},{row['Low_Freq_Blast_Ratio']},{row['High_Freq_Crack_Ratio']},"
                f"{row['Spectral_Centroid_Hz']},\"{row['Rationale']}\"\n"
            )

    # 2. Compute Summary Metrics
    metrics_summary = []
    print("\n" + "=" * 80)
    print("📊 STRICTLY BALANCED 1:1:1 BENCHMARK RESULTS (50 Gunshots, 50 Imposters, 50 Ambient)")
    print("=" * 80)

    for m in model_names:
        stats = cohort_results[m]
        tp = stats['tp']
        fn = stats['fn']
        fp = stats['fp_fake'] + stats['fp_amb']
        tn = stats['tn_fake'] + stats['tn_amb']

        total = tp + fn + fp + tn
        acc = (tp + tn) / max(total, 1)
        prec = tp / max(tp + fp, 1e-6)
        recall = tp / max(tp + fn, 1e-6)
        f1 = 2 * prec * recall / max(prec + recall, 1e-6)
        f2 = 5 * prec * recall / max(4 * prec + recall, 1e-6)

        gunshots_caught = f"{tp}/50"
        fakes_rejected = f"{stats['tn_fake']}/50"
        ambient_ignored = f"{stats['tn_amb']}/50"

        metrics_summary.append({
            'Model': m,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(recall, 4),
            'F1': round(f1, 4),
            'F2': round(f2, 4),
            'Gunshots_Caught': gunshots_caught,
            'Fakes_Rejected': fakes_rejected,
            'Ambient_Ignored': ambient_ignored,
            'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn
        })

        print(f"🤖 {m:<24} | Acc: {acc*100:5.1f}% | Recall: {recall*100:5.1f}% | Prec: {prec*100:5.1f}% | F2: {f2:.4f}")
        print(f"   ├─ Real Gunshots Caught: {gunshots_caught} ({recall*100:.1f}%)")
        print(f"   ├─ Imposters Rejected:   {fakes_rejected} ({stats['tn_fake']*2:.1f}%)")
        print(f"   └─ Ambient Noise Ignored:{ambient_ignored} ({stats['tn_amb']*2:.1f}%)")

    # Export Metrics CSV
    metrics_csv = VERIF_DIR / "balanced_150_benchmark_metrics.csv"
    with open(metrics_csv, "w", encoding="utf-8") as f:
        f.write("Model,Accuracy,Precision,Recall,F1,F2,Gunshots_Caught,Fakes_Rejected,Ambient_Ignored,TP,FP,TN,FN\n")
        for m in metrics_summary:
            f.write(
                f"{m['Model']},{m['Accuracy']:.4f},{m['Precision']:.4f},{m['Recall']:.4f},{m['F1']:.4f},{m['F2']:.4f},"
                f"{m['Gunshots_Caught']},{m['Fakes_Rejected']},{m['Ambient_Ignored']},{m['TP']},{m['FP']},{m['TN']},{m['FN']}\n"
            )

    # 3. Generate HTML Presentation Dashboard
    generate_html_dashboard(metrics_summary, scorecard_rows)
    print("\n" + "=" * 80)
    print(f"✅ Benchmark Complete! Saved Artifacts:")
    print(f"   1. CSV Scorecard: {scorecard_csv}")
    print(f"   2. Metrics CSV:   {metrics_csv}")
    print(f"   3. HTML Dashboard:{VERIF_DIR / 'balanced_150_benchmark_dashboard.html'}")
    print("=" * 80)


def generate_html_dashboard(metrics, scorecard):
    dashboard_path = VERIF_DIR / "balanced_150_benchmark_dashboard.html"

    # Best performing model by F2
    best_model = max(metrics, key=lambda x: x['F2'])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Shoot_Catcher — Strictly Balanced 1:1:1 Multi-Model Benchmark (150 Files)</title>
    <style>
        :root {{
            --bg: #090d16;
            --card: #131b2e;
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
        .header {{
            margin-bottom: 24px;
        }}
        h1 {{ font-size: 26px; color: var(--primary); margin: 0 0 6px 0; display: flex; align-items: center; gap: 10px; }}
        p.subtitle {{ color: var(--muted); font-size: 14px; margin: 0; }}
        
        .badge-bar {{ display: flex; gap: 10px; margin: 18px 0 24px 0; flex-wrap: wrap; }}
        .badge {{ background: #1e293b; padding: 6px 14px; border-radius: 9999px; font-size: 12px; font-weight: 600; border: 1px solid #334155; }}
        .badge-gold {{ background: rgba(245, 158, 11, 0.15); color: #fbbf24; border-color: rgba(245, 158, 11, 0.4); }}
        
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 28px; }}
        .kpi-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }}
        .kpi-val {{ font-size: 32px; font-weight: 800; color: var(--primary); }}
        .kpi-lbl {{ font-size: 12px; font-weight: 600; text-transform: uppercase; color: var(--muted); letter-spacing: 0.5px; margin-top: 4px; }}
        .kpi-sub {{ font-size: 13px; color: var(--muted); margin-top: 6px; }}

        table {{ width: 100%; border-collapse: collapse; background: var(--card); border-radius: 12px; overflow: hidden; margin-bottom: 32px; font-size: 13px; border: 1px solid var(--border); }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid var(--border); }}
        th {{ background: #0f172a; color: var(--muted); text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }}
        tr:hover {{ background: #1a253c; }}

        .bar-wrap {{ background: #1e293b; border-radius: 9999px; height: 8px; width: 100px; display: inline-block; vertical-align: middle; margin-right: 8px; overflow: hidden; }}
        .bar-fill {{ background: var(--success); height: 100%; display: block; border-radius: 9999px; }}
        .bar-fill-warn {{ background: var(--warning); }}
        .bar-fill-dang {{ background: var(--danger); }}

        .tag {{ display: inline-block; padding: 4px 8px; border-radius: 6px; font-size: 11px; font-weight: 700; text-transform: uppercase; }}
        .tag-gun {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }}
        .tag-fake {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }}
        .tag-amb {{ background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }}

        .search-box {{ width: 100%; max-width: 420px; padding: 10px 14px; background: #1e293b; border: 1px solid #334155; border-radius: 8px; color: #fff; font-size: 13px; margin-bottom: 16px; }}
        .search-box:focus {{ outline: none; border-color: var(--primary); }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Shoot_Catcher — Strictly Balanced 1:1:1 Benchmark Report</h1>
        <p class="subtitle">Evaluated on a strictly balanced, un-cherry-picked cohort of 150 independent audio files</p>
    </div>

    <div class="badge-bar">
        <span class="badge badge-gold">⚖️ Balanced 1:1:1 Design (50 Gunshots / 50 Imposters / 50 Ambient)</span>
        <span class="badge">⏱️ 750ms Window (75% Hop Overlap)</span>
        <span class="badge">🛡️ DC Bias Stripping + Energy Gate Enabled</span>
        <span class="badge">🏆 Top Performer: {best_model['Model']}</span>
    </div>

    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-val" style="color: #34d399;">{best_model['Gunshots_Caught']}</div>
            <div class="kpi-lbl">Top Gunshot Recall</div>
            <div class="kpi-sub">{best_model['Recall']*100:.1f}% sensitivity ({best_model['Model']})</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-val" style="color: #60a5fa;">50 / 50</div>
            <div class="kpi-lbl">Cohorts Per Class</div>
            <div class="kpi-sub">Strictly equal 33.3% distribution</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-val" style="color: #a855f7;">{best_model['Accuracy']*100:.1f}%</div>
            <div class="kpi-lbl">Top Overall Accuracy</div>
            <div class="kpi-sub">{best_model['TP'] + best_model['TN']} of 150 correctly predicted</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-val" style="color: #38bdf8;">{best_model['F2']:.4f}</div>
            <div class="kpi-lbl">Top F2 Safety Score</div>
            <div class="kpi-sub">Recall prioritized 4x over precision</div>
        </div>
    </div>

    <h2>📊 Deep Learning Model Benchmark Summary</h2>
    <table>
        <thead>
            <tr>
                <th>Model Architecture</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall (Sensitivity)</th>
                <th>F1-Score</th>
                <th>F2-Score</th>
                <th>Gunshots Caught</th>
                <th>Fakes Rejected</th>
                <th>Ambient Ignored</th>
            </tr>
        </thead>
        <tbody>
"""

    for m in metrics:
        rec_pct = m['Recall'] * 100
        bar_class = "bar-fill" if rec_pct >= 80 else ("bar-fill-warn" if rec_pct >= 50 else "bar-fill-dang")

        html += f"""            <tr>
                <td><strong>{m['Model']}</strong></td>
                <td><strong>{m['Accuracy']*100:.1f}%</strong></td>
                <td>{m['Precision']*100:.1f}%</td>
                <td><span class="bar-wrap"><span class="{bar_class}" style="width:{rec_pct}%;"></span></span>{rec_pct:.1f}%</td>
                <td>{m['F1']:.4f}</td>
                <td><strong style="color: #38bdf8;">{m['F2']:.4f}</strong></td>
                <td><strong>{m['Gunshots_Caught']}</strong></td>
                <td>{m['Fakes_Rejected']}</td>
                <td>{m['Ambient_Ignored']}</td>
            </tr>
"""

    html += f"""        </tbody>
    </table>

    <h2>📋 Individual Audio File Scorecard (150 Files)</h2>
    <input type="text" id="filterInput" class="search-box" placeholder="🔍 Search audio file, category, or classification..." onkeyup="filterRows()">
    <table id="scoreTable">
        <thead>
            <tr>
                <th>Audio File</th>
                <th>True Class</th>
                <th>Robust CRNN</th>
                <th>Enhanced 2D</th>
                <th>Baseline 1D</th>
                <th>Crest (dB)</th>
                <th>Blast Ratio</th>
                <th>Centroid</th>
                <th>Automated Rationale</th>
            </tr>
        </thead>
        <tbody>
"""

    for row in scorecard:
        tag_cls = "tag-gun" if row['True_Class'] == "Real_Gunshot" else ("tag-fake" if row['True_Class'] == "Fake_Gunshot" else "tag-amb")
        html += f"""            <tr>
                <td><code>{row['Filename']}</code></td>
                <td><span class="tag {tag_cls}">{row['True_Class']}</span></td>
                <td><strong>{row['Prob_Robust_CRNN']*100:.1f}%</strong></td>
                <td><strong>{row['Prob_Enhanced_2D']*100:.1f}%</strong></td>
                <td>{row['Prob_Baseline_1D']*100:.1f}%</td>
                <td>{row['Crest_Factor_dB']} dB</td>
                <td>{row['Low_Freq_Blast_Ratio']}</td>
                <td>{row['Spectral_Centroid_Hz']} Hz</td>
                <td style="color: var(--muted); font-size: 12px;">{row['Rationale']}</td>
            </tr>
"""

    html += """        </tbody>
    </table>

    <script>
        function filterRows() {
            var input = document.getElementById("filterInput");
            var filter = input.value.toLowerCase();
            var table = document.getElementById("scoreTable");
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

    dashboard_path.write_text(html, encoding='utf-8')


if __name__ == "__main__":
    main()
