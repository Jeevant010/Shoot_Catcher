# Results & Formulas — Consolidated Reference for the Research Paper

This file collects, in one place, every mathematical formulation and every measurement used in
`RESEARCH_PAPER.md`, drawn from the repository reports (`RESEARCH_PAPER_BENCHMARK_REPORT.md`,
`CRNN_PCEN_DOMAIN_ROBUSTNESS_REPORT.md`, `VM_BENCHMARK_RESULTS.md`, `FINAL_RESEARCH_BENCHMARK_REPORT.md`,
`MULTI_MODEL_DISTINCTION_PIPELINE_REPORT.md`, `TEACHER_CONCLUSION_MATRIX_REPORT.md`,
`EDGE_IMPULSE_EMPIRICAL_STUDY_AND_WSN_REPORT.md`) and the WSN project report. Nothing here is invented.

---

## Part A — Mathematical Formulations

### A.1 Signal conditioning and framing

- Shockwave kernel duration (1D CNN first layer, K = 80 at 22,050 Hz):
  $$ \Delta t = \frac{80}{22{,}050} \approx 3.63\ \text{ms} $$
- DC rail offset strip:
  $$ y_{\text{clean}}[n] = y[n] - \frac{1}{N}\sum_{i=0}^{N-1} y[i] $$
- Energy-gated peak normalization:
  $$ y_{\text{norm}} = \begin{cases} \frac{y}{\max(|y|)} & \text{RMS} \ge 0.005 \\ y & \text{RMS} < 0.005 \end{cases} $$
- Decibel-to-unit normalization (fixes the Dying ReLU):
  $$ S_{\text{norm}} = \frac{S_{\text{dB}} + 80}{80} $$
- Frame geometry: N = 16,537 samples = 750 ms at 22,050 Hz; hop = 187.5 ms (75% overlap).

### A.2 Onset detection and quality gates

- Spectral flux (half-wave rectified):
  $$ O(m) = \sum_{k} \max\left(0, |X(k,m)| - |X(k,m-1)|\right) $$
- Raw amplitude peak: $n_{\text{peak}} = \arg\max_n |y[n]|$.
- Smoothed envelope kernel: $K = 0.005 \cdot f_s$.
- Crest factor: $\dfrac{\max(|y|)}{\text{RMS} + \epsilon}$
- Prominence: $\dfrac{\max(|y|)}{\text{Median}(|y|) + \epsilon}$
- Attack ratio: $\dfrac{\sum_{n=0}^{N/2-1} y[n]^2}{\sum_{n=N/2}^{N-1} y[n]^2 + \epsilon}$

### A.3 PCEN front-end

- Slaney Mel filter weight (bandwidth-normalized):
  $$ \text{FilterWeight}_m(f) = \text{Triangle}_m(f) \times \frac{2}{f_{m+2} - f_m} $$
- IIR noise envelope and smoothing coefficient:
  $$ M[f,t] = (1-b)\,M[f,t-1] + b\,S[f,t], \qquad b \approx 0.2068 $$
  $$ t_{\text{frames}} = \frac{T \cdot f_s}{\text{hop}} = \frac{0.025 \times 22050}{128} \approx 4.3066, \quad
     b = \frac{\sqrt{1 + 4\,t_{\text{frames}}^2} - 1}{2\,t_{\text{frames}}^2} $$
- PCEN non-linearity:
  $$ P[f,t] = \left(\frac{S[f,t]}{(\epsilon + M[f,t])^{\alpha}} + \delta\right)^{r} - \delta^{r} $$
  with α = 0.98, δ = 2.0, r = 0.5, ε = 1e-6.
- Steady-state stationary-noise gain: $\lim_{t\to\infty} E(t) \approx S_0^{1-\alpha} = S_0^{0.02} \approx 1.0$.
- Microphone-gain invariance: $g^{2(1-\alpha)} = g^{0.04} \approx 1.0$.
- Z-score standardization: $X = \dfrac{P - \mu_{\text{train}}}{\sigma_{\text{train}}} = \dfrac{P - 0.439023}{0.187652}$.

### A.4 Training regularization and loss

- MixUp: $\tilde{x} = \lambda x_i + (1-\lambda) x_j,\ \tilde{y} = \lambda y_i + (1-\lambda) y_j,\ \lambda \sim \text{Beta}(0.3,0.3)$.
- Class-weighted F2 loss uses class weights C1 ≈ 6.80 (gunshot) and C0 ≈ 0.54, penalising false negatives ≈ 12.5× more than false positives.

### A.5 Ensemble decision rule and acoustic physics

- Adaptive OR vote:
  $$ \text{IsGunshot} = (P_{\text{Enh2D}} \ge 0.45) \lor (P_{\text{CRNN}} \ge 0.50) \lor (P_{\text{Enh2D}} \ge 0.35 \land P_{\text{CRNN}} \ge 0.35) \lor (P_{\text{Enh2D}} \ge 0.35 \land P_{\text{1D}} \ge 0.70) $$
- Low-frequency blast energy ratio: $E_{\text{low}} = \dfrac{\sum_{f<200}|X(f)|^2}{\sum_f |X(f)|^2}$
- High-frequency crack ratio: $E_{\text{high}} = \dfrac{\sum_{f>4000}|X(f)|^2}{\sum_f |X(f)|^2}$
- Spectral centroid: $C_s = \dfrac{\sum_f f\,|X(f)|^2}{\sum_f |X(f)|^2}$
- Physics rules: clap → imposter if $E_{\text{low}} < 2\%$ and $C_s > 3000$ Hz; door slam → imposter if $E_{\text{low}} > 88\%$ and $E_{\text{high}} < 1\%$; impulsive ambient if crest factor $> 13.5$ dB.

### A.6 WSN networking

- 1-bit alert payload: $\{\text{Node\_ID (1 B)},\ \text{Timestamp (4 B)},\ \text{Detection (1 bit)},\ \text{Confidence (1 B)}\}$.
- Coincidence window: $\Delta t \le d_{\max}/c_{\text{sound}} \approx 34/343 \approx 100$ ms.
- TDOA hyperbolic equation: $\sqrt{(x-x_i)^2+(y-y_i)^2} - \sqrt{(x-x_j)^2+(y-y_j)^2} = c\,(t_i-t_j)$.

---

## Part B — Results Tables

### B.1 Edge Impulse experiments (Cortex-M4F, INT8)

| Experiment | Representation | Classifier | Val. Acc. | INT8 Latency | Peak RAM | Flash | Live behaviour |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---|
| 1 | 13 MFCCs | 1D CNN (8,16) | 99.4% | 1 ms | 11.9 KB | 45.4 KB | Voice-trap (33/47 FP) |
| 2 | 13 MFCCs + K-means | 1D CNN + Anomaly | 99.4% | 2 ms (DSP 120 ms) | 11.9 KB | 45.4 KB | Widespread OOD alarms |
| 3 | MFE Spectrogram (3,383 feats) | 1D CNN | 98.8% | 12 ms | 18.1 KB | 47.1 KB | Moderate (24/48 FP) |
| 4 | MFE Spectrogram (FFT 32) | 2D CNN | 99.6% | 167 ms | 30.2 KB | 52.2 KB | Wind 75.8% / Thunder 80.0% reject |
| 5 | MFE Spectrogram + clusters | 2D CNN + Anomaly | 99.7% | 167 ms | 30.2 KB | 52.2 KB | Robust OOD gate |

### B.2 Closed-split baselines (WSN project report)

| Model | Accuracy | Precision | Recall | F1 |
|:---|:---:|:---:|:---:|:---:|
| SVM (RBF, MFCC) | 0.953 | 0.99 | 0.91 | 0.95 |
| Random Forest (MFCC) | 0.99 | 0.99 | 0.99 | 0.99 |
| 1D CNN (MFCC) | 0.96 | 0.96 | 0.95 | 0.96 |
| 2D CNN (MFE spectrogram) | 0.98 | 0.98 | 0.97 | 0.98 |

### B.3 Architecture specifications and INT8 footprint

| # | Architecture | Input | Tensor | Params | INT8 Size |
|:-:|:---|:---|:---:|:---:|:---:|
| 1 | Baseline 1D CNN | Raw waveform | (1, 16537, 1) | 92,513 | 112.5 KB |
| 2 | Baseline 2D CNN | Log-Mel | (1, 64, 130, 1) | 249,985 | 259.9 KB |
| 3 | Enhanced 1D CNN | Raw (dilated, dual-head) | (1, 16537, 1) | 96,674 | 118.5 KB |
| 4 | Enhanced 2D CNN | Log-Mel | (1, 64, 33, 1) | 110,594 | 123.6 KB |
| 5 | Robust CRNN-PCEN | PCEN + unrolled GRU | (1, 64, 130, 1) | 397,842 | 478.9 KB |

### B.4 300-clip in-the-wild benchmark, τ = 0.50

**Sliding window**

| Model | Recall | Ambient Spec. | Imposter Rej. | Latency | Hop Budget |
|:---|:---:|:---:|:---:|:---:|:---:|
| Baseline 1D CNN | 45.0% | 85.0% | 72.0% | 2.79 ms | 1.5% |
| Baseline 2D CNN | 66.0% | 88.0% | 59.0% | 5.36 ms | 2.9% |
| Enhanced 1D CNN | 100.0% (degenerate) | 0.0% | 0.0% | 3.36 ms | 1.8% |
| Enhanced 2D CNN | 67.0% | 82.0% | 66.0% | 0.75 ms | 0.4% |
| Robust CRNN-PCEN | 57.0% | 86.0% | 61.0% | 6.26 ms | 3.3% |

**Peak-centered**

| Model | Recall | Ambient Spec. | Imposter Rej. | Latency | Hop Budget |
|:---|:---:|:---:|:---:|:---:|:---:|
| Baseline 1D CNN | 30.0% | 95.0% | 84.0% | 2.84 ms | 1.5% |
| Baseline 2D CNN | 45.0% | 94.0% | 77.0% | 4.31 ms | 2.3% |
| Enhanced 1D CNN | 100.0% (degenerate) | 0.0% | 0.0% | 4.90 ms | 2.6% |
| Enhanced 2D CNN | 38.0% | 99.0% | 89.0% | 0.76 ms | 0.4% |
| Robust CRNN-PCEN | 33.0% | 97.0% | 85.0% | 6.41 ms | 3.4% |

### B.5 CRNN-PCEN domain robustness (clean vs. synthetic microphone distortion)

| τ | Clean Recall | Distorted Recall | Δ | Clean Ambient | Distorted Ambient | Clean Imposter | Distorted Imposter |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.10 | 69.0% | 69.0% | 0.0% | 75.0% | 74.0% | 48.0% | 52.0% |
| 0.20 | 64.0% | 67.0% | +3.0% | 78.0% | 76.0% | 51.0% | 55.0% |
| 0.30 | 62.0% | 66.0% | +4.0% | 78.0% | 77.0% | 53.0% | 59.0% |
| 0.40 | 61.0% | 63.0% | +2.0% | 79.0% | 77.0% | 55.0% | 59.0% |
| 0.50 | 61.0% | 61.0% | 0.0% | 80.0% | 77.0% | 56.0% | 61.0% |
| 0.60 | 60.0% | 61.0% | +1.0% | 81.0% | 79.0% | 57.0% | 62.0% |
| 0.70 | 59.0% | 59.0% | 0.0% | 84.0% | 81.0% | 59.0% | 66.0% |

### B.6 Multi-model ensemble — 3×3 confusion matrix (300 clips)

| Ground truth | Real Gunshot | Fake Gunshot | Not Gunshot | Total | Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|
| Real gunshot | **77** | 15 | 8 | 100 | 77.0% |
| Hard imposter | 44 | **46** | 10 | 100 | 46.0% |
| Ambient non-gunshot | 20 | 38 | **42** | 100 | 42.0% |

Per-model contribution inside the ensemble:

| Model | Accuracy | Precision | Recall | F1 |
|:---|:---:|:---:|:---:|:---:|
| Baseline 1D CNN | 67.67% | 51.69% | 46.0% | 48.68% |
| Baseline 2D CNN (Mel) | 66.67% | 0.00% | 0.0% | 0.00% |
| Robust CRNN (PCEN) | 67.33% | 50.88% | 58.0% | 54.21% |
| Enhanced 1D CNN (Dual) | 33.33% | 33.33% | 100.0% | 50.00% |
| Enhanced 2D CNN (Dual) | 70.00% | 53.79% | 71.0% | 61.21% |

### B.7 Latency profiling (100 iterations per model, INT8)

| Model | Mean ± Std | P95 | Hop Budget (187.5 ms) |
|:---|:---:|:---:|:---:|
| Baseline 1D CNN | 2.29 ± 8.25 ms | 1.98 ms | 1.2% |
| Baseline 2D CNN | 2.15 ± 6.76 ms | 1.69 ms | 1.2% |
| Enhanced 1D CNN | 2.69 ± 8.23 ms | 6.48 ms | 1.4% |
| Enhanced 2D CNN | 0.09 ± 0.01 ms | 0.10 ms | 0.05% |
| Robust CRNN-PCEN | 11.86 ± 4.12 ms | 14.50 ms | 6.3% |

### B.8 Cross-platform consistency (Host CPU vs. RPi4 edge VM)

| Model | Host Recall (peak) | VM Recall (peak) | Variance |
|:---|:---:|:---:|:---:|
| Baseline 1D CNN | 35.6% | 30.0% | ±5.6% |
| Baseline 2D CNN | 49.2% | 45.0% | ±4.2% |
| Enhanced 2D CNN | 44.1% | 38.0% | ±6.1% |
| Robust CRNN-PCEN | 32.0% | 33.0% | ±1.0% |

> Variance is explained by INT8 rounding differences between x86 XNNPACK and ARM NEON kernels.

---

## Part C — Source map (which file each number came from)

| Content | Source file |
|:---|:---|
| Edge Impulse 5 experiments | `WSN Report.pdf`, `EDGE_IMPULSE_EMPIRICAL_STUDY_AND_WSN_REPORT.md` |
| Closed-split SVM/RF/CNN results | `WSN Report.pdf` |
| 300-clip peak/sliding tables, latency, INT8 sizes | `VM_BENCHMARK_RESULTS.md`, `FINAL_RESEARCH_BENCHMARK_REPORT.md`, `TEACHER_CONCLUSION_MATRIX_REPORT.md` |
| PCEN domain-robustness sweep | `CRNN_PCEN_DOMAIN_ROBUSTNESS_REPORT.md` |
| Ensemble 3×3 + physics rules | `MULTI_MODEL_DISTINCTION_PIPELINE_REPORT.md`, `FINAL_RESEARCH_BENCHMARK_REPORT.md` |
| PCEN equations, Slaney filter, z-score | `CRNN_PCEN_DOMAIN_ROBUSTNESS_REPORT.md` |
| Onset gates, DC strip, energy gate, MixUp, F2 loss | `baseline_research_paper.tex`, `ieee_conference_paper.tex` |
| WSN node/sleep/1-bit/TDOA | `EDGE_IMPULSE_EMPIRICAL_STUDY_AND_WSN_REPORT.md`, `WSN Report.pdf` |
| Handwritten wording and structure | `research_paper/HANDWRITTEN_NOTES_VERBATIM.md` |
