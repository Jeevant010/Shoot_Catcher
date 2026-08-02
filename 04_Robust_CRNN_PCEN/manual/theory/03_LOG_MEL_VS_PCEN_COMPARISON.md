# ⚖️ Part 3: Log-Mel vs. PCEN Mathematical Comparison & Domain Shift

This document provides a comparative mathematical analysis showing why Log-Mel Spectrograms fail under microphone mismatch and how PCEN + Synthetic Mic Distortion achieves real-world robustness.

---

## 1. Comparative Mathematical Analysis

| Mathematical Property | Log-Mel Spectrogram | PCEN (Per-Channel Energy Normalization) |
| :--- | :--- | :--- |
| **Primary Operator** | Logarithm: $\log_{10}(S)$ | Adaptive Channel Division: $\frac{S}{(\epsilon + M)^\alpha}$ |
| **Gain Control** | Static global peak subtraction | Dynamic temporal IIR smoothing per frequency |
| **Noise Floor Behavior** | Amplifies low-energy background noise | Normalizes background noise to $\approx 0$ |
| **Derivative near zero** | $\frac{d}{dx}\log(x) = \frac{1}{x} \to \infty$ | $\frac{d}{dx}x^{0.5} = \frac{0.5}{\sqrt{x}}$ (bounded) |
| **Microphone Gain Bias** | Shifted vertically in magnitude | Self-calibrating across time |
| **Transient Contrast** | Blends with background room echo | Sharp, high-contrast vertical spikes |

---

## 2. Why Log-Mel Fails Under Domain Shift

### 2.1 The Microphone Transfer Function Problem
When audio $s(t)$ passes through a microphone capsule, its spectrum is multiplied by the microphone's frequency response $H_{\text{mic}}(f)$:

$$Y_{\text{mic}}(f, t) = S(f, t) \cdot H_{\text{mic}}(f)$$

Taking the Log-Mel transformation:
$$\log\left(Y_{\text{mic}}(f, t)\right) = \log\left(S(f, t)\right) + \log\left(H_{\text{mic}}(f)\right)$$

Notice that $\log(H_{\text{mic}}(f))$ acts as an **additive static offset** across every frame in the spectrogram. A CNN trained on a clean studio mic ($H_{\text{clean}}$) receives an input shifted by $\log(H_{\text{laptop}})$, causing severe feature distortion and classification failure.

### 2.2 How PCEN Eliminates $H_{\text{mic}}(f)$
In PCEN, both the signal $S(f, t)$ and its running average $M(f, t)$ are scaled by $H_{\text{mic}}(f)$:

$$M_{\text{mic}}(f, t) \approx M(f, t) \cdot H_{\text{mic}}(f)$$

When calculating the PCEN gain normalization ratio:
$$\frac{Y_{\text{mic}}(f, t)}{M_{\text{mic}}(f, t)^\alpha} = \frac{S(f, t) \cdot H_{\text{mic}}(f)}{\left( M(f, t) \cdot H_{\text{mic}}(f) \right)^\alpha} = \frac{S(f, t)}{M(f, t)^\alpha} \cdot H_{\text{mic}}(f)^{1 - \alpha}$$

With $\alpha = 0.98$:
$$H_{\text{mic}}(f)^{1 - 0.98} = H_{\text{mic}}(f)^{0.02} \approx 1.0$$

The microphone's frequency bias $H_{\text{mic}}(f)$ is almost **completely cancelled out mathematically**!

---

## 3. Synthetic Microphone Distortion Equations

To train the CRNN model on mic-distorted data without firing real weapons, clean dataset clips $y[n]$ pass through 4 synthetic physical transformations:

### 3.1 2nd-Order Butterworth Bandpass Filter
Simulates cheap microphone capsule frequency roll-off ($200\text{ Hz} - 8000\text{ Hz}$):

$$H_{\text{bp}}(z) = g \cdot \frac{1 - z^{-2}}{1 - a_1 z^{-1} - a_2 z^{-2}}$$

### 3.2 Non-Linear Peak Saturation (Clipping)
Simulates microphone pre-amp clipping under loud acoustic transients:

$$y_{\text{clip}}[n] = \begin{cases} 
+c & \text{if } y[n] > +c \\
y[n] & \text{if } -c \le y[n] \le +c \\
-c & \text{if } y[n] < -c 
\end{cases}, \quad c \in [0.6, 0.95]$$

### 3.3 Additive Noise Injection
Injects Gaussian noise $e[n] \sim \mathcal{N}(0, \sigma_e^2)$ at target Signal-to-Noise Ratio ($\text{SNR}_{\text{dB}} \in [10, 30]$):

$$\sigma_e^2 = \frac{\frac{1}{N}\sum_{n=0}^{N-1} y[n]^2}{10^{\text{SNR}_{\text{dB}} / 10}}$$

### 3.4 Dynamic Gain Scaling
$$y_{\text{final}}[n] = g_{\text{scale}} \cdot y_{\text{clip}}[n], \quad g_{\text{scale}} \in [0.3, 1.7]$$

---

## 4. Summary Matrix

```text
Clean WAV Clip ───► [Synthetic Mic Distortion] ───► [PCEN Extraction] ───► [CRNN Model]
                          │                            │                       │
                          ▼                            ▼                       ▼
                   Simulates Cheap Mic          Cancels Static          Evaluates Spatial
                   Capsule & Peak Clip          Noise & Mic Bias        Onset & Decay Sequence
```
