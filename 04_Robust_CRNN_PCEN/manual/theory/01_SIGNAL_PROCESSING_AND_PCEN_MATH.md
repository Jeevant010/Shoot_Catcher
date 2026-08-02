# 📐 Part 1: Discrete Audio Signal Processing & PCEN Mathematics

This document details the complete mathematical derivation of digital audio sampling, frequency analysis, and Per-Channel Energy Normalization (PCEN).

---

## 1. Discrete Audio Signal & Sampling Math

Sound originates as a continuous time-varying air pressure wave $p(t)$. An Analog-to-Digital Converter (ADC) discretizes this wave into a 1D sequence of digital numbers $y[n]$.

### 1.1 Sampling Equation
$$y[n] = p(n \cdot T_s) = p\left(\frac{n}{f_s}\right)$$

Where:
- $f_s = 22,050 \text{ Hz}$ (Sampling rate = 22,050 samples per second).
- $T_s = \frac{1}{f_s} \approx 45.35 \mu\text{s}$ (Time duration between samples).
- For a $750\text{ms}$ audio clip:
  $$N = 0.75 \times 22,050 = 16,537 \text{ discrete scalar samples}$$

### 1.2 Vector Representation
$$\mathbf{y} = \begin{bmatrix} y_0 & y_1 & y_2 & \dots & y_{16536} \end{bmatrix}^T \in \mathbb{R}^{16537}$$

---

## 2. Short-Time Fourier Transform (STFT) Math

To analyze frequency content across time, $\mathbf{y}$ is divided into overlapping frames using a FFT window size $N_{\text{fft}} = 512$ and hop size $H = 128$.

### 2.1 Hann Windowing Function
To prevent boundary discontinuities during Fourier transformation:
$$w[n] = 0.5 \cdot \left(1 - \cos\left(\frac{2\pi n}{N_{\text{fft}} - 1}\right)\right), \quad 0 \le n < N_{\text{fft}}$$

### 2.2 Complex Discrete Fourier Transform (DFT)
For frame index $m$, the complex spectrum $Z[k, m]$ is computed via Euler's formula ($e^{-j\theta} = \cos\theta - j\sin\theta$):

$$Z[k, m] = \sum_{n=0}^{N_{\text{fft}}-1} y[m \cdot H + n] \cdot w[n] \cdot e^{-j \frac{2\pi k n}{N_{\text{fft}}}}$$

Where:
- $k \in [0, 256]$ is the discrete frequency bin index ($0\text{ Hz}$ to Nyquist $11,025\text{ Hz}$).
- $m \in [0, 129]$ is the time frame index ($\approx 130$ time steps).
- $Z[k, m] \in \mathbb{C}$ is a complex number: $Z[k, m] = a + jb$.

### 2.3 Power Spectrogram
$$\mathbf{P}_{\text{raw}}[k, m] = |Z[k, m]|^2 = a^2 + b^2 \in \mathbb{R}^{257 \times 130}$$

---

## 3. Mel Filterbank Linear Algebra

Human auditory perception is logarithmic. We map 257 linear Hertz frequency bins into 64 Mel-scale frequency bins using matrix multiplication.

### 3.1 Hertz to Mel Scale Conversion
$$m = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)$$

### 3.2 Matrix Multiplication
$$\mathbf{S} = \mathbf{W}_{\text{Mel}} \times \mathbf{P}_{\text{raw}}$$

Where:
- $\mathbf{W}_{\text{Mel}} \in \mathbb{R}^{64 \times 257}$ is a triangular Mel weighting filterbank matrix.
- $\mathbf{P}_{\text{raw}} \in \mathbb{R}^{257 \times 130}$ is the raw STFT power spectrogram.
- $\mathbf{S} \in \mathbb{R}^{64 \times 130}$ is the output Mel Power Spectrogram.

---

## 4. Per-Channel Energy Normalization (PCEN) Calculus

PCEN replaces traditional Log-Mel spectrograms to eliminate background noise and microphone frequency bias.

### 4.1 Low-Pass Temporal Background Noise Envelope $M[f, t]$
For each Mel frequency row $f$, a 1st-order Infinite Impulse Response (IIR) filter calculates a running average of background noise over time $t$:

$$M[f, t] = (1 - s) \cdot M[f, t-1] + s \cdot S[f, t]$$

Where:
- $s = 0.025$ (Smoothing factor, corresponding to a $\approx 25\text{ms}$ integration time constant).
- Differential Equation Equivalent:
  $$\frac{d M(t)}{dt} = -\frac{1}{\tau} M(t) + \frac{1}{\tau} S(t)$$

### 4.2 Adaptive Channel Gain Normalization
$$E[f, t] = \frac{S[f, t]}{\left( \epsilon + M[f, t] \right)^{\alpha}}$$

Where:
- $\epsilon = 10^{-6}$ (Prevents division by zero).
- $\alpha = 0.98$ (Controls the strength of gain suppression).
- **Behavior**: If $S[f, t] \approx M[f, t]$ (stationary noise), $E[f, t] \approx \text{constant}$. Static noise is suppressed!

### 4.3 Root Dynamic Range Compression
$$P[f, t] = \left( E[f, t] + \delta \right)^r - \delta^r$$

Where:
- $\delta = 2.0$ (Bias offset).
- $r = 0.5$ (Square-root power compression).

Unlike $\log(x)$ (whose derivative $\frac{d}{dx}\log(x) = \frac{1}{x}$ explodes near zero), the derivative of root compression $\frac{dP}{dx} = r(x + \delta)^{r-1}$ remains bounded and smooth near zero, preventing low-level noise amplification.
