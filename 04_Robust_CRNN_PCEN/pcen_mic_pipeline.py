"""
===============================================================================
🛡️ Shoot_Catcher — Module 04: PCEN & Synthetic Microphone Pipeline
===============================================================================
Per-Channel Energy Normalization (PCEN) & Synthetic Microphone Distortion
designed to eliminate Acoustic Domain Shift (microphone mismatch).

Based on Forest Acoustic & Chainsaw Detection Research (NYU Bioacoustics / DCASE).
===============================================================================
"""

import numpy as np
import scipy.signal as signal

try:
    import librosa
    HAS_LIBROSA = True
except Exception:
    HAS_LIBROSA = False


def compute_pcen_scipy(y, sr=22050, n_mels=64, n_fft=512, hop_length=128,
                       s=0.025, alpha=0.98, delta=2.0, r=0.5, eps=1e-6):
    """
    Pure NumPy / Scipy implementation of Per-Channel Energy Normalization (PCEN).
    
    Formula:
        P[f, t] = ( S[f, t] / (eps + M[f, t])^alpha + delta )^r - delta^r
    where M[f, t] is a smoothed time-frequency envelope computed via an IIR filter:
        M[f, t] = (1 - s) * M[f, t-1] + s * S[f, t]
    """
    # 1. Compute STFT & Power Spectrogram
    f, t, Zxx = signal.stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length, boundary=None)
    power = np.abs(Zxx) ** 2  # shape: (num_bins, time_steps)

    # 2. Build Mel Filterbank
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

    # S: Mel Power Spectrogram shape: (n_mels, time_steps)
    S = np.dot(fb, power).astype(np.float32)

    # 3. Temporal Smoothing (IIR Low-pass Filter along time axis)
    # M[f, t] = (1 - s) * M[f, t-1] + s * S[f, t]
    M = signal.lfilter([s], [1.0, -(1.0 - s)], S, axis=-1)

    # 4. Adaptive Gain Control & Dynamic Range Compression
    # P = (S / (eps + M)^alpha + delta)^r - delta^r
    smooth = (eps + M) ** alpha
    normalized = S / smooth
    pcen = (normalized + delta) ** r - (delta ** r)

    return pcen.astype(np.float32)


def compute_pcen(y, sr=22050, n_mels=64, n_fft=512, hop_length=128):
    """
    Computes PCEN feature matrix (n_mels x time_steps).
    Tries Librosa PCEN first; falls back to pure Scipy if needed.
    """
    if HAS_LIBROSA:
        try:
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, power=1)
            pcen = librosa.pcen(S * (2**31), sr=sr, hop_length=hop_length, time_constant=0.025, gain=0.98, bias=2.0, power=0.5)
            return pcen.astype(np.float32)
        except Exception:
            pass

    return compute_pcen_scipy(y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)


def simulate_microphone_effects(y, sr=22050):
    """
    Simulates real-world microphone & acoustic hardware artifacts:
    1. Bandpass Filtering (200Hz - 8000Hz simulating cheap mic capsule)
    2. Dynamic Range Clipping (simulates mic overload)
    3. Random Gain Scaling (simulates mic distance / sensitivity)
    4. Additive Noise Injection (simulates HVAC / mic self-noise)
    """
    augmented = y.copy()

    # 1. Bandpass Filter (Cut frequencies < 200Hz and > 8000Hz)
    try:
        nyquist = sr / 2.0
        low = max(50.0, np.random.uniform(150.0, 300.0)) / nyquist
        high = min(nyquist - 100.0, np.random.uniform(6000.0, 9000.0)) / nyquist
        b, a = signal.butter(2, [low, high], btype='band')
        augmented = signal.filtfilt(b, a, augmented)
    except Exception:
        pass

    # 2. Dynamic Range Clipping (non-linear saturation)
    clip_thresh = np.random.uniform(0.6, 0.95)
    augmented = np.clip(augmented, -clip_thresh, clip_thresh)

    # 3. Additive Gaussian & Pink Noise Injection
    snr_db = np.random.uniform(10.0, 30.0)
    signal_power = np.mean(augmented ** 2)
    if signal_power > 1e-8:
        noise_power = signal_power / (10 ** (snr_db / 10.0))
        noise = np.random.normal(0, np.sqrt(noise_power), len(augmented))
        augmented += noise

    # 4. Gain Scaling
    gain = np.random.uniform(0.3, 1.7)
    augmented = augmented * gain

    # Peak Normalize
    peak = np.max(np.abs(augmented))
    if peak > 1e-6:
        augmented = augmented / peak

    return augmented.astype(np.float32)
