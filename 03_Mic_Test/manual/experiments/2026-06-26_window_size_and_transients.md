# Gunshot Detection Experiment Report
**Date**: 2026-06-26
**Topic**: Window Size, 1D CNN Transient Loss, and Arduino Feasibility

## 1. The 250ms Window Size Problem

In our testing, we noticed that a 250ms sliding window struggles to accurately detect gunshots in real-world scenarios, especially when reverberation or distance is involved. You suspected this was too short, and academic research proves you are right.

### Academic Standard for Classification Windows
In gunshot detection research, "window size" refers to two things:
1. **Framing Window (10ms - 30ms):** Used purely for feature extraction (like grabbing a single slice of a spectrogram or MFCC).
2. **Classification Window (0.5s - 2.0s):** The length of the entire audio clip fed into the Neural Network.

Research papers consistently show that **0.5 seconds to 2.0 seconds** is the standard classification window length required to capture the full acoustic signature of a gunshot (the muzzle blast followed by the shockwave and environmental reverberation). 

*Our current window is 250ms (0.25s), which is exactly half of the minimum recommended size for robust classification.*

**Supporting Research & Repositories:**
- **[gabemagee/gunshot_detection (GitHub)](https://github.com/gabemagee/gunshot_detection):** A highly cited project deployed on Raspberry Pi that explicitly uses **2.0 second** audio buffers for its CNN ensembles.
- **[hasnainnaeem/Gunshot-Detection-in-Audio (GitHub)](https://github.com/hasnainnaeem/Gunshot-Detection-in-Audio):** Uses models trained on the UrbanSound8K dataset, which by default processes clips up to **4.0 seconds** long.
- **Academic Studies:** Papers published in MDPI and IEEE frequently cite the trade-off between latency and accuracy, settling on sliding windows of ~1.0s to 2.0s.

## 2. The Transient Loss Problem (Why YouTube testing fails)

When we played a gunshot from YouTube out of laptop speakers and recorded it with a microphone, the 1D CNN completely missed the loud gunshot (giving it 0.0000 confidence).

**Why?** A 1D CNN looks directly at the raw physical shape of the pressure wave. A real gunshot contains a "transient"—a near-instantaneous, violently sharp spike in pressure lasting less than 3 milliseconds.
1. **Speaker Physics:** Small laptop/phone speakers physically cannot vibrate fast enough or hard enough to recreate that sharp transient. They "smooth" it out into a dull thump.
2. **Microphone Smoothing:** The receiving microphone also struggles to capture extreme impulses without clipping.
3. **Room Acoustics:** The sound bounces off walls, further stretching the impulse.

To the 1D CNN, the speaker-played gunshot looks completely different from the real `.wav` files it was trained on. 

**Industry Solution:**
This is exactly why most of the GitHub repositories mentioned above (and industry systems like ShotSpotter) use **2D Spectrograms (Mel-Spectrograms)** instead of raw 1D waveforms. Spectrograms look at the frequency content ("pitch" and "tone") over time, which is much more resilient to speaker/mic distortion than the raw physical shape of the wave.

## 3. Arduino Feasibility & Inference Speed

Our current 1D CNN has roughly **92,000 parameters**. During testing on a powerful PC CPU, inference takes a noticeable amount of time (causing the live runner to drop frames if not carefully multi-threaded).

An Arduino (even an advanced one like the Nano 33 BLE Sense) has extremely limited RAM (e.g., 256KB) and processing power. A 92k parameter CNN is generally **too large and too slow** for standard Arduino deployment.

**Options for Edge Deployment:**
1. **Shrink the Model:** We need to drastically reduce the number of filters/layers to get under 20k parameters.
2. **Switch to Raspberry Pi:** Projects like `gabemagee/gunshot_detection` use Raspberry Pis specifically because they can run larger TensorFlow models in real-time.
3. **Hardware Accelerators:** Use specialized boards like the ESP32 or Coral Edge TPU.

## Conclusion

Our goal is near-zero False Negatives (catch all gunshots) while accepting some False Positives. To achieve this in the real world:
1. We must increase the training window size to at least 0.5s or 1.0s.
2. We must recognize that 1D CNN testing via laptop speakers will always yield artificially poor results due to transient loss.
3. If Arduino is the strict final target, we need an aggressive model pruning strategy.
