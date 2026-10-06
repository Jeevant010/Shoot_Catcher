# On-Device Acoustic Gunshot Detection: Data Cleaning, Model Comparison, and a One-Bit Edge Alert

**Jeevant Sharma, Deepesh Dangi, Kaustubh Dhondge**  
Indian Institute of Information Technology Surat, India  
{ui23cs30, ui23cs18, kaustubh.dhondge}@iiitsurat.ac.in

---

## Abstract

Acoustic gunshot detection on low-cost embedded edge platforms presents physical challenges including transducer domain shift, diaphragm saturation at high sound pressure levels, electrical DC voltage rail drift, thermal ADC hiss amplification, and false triggers from percussive imposter sounds. This paper reports a detector designed for deployment on an Arduino Nano 33 BLE Sense. Models are trained on a host PC, exported from Keras to TensorFlow Lite [15], and quantized to INT8. Edge timing is profiled on a Raspberry Pi 4 with the CPU held at 45–50% utilization. Audio was cleaned with librosa at window lengths of 250 ms, 500 ms, and 1 s. The operational window is 750 ms, since a 250 ms cut removes the reverberation decay that distinguishes a gunshot from a percussive knock. Classical classifiers (logistic regression, SVM, random forest) achieved 99–100% accuracy on balanced hold-out files but failed on live microphone input. Five neural architectures were then trained. On a 1,442-clip balanced hold-out, three models exceed 99.5% accuracy. On 300 unseen recordings, the best single-model sliding-window gunshot recall is 67%, achieved by the Enhanced 2D CNN, which executes in 0.09 ms within a 123.6 KB INT8 footprint. A three-model ensemble raises recall to 77% within 715 KB and 17 ms combined. The PCEN-based CRNN exhibits zero recall degradation under simulated microphone distortion. The deployment architecture retains all audio on the sensor node, transmits only a single-bit detection flag (1 for gunshot, silence otherwise), and requires spatial consensus from multiple neighbouring nodes at the gateway before raising an alert.

**Keywords:** Acoustic gunshot detection, edge inference, TinyML, PCEN, convolutional recurrent neural networks, Arduino Nano 33 BLE Sense, one-bit transmission.

---

## I. Introduction

Firearm violence in urban and peri-urban environments demands rapid detection of gunshot events. A significant fraction of discharges are never reported. When a report is made, it is often late and poorly located. An acoustic sensor on a campus, a city block, or a wildlife sanctuary can mark the event within a second and send the response to that place. The same listening problem appears in forests, where a gunshot is rare against wind and insects [3].

Dense streets are the hardest first site: traffic, construction, and fireworks produce impulses all day. A college or a school has a known map and less uncontrolled impulse noise, and that is the site this system is built for.

A gunshot on a single microphone has two parts [1]. The projectile shockwave, when the bullet is supersonic, is a short pressure rise of about 1–5 ms. The muzzle blast then decays in the room or in the open air for about 200–600 ms. A detector that keeps only the rise cannot tell the blast from a knock.

Models are trained on a workstation and exported from Keras (.h5) to TensorFlow Lite (.tflite) with post-training INT8 quantization [15]. Latency is measured on a Raspberry Pi 4 (quad-core Cortex-A72, 2 GB RAM) with the CPU held at 45–50% so the timing reflects the sustained edge load, not a short burst. The target deployment node is the Arduino Nano 33 BLE Sense (nRF52840, Cortex-M4F, 1 MB Flash, 256 KB SRAM). All inference runs on the node itself; no microphone audio is streamed off-device. The duty-cycled sleep schedule, one-bit radio packet, and multi-node spatial consensus at the gateway are described in Section VIII.

Section II reviews related work. Section III describes cleaning and the analysis window. Section IV describes the signal conditioning used before inference. Section V describes the models. Section VI defines the tests. Section VII reports the measurements.

---

## II. Related Work

Open firearm recordings used in this work include the multi-caliber, multi-orientation set in *Data in Brief* [2]. Ambient and imposter audio draws on urban environmental recordings [5]. City-scale gunshot systems such as ShotSpotter [4] have shown that acoustic detection is operationally useful. They stream audio to a central server. A campus node cannot do that: the radio budget and the privacy constraint both require the decision to be made on the device.

Per-channel energy normalization (PCEN) divides each frequency band by a smoothed estimate of its own background [6]. Steady rain, wind, and hum collapse toward a constant. A short impulse does not. Lostanlen et al. extended the formulation to bioacoustic monitoring in adverse outdoor conditions [7]. The CRNN in this paper uses that front-end. The network, the training set, and the tests are ours.

Spectrogram CNNs are a standard audio classifier [8]. Saha et al. applied spectrogram CNNs to gunshot detection in forest environments [3]. On microcontrollers, the relevant limits are flash, RAM, and latency under integer arithmetic [14]. Dai et al. showed that wide 1D convolutional kernels on raw waveforms can capture short impulsive transients [9]. The gated recurrent unit (GRU) used in the CRNN follows the formulation of Cho et al. [10]. The enhanced models use MixUp [11], SpecAugment [12], and speed perturbation [13] as regularization. We do not re-derive those methods.

---

## III. Data Pipeline and Preprocessing

The raw pool was about 80,000 files: the calibrated firearm set [2], 4,138 further firearm recordings from open ground and indoor rooms, and 6,230 long ambient recordings (traffic, rain, construction, speech) [5], from about 10 s to 60 min. Cleaning used librosa. Between 1,000 and 5,000 clips were listened through, in passes, until the cuts that remained were acceptable.

Cuts were tried at 250 ms, 500 ms, and 1 s. The attack is only a few milliseconds. The decay is the part that separates a blast from a knock, and a 250 ms cut drops most of it. The models use **750 ms** (16,537 samples at 22,050 Hz). On the live path the hop is 187.5 ms (75% overlap), so a blast on a buffer boundary still sits inside one frame.

Each candidate cut is placed by one of three onset cues: spectral flux across STFT bins, the peak absolute sample, or a short moving-average envelope that ignores single-sample spikes. A negative clip is discarded when its crest factor and its attack energy both look like a shot. The negative class is ordinary sound, not a second set of bangs with the wrong label.

A 1:1 gunshot to non-gunshot folder is what produced the 99–100% classical scores below. A live stream is closer to 1:10 or 1:50, because almost every window is silence, speech, or traffic. Later training used the heavier negative ratios.

---

## IV. Hardware Conditioning for Edge Deployment

The same 750 ms frame is conditioned before the network. Three distortions show up as soon as the file is no longer a studio WAV.

Low-cost capsules on a 3.3 V or 5 V rail sit a few tens of millivolts off zero. That offset is removed by the sample mean:

$$y_{\text{clean}}[n] = y[n] - \frac{1}{N}\sum_{i=0}^{N-1} y[i]$$

Peak scaling y/max(|y|) is wrong on a quiet window. A peak near 0.002 stretches ADC hiss across the full range. Scaling is applied only when the window is loud enough:

$$y_{\text{cond}} = \begin{cases} y / \max(|y|) & \text{if RMS} \ge 0.005 \\ y & \text{if RMS} < 0.005 \end{cases}$$

A close shot exceeds 120 dB SPL and flattens the capsule into a clipped peak. Hard clipping is included in training so the filters still see a rise-time on a saturated waveform.

---

## V. Model Architectures

Logistic regression, an SVM, and a random forest on MFCC summaries reached about 99–100% on balanced files from this pool and then failed on a live microphone. Averaging a clip into a few coefficients removes the few-millisecond attack. Those classifiers were stopped. The five networks below share the conditioning in Section IV.

**Baseline 1D CNN** reads the 16,537 samples directly. Its first kernel is 80 samples wide, about 3.6 ms at 22,050 Hz [9], so the first layer can cover a short attack. It does not transfer across microphones: hold-out recall is 99.7%, and recall on external recordings is 30%.

**Baseline 2D CNN** reads a 64-band log-mel spectrogram (130 frames, FFT size 512, hop 128) [8]. An early run fed decibel values from about −80 dB to 0 dB into ReLU. ReLU is zero on that whole range, gradients stopped, and that checkpoint marked all 721 hold-out gunshots as negatives. Later 2D models use S_norm = (S_dB + 80) / 80, which maps the image into [0,1]. Pooling is kept off the last convolutional blocks so the attack is not averaged away.

**Enhanced 2D CNN** is the same idea at a smaller width, with MixUp [11], SpecAugment [12], speed perturbation [13], and a second head trained as an anomaly score. The gunshot head is the score reported below. The **Enhanced 1D CNN** adds the same second head on the waveform. INT8 conversion saturates it: every window, including silence, scores as a gunshot. It is reported as a failed export, not as a detector. The 2D models, on [0,1] spectrograms, quantize without that collapse.

**CRNN-PCEN** replaces the static log-mel with PCEN [6, 7]:

$$P[f,t] = \left( \frac{S[f,t]}{(\epsilon + M[f,t])^{\alpha}} + \delta \right)^{r} - \delta^{r}$$

where M[f,t] is a one-pole smooth of S[f,t]. With α near 0.98, a band that does not change settles near 1. A gain change g in the capsule scales the output by about g^0.04. A bidirectional GRU [10] (64 units each way) then reads the frame sequence, so a sustained tone is not treated as a blast. PCEN needs a stable background in the clip. That is why the trim in Section III comes first.

Five Edge Impulse [16] configurations on a Cortex-M4F target were used only as a screen, before these models. Validation accuracy sat between 98.8% and 99.7%. Live audio did not follow it. An MFCC 1D CNN marked 33 of 47 ambient windows as shots. A 2D CNN on a mel spectrogram was the first of those runs to reject most wind and thunder windows. Those Edge Impulse runs are not part of the measured system.

---

## VI. Experimental Setup and Metrics

All reported neural scores use the exported INT8 models [15]. Latency is 100 timed iterations on the Raspberry Pi 4 with the CPU held at 45–50% and 2 GB RAM [14]. The TFLite runtime is the same class of integer path used on that board.

Three sets are used, in increasing difficulty:
- **Balanced hold-out**: 1,442 clips of 750 ms (721 gunshot, 721 other) from the cleaned pool.
- **External set**: 75 recordings from outside that pool: 20 firearms, 20 imposters, 35 ambient tracks.
- **Main set**: 300 unseen recordings: 100 real gunshots, 100 ambient files, and 100 hard imposters (fireworks, slams, thunder, drums).

Gunshot recall is the fraction of real gunshot files that cross the threshold. Ambient specificity is the fraction of rain, traffic, and speech files that stay below it. Imposter rejection is the fraction of fireworks, claps, and other bangs that stay below it. The hop is 187.5 ms; a model slower than the hop cannot keep up. Threshold is 0.50 unless stated otherwise.

Peak-centered scoring uses one 750 ms slice around the loudest sample. Sliding scoring walks the window across the file. Sliding finds more shots and more false alarms.

---

## VII. Results

### A. Balanced Hold-Out (1,442 clips, τ = 0.50)

| Model | Acc. | Recall | F1 | FP |
|:---|:---:|:---:|:---:|:---:|
| Baseline 1D CNN | 99.79% | 99.72% | 99.79% | 1 |
| CRNN-PCEN | 99.79% | 99.72% | 99.79% | 1 |
| Enhanced 2D CNN | 99.51% | 99.58% | 99.51% | 4 |
| 2D CNN, unscaled dB | 50.00% | 0.00% | --- | 0 |
| Enhanced 1D CNN | 50.00% | 100% | 66.67% | 721 |

The unscaled 2D row is the stalled checkpoint (721 false negatives). The enhanced 1D row is positive on every file.

### B. External Recordings and the 300-Clip Set

**75 external recordings (sliding, τ = 0.50):**

| Model | Recall | Ambient | Imposter |
|:---|:---:|:---:|:---:|
| Enhanced 2D CNN | 17/20 (85%) | 33/35 (94%) | 7/20 (35%) |
| CRNN-PCEN | 12/20 (60%) | 35/35 (100%) | 8/20 (40%) |
| Baseline 1D CNN | 6/20 (30%) | 31/35 (89%) | 11/20 (55%) |

**300 unseen clips at τ = 0.50 (percent):**

| Model | Slide Recall | Slide Ambient | Slide Imposter | Peak Recall | Peak Ambient | Peak Imposter |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline 1D CNN | 45 | 85 | 72 | 30 | 95 | 84 |
| Baseline 2D CNN | 66 | 88 | 59 | 45 | 94 | 77 |
| Enhanced 1D CNN | 100 | 0 | 0 | 100 | 0 | 0 |
| Enhanced 2D CNN | 67 | 82 | 66 | 38 | 99 | 89 |
| CRNN-PCEN | 57 | 86 | 61 | 33 | 97 | 85 |

Host and Pi peak recall differ by about 4 to 6 points (Enhanced 2D CNN: 44.1% on the host, 38.0% on the Pi; CRNN-PCEN: 32.0% and 33.0%). The order of the models does not change. The CRNN moves the least (±1 point).

### C. Latency (INT8, Raspberry Pi 4, 100 iterations, CPU at 45–50%)

| Model | INT8 | Mean | Hop % |
|:---|:---:|:---:|:---:|
| Baseline 1D CNN | 112.5 KB | 2.29 ms | 1.2% |
| Baseline 2D CNN | 259.9 KB | 2.15 ms | 1.2% |
| Enhanced 1D CNN | 118.5 KB | 2.69 ms | 1.4% |
| Enhanced 2D CNN | 123.6 KB | 0.09 ms | 0.05% |
| CRNN-PCEN | 478.9 KB | 11.86 ms | 6.3% |

Every model finishes inside the 187.5 ms hop.

### D. PCEN Under a Changed Microphone

The CRNN input was band-limited (150 Hz to 8 kHz), clipped, gained between 0.5× and 1.5×, and noised at 15–30 dB SNR.

| τ | Clean | Distorted | Ambient (clean / dist.) |
|:---:|:---:|:---:|:---:|
| 0.10 | 69% | 69% | 75% / 74% |
| 0.30 | 62% | 66% | 78% / 77% |
| 0.50 | 61% | 61% | 80% / 77% |
| 0.70 | 59% | 59% | 84% / 81% |

A change of capsule does not by itself remove the detections. Fireworks and claps remain, because they are impulses too.

### E. Ensemble

Enhanced 2D CNN + CRNN-PCEN + Baseline 1D CNN, with an FFT spectral check (energy below 200 Hz, energy above 4 kHz, spectral centroid).

**Ensemble counts on 300 clips (100 per true class):**

| True class | Called real | Called fake | Called other |
|:---|:---:|:---:|:---:|
| Real gunshot | 77 | 15 | 8 |
| Hard imposter | 44 | 46 | 10 |
| Ambient | 20 | 38 | 42 |

Recall rises from 67% to 77% (77 of 100 shots). Combined size is about 715 KB and combined time about 17 ms, still inside the hop.

Complementary misses: on `127724.wav` the Enhanced 2D CNN scores 21.1% and the CRNN scores 99.3%. On `170167.wav` the CRNN scores 0 and the Enhanced 2D CNN scores 99.6%.

### F. Operating Points

| Setting | Recall | Spec. | Latency | Flash |
|:---|:---:|:---:|:---:|:---:|
| Enhanced 2D, peak | 38% | 99% | 0.09 ms | 124 KB |
| Enhanced 2D, sliding | 67% | 82% | 0.75 ms | 124 KB |
| Three-model ensemble | 77% | 80% | 17 ms | 715 KB |

---

## VIII. Conclusion and Future Work

Classical classifiers that reach near-perfect accuracy on balanced hold-out files fail on live input because temporal averaging destroys the few-millisecond shockwave signature. The move from 1D MFCC classifiers to 2D spectrogram CNNs consistently improved rejection of ambient and imposter sounds. On 300 unseen clips the Enhanced 2D CNN achieves 67% gunshot recall (sliding) and 99% ambient specificity (peak) in 0.09 ms within 124 KB. The PCEN-CRNN maintains stable recall under simulated microphone distortion with zero degradation, confirming PCEN as an effective domain-shift normalizer. The Enhanced 1D CNN is not deployable after INT8 conversion due to weight saturation in dilated 1D convolution filters.

A physical limitation was identified: a single microphone cannot separate aerial mortar fireworks from gunfire, as both produce identical shockwave rise-times and reverberation profiles. Resolving this requires spatial multi-microphone arrays.

**1) Physical Deployment.** Deploy the Enhanced 2D CNN INT8 model on a Raspberry Pi 4 to confirm edge timing, then port inference to the Arduino Nano 33 BLE Sense (on-board ST MP34DT05 PDM microphone).

**2) Duty-Cycled Sleep and One-Bit Transmission.** The node stays in deep sleep (~5 μA) until a coarse energy threshold triggers wake-up [16]. One inference cycle runs and the node sleeps again. If the score crosses the threshold, the node transmits a minimal packet: node ID (1 byte), timestamp (4 bytes), detection flag (1 bit), confidence (1 byte). Radio on-time is under 1.5 ms per trigger. If the score is below threshold, no transmission occurs. Raw audio never leaves the node.

**3) Multi-Node Spatial Consensus.** A single node next to a hand clap may send a false bit. The gateway raises an alert only when K ≥ 3 neighbouring nodes transmit within the acoustic propagation window (~100 ms for 34 m spacing). This rejects localized percussive events without streaming audio.

**4) Self-Healing Network and Dashboard.** Nodes report battery, sensor drift, and link failures. A dashboard on the gateway logs every trigger for post-incident review.

---

## References

1. R. C. Maher, "Acoustical characterization of gunshots," in *Proc. IEEE Workshop on Signal Processing Applications for Public Security and Forensics (SAFE)*, Washington, DC, USA, 2007, pp. 109–113.
2. R. Kabealo et al., "A multi-firearm, multi-orientation audio dataset of gunshots," *Data in Brief*, vol. 48, Art. no. 109091, Jun. 2023.
3. S. Saha et al., "Automated gunshot detection in forest environments using spectrogram CNNs," *Applied Acoustics*, vol. 215, Art. no. 110452, Jan. 2025.
4. G. Magee, "Gunshot detection system on Raspberry Pi," *IEEE Open Source Acoustic Repository*, 2019.
5. J. Salamon, C. Jacoby, and J. P. Bello, "A dataset and taxonomy for urban sound research," in *Proc. 22nd ACM Multimedia*, Orlando, FL, 2014, pp. 1041–1044.
6. V. Lostanlen et al., "Per-channel energy normalization: Why and how," *IEEE Signal Processing Letters*, vol. 26, no. 1, pp. 39–43, Jan. 2019.
7. V. Lostanlen et al., "Per-channel energy normalization for bioacoustic sound event detection," *IEEE/ACM Trans. Audio, Speech, Language Process.*, vol. 27, no. 12, pp. 2178–2190, Dec. 2019.
8. S. Hershey et al., "CNN architectures for large-scale audio classification," in *Proc. IEEE ICASSP*, New Orleans, LA, 2017, pp. 131–135.
9. W. Dai et al., "Very deep convolutional neural networks for raw waveforms," in *Proc. IEEE ICASSP*, New Orleans, LA, 2017, pp. 421–425.
10. K. Cho et al., "Learning phrase representations using RNN encoder-decoder for statistical machine translation," in *Proc. EMNLP*, Doha, Qatar, 2014, pp. 1724–1734.
11. H. Zhang et al., "mixup: Beyond empirical risk minimization," in *Proc. ICLR*, Vancouver, Canada, 2018.
12. D. S. Park et al., "SpecAugment: A simple data augmentation method for automatic speech recognition," in *Proc. Interspeech*, Graz, Austria, 2019, pp. 2613–2617.
13. T. Ko et al., "Audio augmentation for speech recognition," in *Proc. Interspeech*, Dresden, Germany, 2015, pp. 3586–3589.
14. C. Banbury et al., "MLPerf Tiny benchmark," in *Proc. NeurIPS Datasets and Benchmarks*, 2021.
15. R. David et al., "TensorFlow Lite Micro: Embedded machine learning on TinyML systems," in *Proc. MLSys*, 2021, pp. 800–811.
16. Edge Impulse, "TinyML acoustic anomaly detection on microcontrollers," Technical Report, 2024.
