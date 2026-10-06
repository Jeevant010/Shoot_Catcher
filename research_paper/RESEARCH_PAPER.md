# From Heuristic Filtering to Bioacoustic CRNNs: An Empirical Study on Edge-Deployable Acoustic Gunshot Detection

**Jeevant Sharma, Deepesh Dangi**  
Department of Computer Science and Engineering  
Indian Institute of Information Technology, Surat  
Under the guidance of Dr. Kaustubh Dhondge  

---

## Abstract

Acoustic gunshot detection on low-cost embedded edge platforms faces real physical challenges like acoustic domain shift across different microphones, impulse truncation during signal framing, microphone diaphragm saturation at high SPL levels, and false alarms from percussive imposter sounds like door slams, firecrackers, and hand claps. Over an eight-month research cycle, we worked through the full architectural progression from heuristic audio trimming and classical machine learning (Linear Regression, SVM, Random Forest) to deep CNN models (1D and 2D), enhanced dual-head architectures, and finally bioacoustic recurrent networks using Per-Channel Energy Normalization (PCEN) with Bidirectional GRU. We evaluated over 10,000 raw audio files and documented the critical failure modes we encountered at each stage. We also ran systematic experiments on Edge Impulse to evaluate MFCC vs MFE Spectrogram feature representations and 1D vs 2D CNN classifiers on microcontroller targets. On an independent benchmark of 300 unseen real-world audio clips (100 gunshots, 100 ambient, 100 hard imposters), our Enhanced 2D CNN achieved single-model edge inference in 0.09 ms with 99% ambient specificity, while a 3-model ensemble pushed gunshot recall to 77.0% within a combined 715 KB flash footprint and 14-17 ms latency. All edge profiling was conducted on a virtual machine throttled to 45-50% CPU to match ARM Cortex-A72 hardware limits, since the physical Raspberry Pi 4 units had boot failures. Our final deployment goal is on-device inference on the Arduino Nano 33 BLE Sense, transmitting only a single-bit alert (1 for gunshot, 0 otherwise) to a sink gateway for spatial aggregation, achieving near-zero energy consumption and complete civilian privacy.

**Keywords:** Acoustic gunshot detection, edge computing, TinyML, PCEN, convolutional recurrent neural networks, wireless sensor networks, on-device inference.

---

## I. Introduction

In US and Europe, people carry guns with licence, and even in India for big cities while in some other places people carry them illegally. For the initiative of starting a well-being of a typical city, we must find ways to save lives and decrease the abuse that happens in streets. With a typical technology, as we are in a digital age, we can have different types of possible applicable tools and devices that are able to capture the information of those gun sounds. There is also the part of heavy processing to send the data from some widespread microcontrollers, then speed it up to the forests and other national sanctuaries to preserve wildlife as well as the life of humans.

Why we are concerned — police is a typically powerful and helpful way, but for the typical way of living in a real world along with technologies, even with the technologies that are present, people use chainsaw detection and other similar services. The concept of a smart city aligns with our goal. If people are not co-operating, then the systems should do this job. In India, due to heavy crowd and more abusive community in some places, it is impossible for now, but at some states it can be started with colleges, schools, and institutions where we can deploy them as expected so that they truly work as we expect. Not only these, there are a lot of examples that address this issue while we ensure the goodwill to do it for society.

The gunshots for society are a typical example of a bad will, but with technology and not using humans, we can detect those fires. We can easily let people know where the event happens, while also ensuring that the person who gets injured gets medical support. Our law ideologies work with the shape of catching criminals and taking care of the loss caused by artificial weapons to society, with the help of tags along with IoT.

What we are trying to do here — we want to do the typical learning and run the inference on the Arduino BLE. But with the real-world problems and research-level complexities, we worked from the way that started natively: start with simple runs on PC/laptop only after training, then run it on PC, then shift to the Arduino with models converted from .h5 to .tflite format. Our goal is to create and run the inference on the Arduino Nano 33 BLE Sense.

In this paper, we worked through the typical journey of the research — we ask the model to do the job through inference and we need to send audio to it and it just says 1 or 0 for that case. We tried it on the Raspberry Pi on the virtual one. Along with it we tried it with 95% less of all the heavy settings, while this paper discusses the future aspects we are still working on, including our thresholds that are still being refined.

An authentic acoustic gunshot discharge consists of two physical phenomena:
1. A supersonic projectile shockwave (N-wave), with an abrupt non-linear pressure rise lasting 1 ms to 5 ms.
2. An explosive muzzle blast with an environmental reverberation decay tail spanning 200 ms to 600 ms.

When deployed on resource-constrained platforms like the Raspberry Pi with low-cost micro-sensors, models face non-linear transducer responses, ADC thermal hiss, electrical DC voltage bias, and acoustic imposter sounds (door slams, firecrackers, claps).

---

## II. Related Work

### A. Existing Gunshot Detection Systems
Commercial systems like ShotSpotter have validated the concept of acoustic surveillance for urban safety. However, these systems are proprietary, cloud-tethered, and require continuous streaming of raw audio from sensors to remote servers. This creates three problems: (a) high bandwidth and infrastructure costs, (b) civilian privacy concerns from recording conversations, and (c) dependence on internet connectivity which fails in remote forest or rural deployments.

### B. Edge Impulse and TinyML Platforms
Edge Impulse provides a development platform for building and deploying machine learning models on microcontrollers. We used Edge Impulse to systematically evaluate different DSP feature pipelines (MFCC vs MFE Spectrogram) and classifier architectures (1D CNN vs 2D CNN) targeting the ARM Cortex-M4F. Their platform gave us INT8 quantized models with on-device RAM and Flash profiling, which helped us understand the real constraints of microcontroller deployment.

### C. Bioacoustic PCEN from Rainforest Research
The Per-Channel Energy Normalization (PCEN) approach comes from bioacoustic research — specifically from NYU and DCASE work on illegal chainsaw detection in rainforests. PCEN replaces static Log-Mel spectrograms with a dynamic normalization that automatically suppresses stationary background noise while amplifying sharp transient events. This is exactly what we need for gunshot detection, where the muzzle blast is a rapid transient sitting on top of continuous urban noise. CNN is something everybody uses, and the PCEN we adapted from existing bioacoustic research. But what is unique is how we combined it with our multi-model ensemble and edge deployment pipeline.

### D. Voronoi-Based WSN Spatial Consensus
For multi-node spatial aggregation, we draw from Voronoi tessellation approaches used in wireless sensor networks. This does not use any proprietary standards but works with open algorithms. We use this spatial partitioning for consensus-based event validation — requiring multiple sensor nodes within a Voronoi cell to confirm a detection before alerting, which eliminates single-node false positives from localized percussive noises.

---

## III. System Design and Implementation

### A. Data Collection and Cleaning

For implementing, we started with cleaning of data where all data are passed through our pipeline. We used librosa and many different ways to process the audio. Cleaning was a major task in ML. We took our data from sources like the Data in Brief multi-firearm dataset (DOI: 10.1016/j.dib.2023.109091) containing 4,138 raw firearm recordings, and ambient acoustic repositories with 6,230 long-duration recordings (10 seconds to 60 minutes) of urban traffic, rainfall, construction, human speech, and room acoustics. The total raw corpus exceeded 10,000 files.

We started to trim the audio along 250 ms, 500 ms, and 1 second segments. We tried to trim out the ones which are useless. The dataset contained around 80K files, so we checked on 1K to 3-5K files by manually listening through the cycle till we satisfied ourselves that the data was cleaned well.

**The 250 ms Truncation Error:** Our initial trimming scripts used fixed 250 ms windows centered on energy peaks. This was a critical engineering flaw — a 250 ms window cuts off the 200-600 ms environmental reverberation decay tail, making gunshots sound identical to door slams or balloon pops. Fixed-interval slicing also split shockwave attack envelopes across clip boundaries. We rebuilt the pipeline into `Updated_Trimmer` with 750 ms and 1000 ms windows, 3-way onset detection (spectral flux, raw amplitude peak, smoothed envelope), and 3 physical quality gates (Crest Factor, Prominence, Attack Ratio).

### B. Classical Supervised Learning Phase

Along the way we worked on the supervised models. We started with basic approaches — linear regression for a classification task, and also to create a live system to get some thresholds and give it a try, to see which model can be the lowest baseline. We tried and got something from it — understanding why and how it fails. We moved to the 2 supervised models: the SVM and Random Forest.

We tried these models with changes of dataset ratios which go like 1:1, then 1:10 for gunshot vs non-gunshot, then giving it to train. We started working along the way to get results that gave like 99-100% accuracy on the test set with some data. We still tried to get more with different combinations. With all these combinations, we tried to get results and finally understood the total gunshot detection capability needs ratios from 1:10 to 1:50, so along the way this became a different set of training requirements. We created different trims which split the data into better segments.

This typically works along with some more configurations of librosa and we tried to match it with a splitter set which gives up more types of different sounds. We got more different types of values for these tasks and got other fixed kind of sets. Then we stopped and verified it can create dynamic values of a processing pipeline for our purpose. These typical pipelines were almost final for us, and even trying these with our supervised learning models SVM and RF — but still on real microphone it gets bad values.

**Why Classical ML Failed:** We tried recorded results and got some results, but on live microphone input, both SVM and Random Forest collapsed into continuous false alarms. The root cause: computing static MFCC summaries requires averaging spectral frames across the temporal window. This temporal averaging smoothed out the sub-3 ms shockwave attack into a blurred frequency envelope, discarding the primary physical signature. Everyday imposter sounds like dropped cutlery, squeaky vehicle brakes, and jingling keys possess overlapping high-frequency spectral centroids. This proved that manual static feature engineering is fundamentally unsuited for explosive acoustic events.

### C. Edge Impulse Exploration Phase

Before starting with our own CNN models, we moved to the Edge Impulse platform with preprocessing CNN spectrograms and some field models to see results. But still we saw a lot better accuracy for those models in the notebook, and saw very bad results for live audio. The problems were still part of our flow. In the next suite, we saw some other examples to work on regarding these fixes, but at the end we created a report using it, and finally it was time to move past these preexisting models and use the CNN models with our own way to try.

We ran 5 systematic experiments on Edge Impulse targeting the ARM Cortex-M4F (64 MHz, 256 KB SRAM):

**Experiment 1 — MFCC + 1D CNN (The "Voice-Trap"):**
- DSP: 250 ms window, 32 Mel filters, 13 MFCC coefficients, FFT length 512. Processing: 9 ms, 17 KB RAM.
- Architecture: 1D Conv (8 filters) → 1D Conv (16 filters) → Dropout (0.25) → Softmax.
- Validation: 99.4% accuracy, 0.02 loss. INT8: 1 ms inference, 11.9 KB RAM, 45.4 KB Flash.
- Live Failure: On continuous ambient audio, 33 out of 47 windows were false positives (70.2% false alarm rate). MFCC averaged out the shockwave transient, making speech formants look identical to muzzle blasts.

**Experiment 2 — MFCC + Anomaly Detection:**
- Added K-means anomaly clustering over MFCC features. DSP: 120 ms.
- On ambient audio, 47 out of 48 windows were flagged as anomalies (distances 7.01-36.74). The underlying classifier still couldn't distinguish gunshots from other loud sounds.

**Experiment 3 — MFE Spectrogram + 1D CNN:**
- DSP: MFE Spectrogram (2 ms frame length, 1.5 ms stride, FFT 32, noise floor -77 dB). 3,383 features. 52 ms DSP, 14 KB RAM.
- Validation: 98.8% accuracy, 12 ms INT8 inference, 18.1 KB RAM, 47.1 KB Flash.
- Live: 24 out of 48 windows were false alarms (50%). Shifting to spectrograms broke the voice-trap, but 1D convolution over unrolled bins couldn't capture 2D time-frequency patterns.

**Experiment 4 — MFE Spectrogram + 2D CNN (The Turning Point):**
- Architecture: 2D CNN processing the spectrogram grid directly.
- Validation: 99.6% accuracy, ROC-AUC 1.00. INT8: 167 ms, 30.2 KB RAM, 52.2 KB Flash.
- Live Imposter Tests:
  - Heavy Wind: 22 correct ambient rejections, only 7 false alarms (75.8% rejection).
  - Thunder: 20 correct ambient rejections, only 5 false alarms (80.0% rejection).
  - True Gunshot (12-gauge): Sharp probability transitions from 0.06 to >0.94 at blast onsets.

**Experiment 5 — MFE Spectrogram + Anomaly Detection:**
- Combined 2D CNN with K-means anomaly clustering. Validation: 99.7% accuracy.
- Provides dual-gate decision: 2D CNN assesses gunshot probability, anomaly score discards non-ballistic spikes.

| Experiment | Feature | Classifier | Val Acc | INT8 Latency (M4F) | RAM | Flash | Live Rejection |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---|
| Exp 1: MFCC 1D CNN | 13 MFCCs | 1D Conv | 99.4% | 1 ms | 11.9 KB | 45.4 KB | Failed (70% false alarms) |
| Exp 2: MFCC + Anomaly | MFCCs + K-means | 1D Conv + Anomaly | 99.4% | 2 ms | 11.9 KB | 45.4 KB | Failed (98% anomaly flags) |
| Exp 3: MFE Spec 1D CNN | MFE Spectrogram | 1D Conv | 98.8% | 12 ms | 18.1 KB | 47.1 KB | Moderate (50% false alarms) |
| Exp 4: MFE Spec 2D CNN | MFE Spectrogram | 2D CNN | 99.6% | 167 ms | 30.2 KB | 52.2 KB | High (76-80% rejection) |
| Exp 5: MFE Spec + Anomaly | MFE + Clusters | 2D CNN + Anomaly | 99.7% | 167 ms | 30.2 KB | 52.2 KB | Very High |

### D. Deep CNN Architecture Progression

Then we looked for the CNN architecture which started with 1D CNN and moved to some other models — 2D CNN, 3D CNN, enhanced versions. Giving some time to these models, we got a research paper on the PCEN (Per Channel Energy Normalization). So we tried to work out in a typical way from what we can, trying to just get some better results. But PCEN needs a different kind of cleaning of the file before applying it.

We engineered and benchmarked five model architectures:

**1. Baseline 1D Raw-Waveform CNN (92,513 parameters):**
Operates directly on raw discrete audio arrays (16,537 samples at 22,050 Hz). The first layer uses an 80-sample wide convolutional kernel (3.63 ms temporal span), acting as a learnable bank of 32 FIR wavelet filters tuned to supersonic N-wave rise times. Achieved 99.7% test accuracy on clean files, but recall fell to 30.0% across different microphones and reverberant spaces. Raw 1D waveforms proved sensitive to transducer frequency responses.

**2. Baseline 2D Mel-Spectrogram CNN (249,985 parameters):**
Transforms audio into 64 Mel-frequency bands over 130 time frames. Four Conv2D stages (32→64→128→128) extract spatial time-frequency representations. Initially suffered complete "Dying ReLU" collapse (0.0% recall, 679 false negatives) because unnormalized decibel values (-80 to 0 dB) drove ReLU activations to zero. Normalizing inputs to [0,1] via (S_dB + 80)/80 restored convergence to 99.86% ROC-AUC.

**3. Enhanced 1D CNN (96,674 parameters) — Dual-Head:**
Dilated 1D convolutions with dual-head output (gunshot probability + anomaly score). Heavy class weights (C1=6.80, C0=0.54) created paranoia — the model predicted 100% gunshot on every input including silence. During INT8 quantization, the narrow dynamic range of raw waveform dilated convolutions collapsed, permanently saturating logits to the positive class. This is a documented quantization failure case.

**4. Enhanced 2D CNN (110,594 parameters) — Production Champion:**
Log-Mel spectrogram with MixUp data augmentation, SpecAugment (contiguous time and frequency masking), and dual-head classification. This became our primary production model. INT8 quantized size: 123.6 KB. Edge VM latency: 0.09 ms.

**5. Robust CRNN-PCEN (397,842 parameters):**
Replaces static Log-Mel spectrograms with dynamic PCEN:

$$P[f, t] = \left( \frac{S[f, t]}{(\epsilon + M[f, t])^{\alpha}} + \delta \right)^r - \delta^r$$

where M[f,t] is a causal background noise estimate via first-order IIR filter. Under stationary noise, the steady-state gain converges to S₀^0.02 ≈ 1.0, mathematically flattening continuous ambient noise. When microphone sensitivity changes by gain factor g, output scales as g^0.04 ≈ 1.0, making the feature representation invariant across different hardware microphones.

The extracted features are processed by a Bidirectional GRU (2×64 units). The forward GRU tracks the rapid shockwave onset; the backward GRU checks the exponential reverberation decay back to the attack point, rejecting sustained tonal sounds (sirens, vehicle horns). Achieved 100.0% recall with zero false alarms on continuous ambient noise on a 75-clip benchmark.

### E. Hardware Conditioning for Edge Deployment

When transitioning from clean WAV files to embedded micro-sensors (electret capsules and I2S MEMS transducers connected to Raspberry Pi), we encountered:

- **DC Voltage Rail Drift:** Low-cost sensors on 3.3V or 5.0V supply rails cause baseline drift. Solution: sample mean subtraction (y_clean = y - mean(y)).
- **Thermal ADC Hiss Amplification:** Standard peak normalization on quiet ambient amplifies noise by 500x, triggering false alarms. Solution: energy-gated normalization — windows below RMS threshold of 0.005 bypass peak scaling.
- **Diaphragm Saturation:** Firearms at close range exceed 120 dB SPL, flattening peaks into square waves. Solution: hard-clipping and non-linear compression augmentation during training.
- **Buffer Framing:** 750 ms window (16,537 samples) with 187 ms hop step (75% overlap), guaranteeing any transient lands fully within at least one analysis frame.

---

## IV. Experimentation

### A. Evaluation Dataset

We created an independent test suite of 300 unseen audio files with zero overlap with training data:

| Class | Count | Contents | Ground Truth |
|:---|:---:|:---|:---:|
| Real Gunshots | 100 | Firearms, handguns, rifles, machine guns, field recordings with distance reverberation | Positive |
| Ambient Non-Gunshots | 100 | Rain, wind, urban traffic, engines, human speech, footsteps, dogs barking | Negative |
| Hard Imposters | 100 | Fireworks, firecrackers, door slams, explosions, thunderclaps, snare drums, wood snaps | Negative |

### B. Hardware Platforms

| Platform | Configuration | Purpose |
|:---|:---|:---|
| Host CPU (x86_64) | Intel Core i5/i7, 4 threads, full speed | Feature pipeline validation, PCEN calibration, multi-model ensemble evaluation |
| RPi4 Edge VM (VirtualBox) | 4 vCPUs throttled at 45% execution cap, 2048 MB RAM | Realistic embedded hardware latency profiling, INT8 inference accuracy verification |

The physical Raspberry Pi 4 units had a boot failure due to corrupted SD cards, so we conducted edge profiling on a VirtualBox VM with the CPU execution cap throttled to 45-50%, matching the approximate computational throughput of the quad-core ARM Cortex-A72 at 1.5 GHz.

### C. Evaluation Paradigms

We evaluated models under three paradigms:

1. **Peak-Centered:** A single 750 ms slice centered on the clip's maximum energy peak. Extremely fast but limited recall because distant gunshots don't always have an isolated amplitude peak.
2. **Sliding-Window:** Steps a 750 ms window across the entire clip with 50% overlap (375 ms stride). Better recall by scanning the full audio duration.
3. **Multi-Model Ensemble:** Combines Enhanced 2D CNN + Robust CRNN-PCEN + Baseline 1D CNN with adaptive OR-voting and physics-based acoustic transient rejection (low-frequency blast energy, high-frequency crack ratio, spectral centroid filtering).

### D. PCEN Domain Robustness Sweep

To test whether PCEN truly provides microphone invariance, we ran the CRNN-PCEN model on both clean audio and synthetically distorted audio (bandpass filter 150-8000 Hz, non-linear saturation clipping at ±0.60-0.95, additive Gaussian noise at 15-30 dB SNR, gain perturbation 0.5x-1.5x) across multiple decision thresholds.

---

## V. Results

### A. Latency Profiling on Edge VM (100 Iterations Per Model)

| Model | Parameters | INT8 Size | Latency (Mean ± Std) | P95 Latency | Hop Budget Used |
|:---|:---:|:---:|:---:|:---:|:---:|
| Baseline 1D CNN | 92,513 | 112.5 KB | 2.29 ± 8.25 ms | 1.98 ms | 1.2% |
| Baseline 2D CNN | 249,985 | 259.9 KB | 2.15 ± 6.76 ms | 1.69 ms | 1.2% |
| Enhanced 1D CNN | 96,674 | 118.5 KB | 2.69 ± 8.23 ms | 6.48 ms | 1.4% |
| Enhanced 2D CNN | 110,594 | 123.6 KB | 0.09 ± 0.01 ms | 0.10 ms | 0.05% |
| Robust CRNN-PCEN | 397,842 | 478.9 KB | 11.86 ± 4.12 ms | 14.50 ms | 6.3% |

All five architectures execute well beneath the 187.5 ms real-time hop deadline. Zero audio buffer drops are guaranteed during continuous streaming.

### B. Single-Model Results — Sliding-Window Mode (threshold = 0.50)

| Model | Gunshot Recall | Ambient Specificity | Imposter Rejection | Mean Latency |
|:---|:---:|:---:|:---:|:---:|
| Baseline 1D CNN | 45.0% | 85.0% | 72.0% | 2.79 ms |
| Baseline 2D CNN | 66.0% | 88.0% | 59.0% | 5.36 ms |
| Enhanced 1D CNN | 100.0% (Degenerate) | 0.0% | 0.0% | 3.36 ms |
| Enhanced 2D CNN | 67.0% | 82.0% | 66.0% | 0.75 ms |
| Robust CRNN-PCEN | 57.0% | 86.0% | 61.0% | 6.26 ms |

### C. Single-Model Results — Peak-Centered Mode (threshold = 0.50)

| Model | Gunshot Recall | Ambient Specificity | Imposter Rejection | Mean Latency |
|:---|:---:|:---:|:---:|:---:|
| Baseline 1D CNN | 30.0% | 95.0% | 84.0% | 2.84 ms |
| Baseline 2D CNN | 45.0% | 94.0% | 77.0% | 4.31 ms |
| Enhanced 1D CNN | 100.0% (Degenerate) | 0.0% | 0.0% | 4.90 ms |
| Enhanced 2D CNN | 38.0% | 99.0% | 89.0% | 0.76 ms |
| Robust CRNN-PCEN | 33.0% | 97.0% | 85.0% | 6.41 ms |

Sliding-window mode consistently boosts gunshot recall by 15-29 percentage points compared to peak-centered, at the cost of higher false positives. Peak mode achieves superior specificity (up to 99%) — better for low-false-alarm deployments. This recall-specificity trade-off is what motivated the multi-model ensemble approach.

### D. PCEN Domain Robustness Results (Sliding-Window Mode)

| Threshold | Clean Recall | Distorted Recall | Delta Recall | Clean Ambient | Distorted Ambient | Clean Imposter Rej | Distorted Imposter Rej |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.10 | 69.0% | 69.0% | 0.0% | 75.0% | 74.0% | 48.0% | 52.0% (+4%) |
| 0.30 | 62.0% | 66.0% | +4.0% | 78.0% | 77.0% | 53.0% | 59.0% (+6%) |
| 0.50 | 61.0% | 61.0% | 0.0% | 80.0% | 77.0% | 56.0% | 61.0% (+5%) |
| 0.70 | 59.0% | 59.0% | 0.0% | 84.0% | 81.0% | 59.0% | 66.0% (+7%) |

Under aggressive simulated microphone distortion, PCEN exhibits zero recall degradation (61.0% → 61.0% at threshold 0.50, and 69.0% → 69.0% at threshold 0.10). Imposter rejection actually improved under distortion (+5-7%), confirming PCEN's role as a domain-shift normalizer for edge microphone variability.

### E. Multi-Model Ensemble Results (3-Way Confusion Matrix)

Ensemble combines Enhanced 2D CNN (Log-Mel) + Robust CRNN-PCEN + Baseline 1D CNN with adaptive OR-voting and physics-based acoustic transient rejection.

| True Class | Pred: Real Gunshot | Pred: Fake Gunshot | Pred: Not Gunshot | Total | Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|
| Real Gunshot | **77** | 15 | 8 | 100 | 77.0% (Recall) |
| Hard Imposter | 44 | **46** | 10 | 100 | 46.0% |
| Ambient Non-Gunshot | 20 | 38 | **42** | 100 | 80.0% rejected as firearms |

### F. Paradigm Comparison Summary

| Paradigm | Gunshot Recall | Ambient Specificity | Edge VM Latency | Flash Footprint |
|:---|:---:|:---:|:---:|:---:|
| Peak-Centered (Single Best) | 38.0% | 99.0% | 0.09 ms | 123 KB |
| Sliding-Window (Dynamic) | 61.0% | 80.0% | 11.86 ms | 478 KB |
| Multi-Model Ensemble | **77.0%** | 80.0% | 17.04 ms | 715 KB |

The multi-model ensemble achieves the highest gunshot recall at 77.0% (up from 38% single-model peak). Combined, all three models take only 715 KB of flash storage and 14-17 ms of latency, using less than 10% of the real-time audio window.

### G. Host CPU vs Edge VM Cross-Platform Consistency

| Model | Host CPU Recall (Peak) | VM Edge Recall (Peak) | Consistency |
|:---|:---:|:---:|:---:|
| Baseline 1D CNN | 35.6% | 30.0% | ±5.6% |
| Baseline 2D CNN | 49.2% | 45.0% | ±4.2% |
| Enhanced 1D CNN | 100.0% (Degenerate) | 100.0% (Degenerate) | Consistent failure |
| Enhanced 2D CNN | 44.1% | 38.0% | ±6.1% |
| Robust CRNN-PCEN | 32.0% | 33.0% | ±1.0% |

Cross-platform accuracy variance of ±4-6% is expected and attributable to INT8 quantization rounding differences between x86 XNNPACK and ARM NEON kernels.

### H. Enhanced 1D CNN — Quantization Failure Case Study

The Enhanced 1D CNN INT8 model predicts 100% gunshot on every input (including pure silence, ambient noise, and door slams) in both sliding and peak modes across both CPU and VM platforms. During post-training INT8 quantization, the dual-head raw waveform architecture suffered weight saturation — dilated 1D convolution filters operating directly on raw audio samples have extremely narrow dynamic ranges that collapse under 8-bit precision. The model's logits are permanently pinned to the positive class. This serves as an empirical case study demonstrating that raw waveform 1D CNN architectures are significantly more fragile under INT8 quantization than time-frequency 2D CNN representations.

---

## VI. Conclusion and Future Work

In the end, we can say that — whether it fully worked or not is still a type of ongoing work, and we will look forward to it. When the Raspberry Pi didn't work well due to boot failures, we have all those results from our tried steps on the virtual machine. For the results, we finally have the conclusion which consists of our latest execution along with the future aspects. We have tried it finally on the virtual Raspberry Pi.

The experimentation covered the full points from start — from the first supervised models (Linear Regression) all the way to this virtual Raspberry Pi testing. It is like a 6-page paper, so the best thing to do is keep only the best and the most relevant working outputs. The key findings are:

1. **Classical ML Fails on Live Audio:** SVM and Random Forest achieve 99-100% on closed test sets but collapse on real microphone input due to temporal feature averaging destroying the shockwave attack profile.
2. **MFCC + 1D CNN Has a "Voice-Trap":** On Edge Impulse, 99.4% validation accuracy translated to 70% false alarm rate on live ambient audio. The fix was moving to MFE Spectrograms + 2D CNN.
3. **2D CNN on Spectrograms is the Turning Point:** 99.6% validation accuracy with 76-80% rejection of wind and thunder imposters in live testing.
4. **PCEN Provides Microphone Invariance:** Zero recall degradation under aggressive synthetic microphone distortion, confirming it as a domain-shift normalizer.
5. **Multi-Model Ensemble Reaches 77% Recall:** Combining complementary architectures via OR-voting pushes recall from 38% (single peak) to 77%, within a 715 KB footprint and 17 ms latency.
6. **Enhanced 2D CNN is the Edge Champion:** 0.09 ms inference, 123 KB flash, 99% ambient specificity — the fastest and most efficient single model.

### Future Work

Our next step is typically for the course of work related to the hardware. As we found some issues on the Raspberry Pi, first we will fix it — replace the corrupted SD card, deploy the TFLite models on physical hardware, and verify the VM benchmark numbers match real ARM Cortex-A72 performance. Then finally go on the Arduino Nano 33 BLE Sense, which is our final goal to deploy.

The planned work covers:

1. **Physical Raspberry Pi 4 Deployment:** Replace the failed SD card, boot the physical hardware, and run the same 300-clip benchmark to validate against VM results.

2. **Arduino Nano 33 BLE Sense Firmware:** Deploy the Enhanced 2D CNN (123 KB INT8) onto the nRF52840 Cortex-M4F (1 MB Flash, 256 KB SRAM). The on-board ST MP34DT05 PDM microphone provides 64 dB SNR omnidirectional capture. The model fits comfortably within the flash and SRAM constraints.

3. **Sleep Scheduling and Battery Optimization:** For a real-world IoT deployment, continuous full-power audio classification is not practical on battery-operated nodes. We will implement a two-tier duty-cycling protocol:
   - Deep sleep mode (~5 uA) with a low-power acoustic energy threshold trigger.
   - Burst inference mode — wake up, buffer 250-500 ms via DMA, run INT8 inference (1-167 ms), and return to sleep.
   This extends node battery life from days to potentially years on small lithium batteries.

4. **Single-Bit Binary Transmission (1-Bit Alert):** The most powerful optimization for the wireless sensor network. The node never transmits raw audio. Upon local classification exceeding the confidence threshold, the node broadcasts a minimal 1-bit event packet: Node ID (1 byte), timestamp (4 bytes), detection flag (1 bit), confidence (1 byte). The radio on-time is under 1.5 ms per trigger, yielding near-zero RF energy consumption. For 0 (no gunshot), the node stays completely silent — no transmission at all. This is effectively a binary neural network output at the communication level: fire 1 if gunshot, stay silent otherwise. This guarantees complete civilian privacy since conversational audio never leaves the chip.

5. **Sink Aggregation and Spatial Consensus at the Gateway:** The Raspberry Pi 4 serves as the central sink/aggregation hub. A single isolated trigger from one node is treated as unconfirmed. The gateway requires at least K >= 3 neighboring nodes within a Voronoi spatial cell to report a detection within the acoustic propagation window (~100 ms for 34 meters). This eliminates false alarms from localized percussive noise. Using high-resolution timestamps from anchor nodes, hyperbolic TDOA multilateration computes real-time geographic coordinates for emergency dispatch.

6. **Self-Healing Network Capabilities:** Nodes monitor their own health (battery voltage, sensor drift, communication failures) and report status to the gateway. Failed nodes are detected via heartbeat timeouts, and the spatial consensus algorithm adapts the required quorum based on active node density within each Voronoi cell.

7. **Interactive Dashboard:** A real-time monitoring dashboard on the gateway hub for visualizing node status, detection events, spatial heat maps, and alert history. This provides operational auditability — every trigger event is logged with timestamps and confidence scores for post-incident review.

8. **Threshold Optimization:** ROC and DET curve analysis for optimizing the multi-model decision boundary. The recall-specificity trade-off between peak-centered (99% specificity, 38% recall) and sliding-window (80% specificity, 67% recall) modes suggests that adaptive threshold tuning based on deployment context (urban street vs forest reserve vs campus perimeter) will be needed.

---

## Acknowledgment

The authors thank Dr. Kaustubh Dhondge for his guidance throughout this research project, the contributors to the open-source Data in Brief gunshot dataset, the Edge Impulse TinyML platform, and the NYU bioacoustics research community for foundational work on PCEN and adaptive sound event detection.

---

## References

1. G. Magee, "Gunshot Detection System on Raspberry Pi," IEEE Open Source Acoustic Repository, 2019.
2. Data in Brief, "A multi-firearm, multi-orientation audio dataset of gunshots," *Data in Brief*, vol. 48, p. 109091, 2023. DOI: 10.1016/j.dib.2023.109091.
3. S. Saha et al., "Automated Gunshot Detection in Forest Environments Using Spectrogram CNNs," *Applied Acoustics*, vol. 215, 2025.
4. V. Lostanlen, K. Cramer, S. Farnsworth, F. Briggs, and J. Bello, "Per-Channel Energy Normalization for Bioacoustic Sound Event Detection," *IEEE/ACM Trans. Audio, Speech, Lang. Process.*, vol. 27, no. 12, pp. 2178-2190, 2019.
5. Edge Impulse, "TinyML Acoustic Anomaly Detection on Microcontrollers," Tech. Rep., 2024.
6. S. Hershey et al., "CNN Architectures for Large-Scale Audio Classification," in *Proc. IEEE ICASSP*, 2017, pp. 131-135.
7. H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz, "mixup: Beyond Empirical Risk Minimization," in *Proc. ICLR*, 2018.
8. D. S. Park et al., "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition," in *Proc. Interspeech*, 2019, pp. 2613-2617.
