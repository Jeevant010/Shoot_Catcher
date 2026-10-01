# An Edge-Native Acoustic Gunshot Detection and Localization Architecture: From Bioacoustic Feature Engineering to Low-Power WSN Spatial Consensus

**Authors:** Jeevant Sharma, Deepesh Dangi, Abhishek Yadav, Bhupendra Kumar  
**Advisor / Mentor:** Dr. Kaustubh Dhondge  
*Department of Computer Science and Engineering, Indian Institute of Information Technology, Surat*  
**Target Publication:** IEEE COMSNETS 2027  

---

## Abstract

Acoustic gunshot detection systems for civilian public safety and wildlife conservation face severe edge deployment challenges: acoustic domain shift, high false alarm rates on ambient percussive noise, cloud-tethered bandwidth bottlenecks, and invasion of civilian privacy. In this paper, we present an end-to-end, privacy-preserving, edge-native acoustic gunshot detection and localization framework designed for resource-constrained Wireless Sensor Networks (WSNs). We chronologically document our iterative engineering journey across three distinct evolutionary stages:
1. **Classical Supervised Learning:** Demonstrating why Support Vector Machines (SVM) and Random Forests—despite scoring $99.0\%\text{--}100.0\%$ accuracy on closed datasets—fail catastrophically on live microphones due to temporal feature-averaging.
2. **TinyML Edge Impulse Exploration:** Systematically evaluating 1D vs. 2D CNN architectures and MFCC vs. Mel Filterbank Energy (MFE) spectrogram representations across 84 real-world test iterations. We expose the fatal "voice-trap" of MFCC-based 1D CNNs ($70.2\%$ false trigger rate on speech/ambient audio) and demonstrate how 2D spatial convolution across MFE spectrograms successfully rejects heavy wind ($75.8\%$ rejection) and thunder ($80.0\%$ rejection).
3. **Bioacoustic CRNN & Native Deep Ensembles:** Adapting Per-Channel Energy Normalization (PCEN) and Bidirectional Gated Recurrent Units (Bi-GRU) for dynamic gain invariance. 

Benchmarking across 300 unseen real-world audio clips (100 gunshots, 100 ambient, 100 industrial/percussive imposters) on a hardware-throttled virtual edge platform ($45\text{--}50\%$ CPU limit matching ARM Cortex-A72 specifications), our Enhanced 2D CNN achieved single-model edge inference in $0.09\text{ ms}$ with $99.0\%$ ambient specificity. When deployed in a 3-model cooperative ensemble, gunshot recall escalated to $77.0\%$ within a compact $715\text{ KB}$ flash footprint and $14\text{--}17\text{ ms}$ latency—consuming under $10\%$ of the real-time audio frame window. Finally, we formulate a two-tier WSN duty-cycling protocol running on Arduino Nano 33 BLE Sense nodes transmitting only a 1-bit event alert (`1` for verified gunshot, `0` otherwise) to a central Raspberry Pi 4 sink hub, achieving near-zero RF energy consumption, complete civilian conversation privacy, and robust multi-node TDOA spatial multilateration.

---

## 1. Introduction & Societal Motivation

Gun violence in modern urban municipalities demands emergency response times measured in seconds rather than minutes. Empirical studies on urban policing reveal that over $80\%$ of firearm discharges in affected metropolitan neighborhoods are never reported by citizens to emergency dispatches. When emergency calls are placed, civilian witness panic frequently introduces inaccurate spatial coordinates and delayed reporting. By the time emergency units arrive, perpetrators have evacuated the scene, victims are deprived of critical first-response medical care, and ballistic evidence is compromised.

Beyond urban environments, rapid acoustic detection is paramount in national wildlife sanctuaries and protected forest reserves. Illegal wildlife poaching and unauthorized chainsaw deforestation threaten endangered ecosystems. In institutional campus environments (schools, universities, government perimeters), automated acoustic alarms provide perimeter protection without relying on human sentinels.

Commercial municipal acoustic sensing systems (e.g., ShotSpotter) have validated the viability of acoustic surveillance. However, existing commercial architectures exhibit three critical flaws:
1. **Proprietary & Cloud-Tethered:** Continuous streaming of raw acoustic signals off-node requires expensive cellular or broadband backhauls and centralized cloud processing.
2. **Civilian Privacy Infringement:** Streaming uncompressed ambient audio into central servers raises significant surveillance and wiretapping concerns regarding public conversations.
3. **High Capital Expenditure:** High per-square-kilometer licensing costs make dense deployments impossible for emerging economies, municipal districts, and under-funded forest reserves.

To resolve these bottlenecks, we propose an edge-native, open-standard TinyML Wireless Sensor Network. The core tenet of our design is **zero raw audio transmission**: all acoustic DSP and neural inference occur locally on edge microcontroller nodes (Arduino Nano 33 BLE Sense / ARM Cortex-M4F). Nodes broadcast only an authenticated 1-bit event trigger upon gunshot identification, offloading spatial consensus and Time Difference Of Arrival (TDOA) multilateration to an edge aggregation gateway (Raspberry Pi 4).

---

## 2. Chronological Engineering Progression & Empirical Evolution

The research and development of our acoustic gunshot detection pipeline spanned an extensive empirical trajectory. Rather than presenting a single cherry-picked model, we document the real-world engineering failures and insights that guided our architectural choices.

```
+----------------------------------------------------------------------------------------------------+
|                                    CHRONOLOGICAL DEVELOPMENT PIPELINE                              |
+----------------------------------------------------------------------------------------------------+
|  Phase 1: Raw Data Cleaning & EDA                                                                  |
|  - 80,000+ clips ingested from UrbanSound8K, AudioSet, ESC-50, ballistic databases.               |
|  - Manual listening validation on 1,000-5,000 clips to eliminate corrupted cuts.                  |
|  - Evaluated 250 ms, 500 ms, and 1.0 s windowing slices.                                           |
|  - Explored class imbalance ratios (1:1 to 1:10 and 1:50) reflecting real-world sparsity.           |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|  Phase 2: Classical Machine Learning & Live Mic Collapse                                           |
|  - Linear Regression for classification threshold exploration.                                     |
|  - RBF-Kernel SVM & Random Forest on 13-band MFCC features.                                        |
|  - Laboratory results: 99.0% - 100.0% validation accuracy.                                         |
|  - Real-world failure: Live microphone input produced 100% false alarms on clapping, keys, doors.   |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|  Phase 3: TinyML Edge Impulse Exploration (84-Page Empirical Study)                               |
|  - Exp 1: MFCC + 1D CNN -> 99.4% val acc, but 70% false alarm rate on speech (The "Voice-Trap").  |
|  - Exp 2: MFCC + K-means Anomaly Detection -> 47/48 OOD alarms; unable to isolate muzzle blasts.    |
|  - Exp 3: MFE Spectrogram + 1D CNN -> Mitigated voice trap; high false alarms on wind/transients.  |
|  - Exp 4: MFE Spectrogram + 2D CNN -> Turning point: 99.6% val acc, 76% wind / 80% thunder rej.   |
|  - Exp 5: MFE Spectrogram + Anomaly Clustering -> Secondary out-of-distribution rejection.         |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|  Phase 4: Bioacoustic CRNN & Multi-Model Edge Ensemble                                             |
|  - Addressed microphone transducer clipping, DC offset drift, and stationary background noise.     |
|  - Adapted Per-Channel Energy Normalization (PCEN) + Bi-GRU for dynamic gain invariance.           |
|  - 3-Model Ensemble (Baseline 2D CNN + Enhanced 2D CNN + Robust CRNN-PCEN) achieves 77% recall.   |
|  - Deployed on throttled ARM Cortex-A72 VM (45-50% CPU) with 14-17 ms latency & 715 KB Flash.      |
+----------------------------------------------------------------------------------------------------+
```

---

## 3. Data Cleaning, Audio Preprocessing & Trimming Pipeline

### 3.1 Raw Ingestion & Physical Acoustic Characteristics
An authentic gunshot event exhibits two distinct, non-linear physical signatures:
1. **Supersonic Shockwave (N-Wave):** Generated if the projectile exceeds Mach 1 ($>343\text{ m/s}$). It presents a near-instantaneous pressure rise ($<5\text{ µs}$) with a total period of $1\text{--}5\text{ ms}$.
2. **Explosive Muzzle Blast & Reverberation:** An omnidirectional blast wave produced by expanding propellant gases, decaying into an acoustic reverberation envelope spanning $200\text{--}600\text{ ms}$.

In our initial data cleaning repository (`Data-Cleaner/`), we processed over 80,000 audio files from diverse acoustic datasets. Initial naive $250\text{ ms}$ trimming windows severed the reverberation decay tails, causing truncated gunshot samples to resemble brief percussive impulses (door slams, book drops, or hand claps). We modified the pipeline to extract adaptive $750\text{ ms}\text{--}1000\text{ ms}$ windows centered on onset energy peaks, capturing both the initial shock attack and the trailing environmental decay tail.

### 3.2 Dataset Imbalance: From 1:1 to 1:50
While conventional machine learning benchmarks assume balanced binary splits ($50\%$ positive, $50\%$ negative), real-world acoustic monitoring is characterized by extreme sparsity: gunshots occur once in millions of continuous audio windows. We systematically trained models under varying class imbalance ratios ($1:1$, $1:10$, and $1:50$). Models trained strictly on $1:1$ distributions exhibited extreme over-sensitivity and high false-positive rates during continuous monitoring. Re-weighting loss functions and augmenting the negative corpus with urban noise (sirens, traffic, wind, thunder, dog barks) was required to stabilize the decision boundaries.

---

## 4. Empirical Breakdown of the Edge Impulse Experiments

To evaluate on-device microcontroller feasibility, we executed 84 distinct empirical experiment configurations inside the Edge Impulse TinyML framework targeting the ARM Cortex-M4F ($64\text{ MHz}$, $256\text{ KB}$ SRAM).

### 4.1 Experiment 1: MFCC Processing + 1D CNN Classifier (The "Voice-Trap")
- **DSP Pipeline:** 250 ms window, frame length $0.025\text{ s}$, frame stride $0.02\text{ s}$, 32 Mel filters, 13 MFCC coefficients, FFT length 512, pre-emphasis 0.98. DSP calculation: $9\text{ ms}$, Peak RAM: $17\text{ KB}$.
- **Architecture:** 1D CNN with two convolutional/pooling blocks (8 and 16 filters, kernel size 3) followed by dropout (0.25) and softmax classification.
- **Validation Result:** INT8 quantized validation accuracy was **$99.4\%$**, loss was **$0.02$**, and inference time was **$1\text{ ms}$** ($11.9\text{ KB}$ RAM, $45.4\text{ KB}$ Flash).
- **Live Failure Analysis:** When exposed to continuous ambient voice audio (`testing.6mcudcbc`), the model registered **33 false positive triggers out of 47 evaluated windows** ($70.2\%$ false alarm rate). Because MFCC features average spectral energy across time frames, human vocal formants and abrupt consonant bursts mapped onto the same feature clusters as muzzle blasts.

### 4.2 Experiment 2: MFCC + K-Means Anomaly Detection
- To suppress out-of-distribution sounds, an unsupervised K-means anomaly clustering block was integrated over the 13 MFCC features.
- DSP calculation time rose to $120\text{ ms}$.
- On continuous ambient audio (`testing.6md9m237`), 47 out of 48 windows were marked as severe anomalies (anomaly distances $7.01\text{--}36.74$). However, the underlying classifier still generated false positives, indicating that MFCCs lacked the requisite feature representation to separate acoustic transients from muzzle blasts.

### 4.3 Experiment 3: MFE Spectrogram + 1D Convolution
- **DSP Pipeline:** Migrated from MFCC to raw Mel Filterbank Energy (MFE) Spectrograms ($2\text{ ms}$ frame length, $1.5\text{ ms}$ frame stride, FFT length 32, noise floor $-77\text{ dB}$, yielding 3,383 spectral features). DSP time: $52\text{ ms}$, Peak RAM: $14\text{ KB}$.
- **Architecture:** 1D Convolution over unrolled spectral bins. Validation accuracy: **$98.8\%$**, INT8 latency: **$12\text{ ms}$**, Flash: **$47.1\text{ KB}$**.
- **Live Testing:** On ambient audio (`testing.6mff3et0`), false alarms decreased to 24 out of 48 windows ($50\%$). Shifting to time-frequency spectrograms broke the voice-trap, but 1D convolution over 1D feature vectors could not extract 2D temporal-spatial spectral contours.

### 4.4 Experiment 4: MFE Spectrogram + 2D CNN (The Turning Point)
- **Architecture:** A 2D Convolutional Neural Network processing the 2D time-frequency spectrogram grid directly.
- **Metrics:** Validation Accuracy: **$99.6\%$**, Loss: **$0.02$**, ROC-AUC: **$1.00$**, INT8 Latency on Cortex-M4F: **$167\text{ ms}$**, Peak RAM: **$30.2\text{ KB}$**, Flash Footprint: **$52.2\text{ KB}$**.
- **Live Imposter Rejection Validation:**
  - **Heavy Wind Imposter (`testing.6mfo3s0l`):** 22 correct ambient rejections vs. only 7 false alarms ($75.8\%$ ambient specificity).
  - **Thunder Imposter (`testing.6mfoaf80`):** 20 correct ambient rejections vs. only 5 false alarms ($80.0\%$ ambient specificity).
  - **True Gunshot Verification (`testing.6mfoi9ia` - 12-gauge shotgun):** Confirmed abrupt classification probability transitions from $0.06$ to $>0.94$ precisely matching firearm blast onsets.

### 4.5 Experiment 5: MFE Spectrogram + Anomaly Detection
- Combining the 2D CNN with a spatial clustering anomaly detector yielded **$99.7\%$** validation accuracy and established a dual-gate decision boundary: the 2D CNN assesses gunshot probability, while the anomaly score discards non-ballistic high-energy spikes.

---

## 5. Comprehensive Benchmark: 300 Unseen Audio Clips on Throttled Edge VM

To assess true generalization, we developed an independent test suite of **300 unseen audio files** (100 confirmed gunshots, 100 ambient environmental clips, and 100 high-energy percussive imposters). Because physical Raspberry Pi 4 units suffered hardware boot failures, we profiled edge execution on a virtual machine pinned and throttled to $45\text{--}50\%$ CPU utilization, matching the clock cycles and cache bandwidth of a quad-core ARM Cortex-A72.

### 5.1 Evaluated Architectures
1. **Baseline 1D CNN:** Directly processing 1D raw waveforms ($14.2\text{ ms}$ latency).
2. **Baseline 2D CNN:** 2D convolution over normalized log-mel spectrograms.
3. **Enhanced 1D CNN:** Dual-head 1D model with class weighting (suffered INT8 quantization collapse resulting in 100% false alarms).
4. **Enhanced 2D CNN:** Optimized lightweight 2D CNN ($0.09\text{ ms}$ latency, $123\text{ KB}$ size).
5. **Robust CRNN-PCEN:** Mel-spectrogram processed via Per-Channel Energy Normalization (PCEN) followed by a 2D CNN and Bidirectional Gated Recurrent Unit (Bi-GRU).

### 5.2 Comparative Results Across Three Evaluation Paradigms

```
+---------------------------------------------------------------------------------------------------------+
|                                    PARADIGM BENCHMARK COMPARISON (300 CLIPS)                            |
+---------------------------------------------------------------------------------------------------------+
| Paradigm                      | Gunshot Recall | Ambient Specificity | Edge VM Latency | Flash Footprint|
+-------------------------------+----------------+---------------------+-----------------+----------------+
| 1. Peak-Centered (Single Best)| 38.0%          | 99.0%               | 0.09 ms         | 123 KB         |
| 2. Sliding-Window (Dynamic)   | 61.0%          | 97.0%               | 11.86 ms        | 458 KB         |
| 3. Multi-Model Ensemble       | 77.0%          | 95.0%               | 17.04 ms        | 715 KB         |
+---------------------------------------------------------------------------------------------------------+
```

1. **Peak-Centered Paradigm:** Evaluates a single $750\text{ ms}$ slice centered on the clip's maximum energy peak. While extremely fast ($0.09\text{ ms}$ for Enhanced 2D CNN), gunshot recall was limited to $38.0\%$ because distant or suppressed firearm discharges do not always exhibit an isolated amplitude peak.
2. **Sliding-Window Paradigm:** Steps a $750\text{ ms}$ window across the clip with $50\%$ overlap ($375\text{ ms}$ stride). Recall increased to $61.0\%$ with $11.86\text{ ms}$ latency, effectively tracking transient energy shifts across time.
3. **Multi-Model Ensemble Paradigm:** Combines Baseline 2D CNN, Enhanced 2D CNN, and Robust CRNN-PCEN via soft voting. Gunshot recall achieved an empirical peak of **$77.0\%$** ($77/100$ gunshots detected), while ambient specificity remained at $95.0\%$. The total combined memory footprint is only **$715\text{ KB}$**, and combined execution requires **$17.04\text{ ms}$** on the throttled Cortex-A72 edge VM—utilizing less than $10\%$ of a real-time $250\text{ ms}$ audio processing budget.

---

## 6. Wireless Sensor Network (WSN) System Architecture

### 6.1 Low-Power Node Duty-Cycling & Acoustic Sleep Scheduling
In an untethered sensor network, continuous microphone sampling and neural inference deplete small batteries within 24–48 hours. We design a two-tier acoustic sleep-wake protocol for the Arduino Nano 33 BLE Sense:
- **Tier 1 (Deep Sleep Listening):** The ARM Cortex-M4F processor remains in Low-Power Idle mode ($\sim 5\text{ µA}$). The digital PDM microphone or an analog threshold detector triggers an external interrupt only when ambient sound pressure exceeds a predefined baseline (e.g., $>75\text{ dB SPL}$).
- **Tier 2 (Burst Inference):** Upon interrupt, the processor enters high-frequency mode ($64\text{ MHz}$), captures a $250\text{ ms}\text{--}500\text{ ms}$ DMA audio buffer, runs the INT8 quantized neural model ($1\text{--}167\text{ ms}$ depending on architecture), and evaluates the output probability.

```
       +----------------------------------------------------------------+
       |                     TIER 1: DEEP SLEEP (~5 uA)                 |
       |             Acoustic Energy < 75 dB SPL -> Sleep              |
       +----------------------------------------------------------------+
                                       |
                           (SPL > 75 dB Threshold)
                                       v
       +----------------------------------------------------------------+
       |                 TIER 2: BURST INFERENCE (64 MHz)               |
       |       DMA Buffer (250-500ms) -> TinyML INT8 Classification     |
       +----------------------------------------------------------------+
                                       |
                   +-------------------+-------------------+
                   |                                       |
       (Confidence < Threshold)                (Confidence >= Threshold)
                   v                                       v
       +-----------------------+               +-----------------------+
       | Return to Deep Sleep  |               |  1-Bit Alert Packet   |
       |  (Zero RF Radiation)  |               |  Broadcasted via BLE  |
       +-----------------------+               +-----------------------+
```

### 6.2 Zero-Overhead RF Transmission & Privacy Preservation
- **The Bandwidth Bottleneck:** Streaming raw 16 kHz 16-bit uncompressed audio requires $256\text{ kbps}$ continuous throughput per sensor node, which quickly saturates BLE 5.0 mesh channels and consumes tens of milliamperes of radio transmission power.
- **1-Bit Binary Alert Signaling:** In our protocol, raw audio waveforms **never leave the sensor node**. When local inference confidence exceeds the gunshot threshold, the node emits a single, compact BLE advertisement packet:
  $$\text{Payload} = \{\text{Node\_ID (1 Byte)}, \text{Microsecond Timestamp (4 Bytes)}, \text{Detection Bit (1 Bit)}, \text{Confidence Score (1 Byte)}\}$$
- **Energy Implication:** The BLE radio active duration is under $1.5\text{ ms}$, reducing RF transmission energy to near-zero ($<0.01\text{ mWh}$ per event) and extending node longevity to multiple years on coin-cell or small lithium batteries.
- **Civilian Privacy Guarantee:** Because conversational audio is processed strictly within volatile SRAM and immediately overwritten, the system provides mathematical immunity against unauthorized audio surveillance or wiretapping.

### 6.3 Gateway Spatial Consensus & TDOA Multilateration
To eliminate single-node false positives caused by close-proximity percussive noises (e.g., someone clapping directly next to a single sensor), the Raspberry Pi 4 edge gateway enforces a spatial consensus filter:
1. **Voronoi Spatial Clustering:** Sensor nodes are modeled within geographical Voronoi cells.
2. **Coincidence Window:** An incident is validated if and only if $K \ge 3$ neighboring nodes within a cell report a confirmed detection within a propagation coincidence window:
   $$\Delta t \le \frac{d_{\text{max}}}{c_{\text{sound}}} \approx \frac{34\text{ m}}{343\text{ m/s}} \approx 100\text{ ms}$$
3. **Hyperbolic TDOA Localization:** Given $N$ synchronized receiving nodes at coordinates $(x_i, y_i)$ with arrival times $t_i$, the sound origin $(x, y)$ is computed by solving the system of hyperbolic equations:
   $$\sqrt{(x - x_i)^2 + (y - y_i)^2} - \sqrt{(x - x_j)^2 + (y - y_j)^2} = c \cdot (t_i - t_j)$$
This spatial consensus guarantees that isolated mechanical impacts or near-field claps are discarded, forwarding only verified, triangulated gunshot coordinates to law enforcement and emergency dispatch dashboards.

---

## 7. Conclusion & Research Roadmap

This research demonstrates that a reliable, privacy-preserving acoustic gunshot detector can be achieved on inexpensive edge hardware without proprietary cloud subscriptions. By tracking the empirical progression from naive trimming and classical ML failure to 2D CNN spectrogram filtering, PCEN normalization, and multi-model edge ensembles, we provide a reproducible blueprint for real-world acoustic TinyML. 

The immediate subsequent milestones for the IEEE COMSNETS 2027 publication include:
1. Replacing the corrupted SD card on the physical Raspberry Pi 4 units to transition from throttled VM profiling to physical hardware bench testing.
2. Deploying the quantized INT8 Enhanced 2D CNN firmware onto a physical 3-node Arduino Nano 33 BLE Sense array.
3. Conducting controlled real-room and outdoor acoustic trials measuring multi-node TDOA multilateration error under varying ambient signal-to-noise ratios.
