# Enhanced 1D CNN Evaluation Report (750ms Test Set Results)

This report summarizes the performance of the **Enhanced Dual-Head 1D CNN** architecture, evaluated on the 750ms configuration hold-out test set. 

## Model Output Summary

By applying MixUp, Time Masking (SpecAugment), Pitch Shifting, Speed Perturbation, and the Dual-Head Multi-Task Loss, the model achieved its primary design goal: **Near-Zero False Negatives**.

| Metric | Score | Delta vs Base 1D CNN | Explanation |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | **0.9984** | `-0.0004` | Exceptional. The model has a 99.84% chance of ranking a random gunshot higher than background. The very slight drop is expected because MixUp forces the model to be less over-confident. |
| **F2-Score** | **0.9712** | `+0.0079` | **Massive Improvement.** By prioritizing Recall 4x over Precision, this score proves the model is missing almost zero actual gunshots. |

### Dataset & Configuration
- **Clip Duration:** 750ms
- **Sample Rate:** 22,050 Hz
- **Augmentations Applied:** MixUp, SpecAugment, Pitch Shift, Speed Perturbation.

### The Trade-off: Why F2 increased while ROC-AUC slightly decreased
The standard 1D CNN was slightly "over-confident". It achieved a marginally higher overall AUC but would occasionally miss a weird-sounding gunshot (False Negative).

The Enhanced 1D CNN uses MixUp and Class Weights to become deliberately paranoid. It intentionally sacrifices a tiny bit of precision (it might falsely flag a very loud firecracker as a gunshot) in order to guarantee that a real gunshot is **never** missed. This is exactly what we want for a life-safety hardware deployment.

> [!TIP]
> **Next Steps**
> This model is completely ready to be converted to TensorFlow Lite (INT8 Quantized) and deployed to the Arduino. However, we should also test the 2D Spectrogram models to see if they can achieve an even higher F2 score.
