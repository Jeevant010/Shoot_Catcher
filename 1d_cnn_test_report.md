# 1D CNN Evaluation Report (Test Set Results)

This report summarizes the results of the 1D CNN model evaluation, automatically generated from the output of your recent notebook run.

## Model Output Summary

The model successfully trained and evaluated on the held-out test set. The results are extremely strong, indicating that the 1D CNN is highly capable of distinguishing between gunshots and non-gunshots directly from the raw waveform.

| Metric | Score | Explanation |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.9988** | Excellent. The model has a 99.88% chance of ranking a random gunshot higher than a random non-gunshot. |
| **PR-AUC** | **0.9847** | Very high. Shows excellent precision and recall balance even on an imbalanced dataset. |
| **F2-Score** | **0.9633** | Strong performance prioritizing recall (ensuring we don't miss actual gunshots). |

### Dataset Splits
- **Training Set:** 82,802 clips
- **Validation Set:** 8,871 clips
- **Test Set:** 8,871 clips

### Model Architecture
- **Total Parameters:** 92,513 (Very lightweight, suitable for edge devices)

> [!WARNING]
> **Configuration Mismatch Detected**
> According to the output logs (`clip_duration_ms: 250`), the model was trained and evaluated on **250ms** clips, not the **750ms** clips we discussed earlier. 
> 
> If you intended to train on 750ms, please double-check that you updated `TARGET_MS = 750` in the trimmer notebooks and `CLIP_DURATION_MS = 750` in the 1D CNN notebook before running them.
