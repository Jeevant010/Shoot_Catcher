# 2D CNN Investigation & Fix Plan

## The "Very Bad" 2D CNN Behavior Explained

I deeply analyzed the training logs for your 2D CNN and found out exactly why it completely failed to detect any gunshots (F2 Score = 0.0, False Negatives = 679).

During training, the model's predictions wildly flipped between predicting **100% gunshots** (Epoch 1, 2, 4) and **100% background** (Epoch 3, 5, and all the way to 50). By Epoch 15, the model completely collapsed and just lazily classified every single audio clip as "Background". 

### The Root Cause: Unnormalized Spectrograms
Neural networks expect inputs to be small numbers, typically between `0.0` and `1.0` or `-1.0` and `1.0`. 
In the 1D CNN, we normalized the raw audio waveform to be between `-1.0` and `1.0`. 

However, in the 2D CNN, we convert the audio to a Mel-Spectrogram in **Decibels (dB)** using `librosa.power_to_db()`. This function outputs values ranging from **`-80.0` to `0.0`**. 

Because we fed these massive negative numbers (`-80.0`) directly into the CNN without normalizing them, it triggered a phenomenon called the **"Dying ReLU" problem**:
1. The huge negative inputs hit the first Convolutional Layer.
2. The gradients "exploded", causing the neural network's weights to swing violently.
3. The `ReLU` activation functions immediately clamped all outputs to `0`.
4. The network got "brain damage" and permanently got stuck outputting `0` (Background) for everything.

---

## User Review Required

> [!IMPORTANT]
> The fix is extremely simple but requires modifying both 2D CNN notebooks. We just need to normalize the Decibel values from `[-80.0, 0.0]` to `[0.0, 1.0]`. 
> Do you approve the plan to patch these notebooks? Once patched, you will need to re-run them.

## Proposed Changes

### [Component Name: 2D CNN Models]

#### [MODIFY] [2d_cnn_gunshot_detector.ipynb](file:///c:/order/Desktop/Gun/Shoot_Catcher/02_2D_CNN/2d_cnn_gunshot_detector.ipynb)
- Modify the `waveform_to_mel` function to mathematically shift the Decibel values: `S_normalized = (S_dB + 80.0) / 80.0`

#### [MODIFY] [enhanced_2d_cnn.ipynb](file:///c:/order/Desktop/Gun/Shoot_Catcher/Enhanced_Models/02_Enhanced_2D_CNN/enhanced_2d_cnn.ipynb)
- Modify the `waveform_to_mel` function in the exact same way to protect the enhanced architecture.

## Verification Plan

### Automated Tests
- N/A

### Manual Verification
- After I apply the patch, you will re-run the 2D CNN notebooks. The `val_accuracy` should no longer jump wildly, and the final F2-score will be vastly improved.
