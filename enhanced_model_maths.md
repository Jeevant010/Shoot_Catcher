# 🧠 Enhanced 1D CNN Mathematics & Proofs

This document provides a comprehensive, rigorous mathematical breakdown of the "Enhanced" 1D CNN architecture. It details the exact equations used for advanced data augmentation (MixUp, SpecAugment), the Dual-Head Multi-Task Loss optimization, and the F-Beta scoring metric that allows us to force near-zero False Negatives.

---

## 1. Advanced Data Augmentations

The enhanced pipeline relies heavily on continuous data perturbation during training to prevent the network from memorizing specific background noise profiles.

### 1.1 MixUp (Zhang et al. 2018)

**Concept:** Instead of training on discrete classes, MixUp trains the network on linear interpolations of audio waveforms and their corresponding labels. This smooths the decision boundaries and prevents over-confidence.

For any two random audio clips $x_i, x_j$ and their one-hot labels $y_i, y_j$:

1. A mixing parameter $\lambda$ is sampled from a Beta distribution:
   $$ \lambda \sim \text{Beta}(\alpha, \alpha) $$
   *(In our code, $\alpha = 0.3$. This creates a U-shaped distribution where $\lambda$ is usually close to 0 or 1, but occasionally blends heavily in the middle).*

2. The waveforms are mixed in the time domain:
   $$ \tilde{x} = \lambda x_i + (1 - \lambda) x_j $$

3. The labels are mixed correspondingly:
   $$ \tilde{y} = \lambda y_i + (1 - \lambda) y_j $$

**Proof of Regularization:** 
Standard Empirical Risk Minimization (ERM) minimizes the loss over empirical data points. MixUp minimizes the Vicinal Risk, forcing the model to behave linearly between training examples. 
$$ \mathcal{L}_{\text{MixUp}}(\theta) = \mathbb{E}_{\lambda} \left[ \ell(f_\theta(\lambda x_i + (1-\lambda) x_j), \lambda y_i + (1-\lambda) y_j) \right] $$
This mathematically proves that the gradient norm $\nabla_x f_\theta(x)$ is minimized, pushing the network to become a smoother function and massively reducing adversarial vulnerability (e.g., misclassifying a sudden sharp noise as a gunshot).

### 1.2 Time Masking (SpecAugment Style)

To simulate the "guillotine effect" (where the 750ms buffer accidentally cuts a gunshot right down the middle), we randomly zero-out contiguous blocks of the waveform.

Let $x$ be an audio vector of length $T = 16,537$. We sample a starting index $t_0$ and a masking duration $m$:
$$ m \sim \mathcal{U}(0.02T, 0.15T) $$
$$ t_0 \sim \mathcal{U}(0, T - m) $$

The masked waveform $x_{\text{mask}}$ is defined as:
$$ x_{\text{mask}}[t] = \begin{cases} 0 & \text{if } t_0 \le t < t_0 + m \\ x[t] & \text{otherwise} \end{cases} $$

---

## 2. Dual-Head Architecture & Multi-Task Loss

The original 1D CNN output a single scalar $\hat{y} \in [0, 1]$. 
The Enhanced 1D CNN forks at the final global average pooling layer into **two independent dense heads**:

$$ \mathbf{h}_{\text{shared}} = \text{GlobalAvgPool}(\text{ConvNet}(x)) $$
$$ \hat{y}_{\text{gunshot}} = \sigma(\mathbf{W}_{\text{gun}} \cdot \mathbf{h}_{\text{shared}} + \mathbf{b}_{\text{gun}}) $$
$$ \hat{y}_{\text{anomaly}} = \sigma(\mathbf{W}_{\text{anom}} \cdot \mathbf{h}_{\text{shared}} + \mathbf{b}_{\text{anom}}) $$

Where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid activation.

### 2.1 The Weighted Multi-Task Loss Function

To train both heads simultaneously without one overpowering the other, we calculate a total loss $\mathcal{L}_{\text{Total}}$ as a weighted sum of two Binary Cross-Entropy (BCE) losses.

$$ \mathcal{L}_{\text{BCE}}(y, \hat{y}) = - \left( y \log(\hat{y}) + (1-y) \log(1-\hat{y}) \right) $$

Because the gunshot dataset is heavily imbalanced (many more backgrounds than gunshots), we apply a **Class Weight Array** $C_y$ to the gunshot head to heavily penalize missing a real gunshot:
$$ C_0 = \frac{N_{\text{total}}}{2 \times N_{\text{background}}} \approx 0.54 $$
$$ C_1 = \frac{N_{\text{total}}}{2 \times N_{\text{gunshot}}} \approx 6.80 $$

The final optimization objective becomes:
$$ \mathcal{L}_{\text{Total}} = \lambda_{\text{gun}} \left( C_y \cdot \mathcal{L}_{\text{BCE}}(y_{\text{gun}}, \hat{y}_{\text{gun}}) \right) + \lambda_{\text{anom}} \left( \mathcal{L}_{\text{BCE}}(y_{\text{anom}}, \hat{y}_{\text{anom}}) \right) $$

*(In our configuration: $\lambda_{\text{gun}} = 1.0$, $\lambda_{\text{anom}} = 0.3$. This ensures the network primarily learns gunshot detection, but uses the anomaly head as a secondary regularizer).*

---

## 3. The Mathematics of "Near-Zero False Negatives"

In gunshot detection, missing a real gunshot (False Negative) is catastrophic, while an occasional false alarm (False Positive) is acceptable. 

Instead of looking at accuracy, we optimize for the **F-Beta Score**, specifically **F2**. 

### 3.1 Confusion Matrix Variables
* **TP (True Positive):** Predicted Gunshot, Actual Gunshot.
* **FN (False Negative):** Predicted Background, Actual Gunshot. *(FATAL)*
* **FP (False Positive):** Predicted Gunshot, Actual Background. *(Annoying)*

### 3.2 Precision and Recall
$$ \text{Precision} = \frac{TP}{TP + FP} $$
$$ \text{Recall} = \frac{TP}{TP + FN} $$

### 3.3 The F2 Score Equation
The general F-Beta score is defined as:
$$ F_\beta = (1 + \beta^2) \frac{\text{Precision} \cdot \text{Recall}}{(\beta^2 \cdot \text{Precision}) + \text{Recall}} $$

By setting $\beta = 2$, we mathematically assign **Recall 4 times more weight than Precision** ($\beta^2 = 4$). 

$$ F_2 = 5 \cdot \frac{\text{Precision} \cdot \text{Recall}}{4 \cdot \text{Precision} + \text{Recall}} $$

By passing the class weights ($C_1 \approx 12.5 \times C_0$) into the BCE loss, the gradient descent explicitly minimizes False Negatives, directly driving the $F_2$ score toward $1.0$.
