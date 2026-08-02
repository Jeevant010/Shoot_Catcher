# 🧠 Part 2: Neural Network & CRNN Layer-by-Layer Mathematics

This document details the exact tensor transformations, matrix multiplications, and recurrent gating equations inside the CRNN (Convolutional Recurrent Neural Network) architecture.

---

## 1. Input Tensor Geometry

The PCEN feature matrix is formatted as a 3D tensor:
$$\mathbf{X}_{\text{input}} \in \mathbb{R}^{B \times F \times T \times C}$$

- $B$: Batch size (e.g., 64).
- $F = 64$: Mel frequency channels.
- $T = 130$: Time frames ($\approx 750\text{ms}$).
- $C = 1$: Single channel input.

---

## 2. 2D Convolutional Layers (Spatial Feature Extraction)

A 2D Convolution layer slides $K$ kernels of size $(3 \times 3)$ over the PCEN feature map.

### 2.1 2D Discrete Cross-Correlation Formula
$$O[i, j, k] = \sum_{m=-1}^{1} \sum_{n=-1}^{1} \sum_{c=1}^{C_{\text{in}}} X[i+m, j+n, c] \cdot W_k[m, n, c] + b_k$$

Where:
- $W_k \in \mathbb{R}^{3 \times 3 \times C_{\text{in}}}$ is the $k$-th filter weight kernel.
- $b_k \in \mathbb{R}$ is the bias scalar.
- Activation: $\text{ReLU}(z) = \max(0, z)$.

### 2.2 Conv2D Tensor Dimensions
1. **Input**: $(64 \times 130 \times 1)$
2. **Conv2D Block 1** ($32$ filters, $3 \times 3$) + **MaxPool** ($2 \times 2$):
   $$\text{Output}_1 \in \mathbb{R}^{32 \times 65 \times 32}$$
3. **Conv2D Block 2** ($64$ filters, $3 \times 3$) + **MaxPool** ($2 \times 2$):
   $$\text{Output}_2 \in \mathbb{R}^{16 \times 32 \times 64}$$

---

## 3. Batch Normalization Mathematics

Batch Normalization stabilizes gradient flow by forcing internal activations to have zero mean and unit variance per mini-batch.

### 3.1 Mini-Batch Mean & Variance
$$\mu_B = \frac{1}{m} \sum_{i=1}^m x_i, \quad \sigma_B^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_B)^2$$

### 3.2 Normalization & Affine Transformation
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

$$y_i = \gamma \hat{x}_i + \beta$$

Where $\gamma$ (scale) and $\beta$ (shift) are learnable parameters.

---

## 4. Sequence Reshaping (Spatial to Temporal Mapping)

To prepare 2D spatial features for 1D recurrent sequence processing, frequency channels and filter channels are collapsed into a unified feature vector per time step:

$$\mathbf{X}_{\text{seq}} = \text{Reshape}\left( \text{Output}_2 \right) \in \mathbb{R}^{32 \times (16 \times 64)} = \mathbb{R}^{32 \times 1024}$$

- Time Steps: $T' = 32$.
- Feature Vector Dimension: $D = 1024$.

---

## 5. Bidirectional Gated Recurrent Unit (Bi-GRU) Math

A GRU processes time sequences step-by-step using 3 gating equations.

### 5.1 Update Gate $z_t$
Decides how much previous memory $h_{t-1}$ to carry forward:
$$\mathbf{z}_t = \sigma\left( \mathbf{W}_z \mathbf{x}_t + \mathbf{U}_z \mathbf{h}_{t-1} + \mathbf{b}_z \right)$$

### 5.2 Reset Gate $r_t$
Decides how much past memory to forget when computing new information:
$$\mathbf{r}_t = \sigma\left( \mathbf{W}_r \mathbf{x}_t + \mathbf{U}_r \mathbf{h}_{t-1} + \mathbf{b}_r \right)$$

### 5.3 Candidate Hidden State $\tilde{h}_t$
$$\mathbf{\tilde{h}}_t = \tanh\left( \mathbf{W}_h \mathbf{x}_t + \mathbf{U}_h (\mathbf{r}_t \odot \mathbf{h}_{t-1}) + \mathbf{b}_h \right)$$

### 5.4 Final Hidden State $h_t$
$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \mathbf{\tilde{h}}_t$$

### 5.5 Bidirectional Concatenation
The GRU runs forward ($\overrightarrow{\mathbf{h}}_t$) and backward ($\overleftarrow{\mathbf{h}}_t$) across time:
$$\mathbf{h}_t^{\text{bi}} = \left[ \overrightarrow{\mathbf{h}}_t \, ; \, \overleftarrow{\mathbf{h}}_t \right] \in \mathbb{R}^{128}$$

Forward GRU tracks **muzzle blast onset**; Backward GRU tracks **reverberation decay tail**.

---

## 6. Classification Head & Sigmoid Math

### 6.1 Global Average Pooling 1D
Averages recurrent states over all 32 time steps:
$$\mathbf{h}_{\text{pooled}} = \frac{1}{32} \sum_{t=1}^{32} \mathbf{h}_t^{\text{bi}} \in \mathbb{R}^{128}$$

### 6.2 Dense Layer & Sigmoid Activation
$$z = \mathbf{w}^T \mathbf{h}_{\text{dense}} + b$$

$$P(\text{Gunshot}) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

- If $z \to +\infty$, $\sigma(z) \to 1.0$ (High confidence Gunshot).
- If $z \to -\infty$, $\sigma(z) \to 0.0$ (Non-Gunshot / Ambient).
