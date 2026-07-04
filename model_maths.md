# Complete Mathematics of Deep Learning Architectures

This document provides a massive, from-scratch mathematical breakdown of the deep learning architectures used in the `Shoot_Catcher` models. It explicitly details the equations running underneath the Keras/TensorFlow abstractions for both 1D and 2D Convolutional Neural Networks.

---

## 1. 1D Convolutional Neural Networks (1D CNN)

The 1D CNN operates directly on the raw audio waveform, extracting temporal features without prior frequency conversion.

### 1.1 The 1D Convolution Operation
Let the input audio waveform be a 1D vector $\mathbf{x}$ of length $N$ (e.g., $N=16,537$ for $750\text{ms}$). Let the convolutional filter (kernel) be a weight vector $\mathbf{w}$ of length $K$.

The 1D discrete convolution operation, outputting the feature map $\mathbf{y}$, is defined at position $i$ as:
$$y[i] = \sum_{k=0}^{K-1} x[i \cdot S + k] \cdot w[k] + b$$
Where:
- $S$ is the **stride** (how many samples the filter moves forward each step).
- $b$ is the scalar **bias** added to the filter's output.
- The size of the output feature map $M$ is mathematically derived as:
  $$M = \left\lfloor \frac{N - K + 2P}{S} \right\rfloor + 1$$
  *(Where $P$ is the padding size. For "valid" padding, $P=0$.)*

### 1.2 Multi-Channel 1D Convolution
In deep layers, the input is not a single waveform but a matrix $X \in \mathbb{R}^{C_{in} \times N}$ with $C_{in}$ channels. The layer applies $C_{out}$ distinct filters.
For the $c_{out}$-th output channel, the convolution is a sum over all input channels $c_{in}$:
$$y_{c_{out}}[i] = \sum_{c_{in}=1}^{C_{in}} \sum_{k=0}^{K-1} X_{c_{in}}[i \cdot S + k] \cdot W_{c_{out}, c_{in}}[k] + b_{c_{out}}$$

---

## 2. 2D Convolutional Neural Networks (2D CNN)

The 2D CNN requires the 1D audio to be converted into a 2D image-like representation (the Mel-Spectrogram).

### 2.1 Mel-Spectrogram Feature Extraction
**Step 1: STFT**
The Short-Time Fourier Transform (as defined in the trimmer maths) produces a complex spectrum $X(k, m)$. We take the power spectrum:
$$P(k, m) = |X(k, m)|^2$$

**Step 2: Mel Filter Bank**
Human hearing is non-linear. The Mel scale relates physical frequency $f$ to perceived pitch $m$:
$$m(f) = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)$$
We create a set of $F_{mel}$ triangular overlapping filters $H_f(k)$ spaced equally on the Mel scale. The Mel-Spectrogram $S_{mel}$ is the dot product of the power spectrum and the Mel filter bank:
$$S_{mel}(f, m) = \sum_{k} P(k, m) \cdot H_f(k)$$

**Step 3: Log Scaling (Decibels)**
Neural networks prefer stable, bounded inputs. We convert the power to the logarithmic decibel scale:
$$S_{dB}(f, m) = 10 \cdot \log_{10}\left( \frac{S_{mel}(f, m)}{S_{ref}} \right)$$

### 2.2 The 2D Convolution Operation
Let the input Mel-Spectrogram be a matrix $X \in \mathbb{R}^{H \times W}$. Let the 2D filter be a matrix $W \in \mathbb{R}^{K_h \times K_w}$.
The 2D convolution is defined at position $(i, j)$ as:
$$Y[i, j] = \sum_{m=0}^{K_h-1} \sum_{n=0}^{K_w-1} X[i \cdot S_h + m, j \cdot S_w + n] \cdot W[m, n] + b$$
The output dimensions are:
$$H_{out} = \left\lfloor \frac{H - K_h + 2P_h}{S_h} \right\rfloor + 1, \quad W_{out} = \left\lfloor \frac{W - K_w + 2P_w}{S_w} \right\rfloor + 1$$

---

## 3. Activation Functions and Pooling

### 3.1 Non-Linear Activations (ReLU)
Convolutions are strictly linear operations ($Wx + b$). To learn complex patterns, non-linear activation functions are applied element-wise to the feature maps.
The Rectified Linear Unit (ReLU) is defined as:
$$f(x) = \max(0, x)$$
Its derivative (crucial for backpropagation) is:
$$f'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \le 0 \end{cases}$$

### 3.2 Max Pooling
Pooling drastically reduces the spatial dimensionality of the feature maps, granting translational invariance (e.g., it doesn't matter exactly *when* the gunshot happened, just that it happened).
For a pooling window of size $P_h \times P_w$ with stride $S$:
$$Y[i, j] = \max_{m \in [0, P_h-1], n \in [0, P_w-1]} X[i \cdot S + m, j \cdot S + n]$$

---

## 4. Fully Connected (Dense) Layers

After several convolution and pooling layers, the resulting 3D tensor is flattened into a 1D vector $\mathbf{x} \in \mathbb{R}^{D}$.
A Dense layer with $U$ units computes a full matrix multiplication with weight matrix $W \in \mathbb{R}^{U \times D}$ and bias vector $\mathbf{b} \in \mathbb{R}^{U}$:
$$\mathbf{z} = W \mathbf{x} + \mathbf{b}$$
$$z_j = \sum_{i=1}^{D} W_{j, i} \cdot x_i + b_j$$

---

## 5. Network Output and Loss Function

### 5.1 Output Activation (Sigmoid)
For binary classification (Gunshot = 1, Non-Gunshot = 0), the final Dense layer has a single neuron. Its output $z$ is passed through the Logistic Sigmoid function to map it to a probability $p \in (0, 1)$:
$$p = \sigma(z) = \frac{1}{1 + e^{-z}}$$
The derivative is beautifully simple:
$$\sigma'(z) = p \cdot (1 - p)$$

### 5.2 Binary Cross-Entropy (BCE) Loss
To measure how "wrong" the network's prediction $p$ is compared to the true label $y \in \{0, 1\}$, we use BCE Loss:
$$L(y, p) = - \left[ y \cdot \log(p) + (1 - y) \cdot \log(1 - p) \right]$$
For a batch of $B$ audio clips, the total cost $J$ is the average loss:
$$J = \frac{1}{B} \sum_{i=1}^{B} L(y_i, p_i)$$

---

## 6. Optimization and Backpropagation

The neural network learns by adjusting its weights $W$ to minimize the cost $J$.

### 6.1 Gradient Descent and the Chain Rule
We must calculate the partial derivative of the loss with respect to every weight in the network: $\frac{\partial J}{\partial W}$.
Using the chain rule of calculus, starting from the final Sigmoid layer:
$$\frac{\partial J}{\partial z} = \frac{\partial J}{\partial p} \cdot \frac{\partial p}{\partial z}$$

For BCE Loss and Sigmoid activation, this simplifies elegantly to:
$$\frac{\partial J}{\partial z} = p - y$$

The gradient with respect to the weights $W_{final}$ of the last layer is:
$$\frac{\partial J}{\partial W_{final}} = \frac{\partial J}{\partial z} \cdot \frac{\partial z}{\partial W_{final}} = (p - y) \cdot \mathbf{x}^T$$

This error signal $\frac{\partial J}{\partial z}$ is propagated backward through the dense layers, pooling layers (where the gradient only flows through the `argmax` index), and convolutional layers, multiplying by the local derivative at every step.

### 6.2 The Adam Optimizer
Instead of standard Gradient Descent ($W = W - \alpha \frac{\partial J}{\partial W}$), the model uses Adam (Adaptive Moment Estimation), which tracks the exponentially decaying average of past gradients ($m_t$) and squared gradients ($v_t$):

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(First Moment / Mean)}$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(Second Moment / Variance)}$$
*(where $g_t = \frac{\partial J}{\partial W}$ at iteration $t$)*

Bias correction is applied:
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Finally, the weights are updated using the learning rate $\alpha$:
$$W_{t} = W_{t-1} - \frac{\alpha \cdot \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$
This sophisticated math allows the network to dynamically adjust the learning rate for every single parameter, converging significantly faster on raw audio data.
