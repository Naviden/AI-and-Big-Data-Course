# Neural Networks

### Objectives:
1. Understand the key concepts and applications of Neural Networks.

---

### Definition:
Neural Networks (NN) are a class of machine learning models inspired by the structure and function of the human brain. They consist of layers of interconnected nodes (neurons) that process input data, learn patterns, and make predictions. Neural networks are widely used for tasks like image recognition, natural language processing, and regression.

### Key Concepts:
- **Neuron:** The basic unit of a neural network that receives inputs, applies a weighted sum, adds a bias, and passes the result through an activation function.
- **Layers:** Neural networks are composed of:
  - **Input Layer:** Receives the raw input features.
  - **Hidden Layers:** Process the data using weights and biases.
  - **Output Layer:** Produces the final predictions.
- **Activation Function:** A function applied to the output of a neuron to introduce non-linearity (e.g., ReLU, sigmoid, or tanh).
- **Backpropagation:** An algorithm used to update weights by minimizing the error between predicted and actual outputs through gradient descent.

### Mathematical Formulation:
In a single-layer neural network:

1. Compute the weighted sum for a neuron:

$$
z = \sum_{i=1}^{n} w_i x_i + b
$$

   Where:
   - $x_i$ are input features,
   - $w_i$ are weights,
   - $b$ is the bias.

2. Apply an activation function $f(z)$:

$$
a = f(z)
$$

3. Use backpropagation to minimize the loss $L(y, \hat{y})$:

$$
\text{Update weights: } w_i \leftarrow w_i - \eta \frac{\partial L}{\partial w_i}
$$

   Where:
   - $\eta$ is the learning rate,
   - $L$ is the loss function (e.g., mean squared error or cross-entropy).

### Numerical Example of a Neural Network

#### Problem:
We have a simple dataset with one feature $x$ and a binary target $y$:

| $x$ | $y$ |
|------|------|
| 1    | 0    |
| 2    | 0    |
| 3    | 1    |

#### Steps:

1. **Initialization**:
   - Weights: $w = 0.5$
   - Bias: $b = 0$
   - Learning rate: $\eta = 0.1$

2. **Forward Pass**:
   For each input $x$, compute:

$$
z = w \cdot x + b
$$

   Apply the sigmoid activation function:

$$
a = \frac{1}{1 + e^{-z}}
$$

   Predict the output: $\hat{y} = a$.

   Example for $x = 1$:

$$
z = 0.5 \cdot 1 + 0 = 0.5, \quad a = \frac{1}{1 + e^{-0.5}} \approx 0.62
$$

3. **Compute Loss**:
   Use the binary cross-entropy loss:

$$
L = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$

   For $x = 1$, $y = 0$:

$$
L = -[0 \cdot \log(0.62) + (1 - 0) \cdot \log(1 - 0.62)] \approx 0.48
$$

4. **Backpropagation**:
   Update the weights and bias:

$$
w \leftarrow w - \eta \frac{\partial L}{\partial w}, \quad b \leftarrow b - \eta \frac{\partial L}{\partial b}
$$

   Gradients for $x = 1$:

$$
\frac{\partial L}{\partial w} = (a - y) \cdot x = (0.62 - 0) \cdot 1 = 0.62
$$

$$
w = 0.5 - 0.1 \cdot 0.62 = 0.438
$$

5. **Repeat**:
   Continue for all data points and epochs until the loss converges.

#### Key Takeaway:
Neural networks learn by iteratively adjusting weights and biases to minimize the error (loss). The combination of multiple layers and non-linear activation functions enables them to model complex patterns in data.

---

### Recommended Reading:
- **[Neural Networks from scratch in Python](https://nnfs.io/order)**
- **["Pattern Recognition and Machine Learning" by Christopher M. Bishop](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)**
- **["An Introduction to Statistical Learning" by Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani](https://www.stat.berkeley.edu/users/rabbee/s154/ISLR_First_Printing.pdf)**
- **["Alice’s Adventures in a differentiable wonderland: A primer on designing neural networks (Volume I)" by Simone Scardapane](https://www.amazon.it/dp/B0D9QHS5NG?ref=ppx_yo2ov_dt_b_fed_asin_title)**
