# Advanced Machine Learning Models

### Topics Covered:
- **Support Vector Machines (SVM)**
- **Gradient Boosting Machines (GBM)**
- **Recurrent Neural Networks (RNN)**
- **Generative Adversarial Networks (GANs)**

### Objectives:
1. Understand the key concepts and applications of advanced machine learning models.
2. Learn how to implement and apply models like SVM, GBM, RNN, and GANs to complex tasks.
3. Recognize the strengths and challenges of each model, particularly in specialized use cases.

---

## 1. Support Vector Machines (SVM)

### Definition:
Support Vector Machines (SVM) are supervised learning models used for classification and regression tasks. They are particularly effective in high-dimensional spaces and cases where the number of dimensions exceeds the number of samples.

### Key Concepts:
- **Hyperplane:** A decision boundary that separates different classes in the feature space. The optimal hyperplane is the one that maximizes the margin between the classes.
- **Margin:** The distance between the hyperplane and the closest data points (support vectors) from each class.
- **Support Vectors:** The data points closest to the hyperplane that influence its position and orientation. These points are critical for determining the decision boundary.
- **Kernel:** A mathematical function that allows SVM to solve non-linear classification problems by mapping the input data into a higher-dimensional space. This transformation enables SVM to find a linear boundary in the transformed space, even if the data is not linearly separable in the original space. 

![SVM Example](https://scikit-learn.org/stable/_images/sphx_glr_plot_iris_svc_001.png)

Example of classification with SVM classifiers.
_[Image Source](https://scikit-learn.org/stable/modules/svm.html)_

### Mathematical Formulation:
Given a set of training data $(x_i, y_i)$ where $x_i$ represents the feature vector and $y_i \in \{-1, 1\}$ represents the class label, SVM aims to find the hyperplane defined by:

$$ w \cdot x + b = 0 $$

Where $w$ is the weight vector and $b$ is the bias. The objective is to maximize the margin:

$$ \text{Maximize } \frac{2}{\|w\|} $$

Subject to the constraint:

$$ y_i(w \cdot x_i + b) \geq 1 \text{ for all } i $$

### Types of Kernels:
SVM uses **kernels** to handle both linear and non-linear data. Kernels compute the similarity between two data points in the transformed space using the **kernel trick**, avoiding explicit computation of high-dimensional transformations.

Common kernel types include:
1. **Linear Kernel:** Suitable for linearly separable data.
   - Formula: $$ K(x, y) = x \cdot y $$

2. **Polynomial Kernel:** Captures polynomial relationships between features.
   - Formula:
     $$
     K(x, y) = (x \cdot y + c)^d
     $$
   - Parameters: \( c \) (constant) and \( d \) (degree of the polynomial).

3. **Radial Basis Function (RBF) Kernel:** Handles non-linear data effectively.
   - Formula:
     $$
     K(x, y) = \exp(-\gamma \|x - y\|^2)
     $$
   - Parameter: \( \gamma \) controls the influence of each training example.

4. **Sigmoid Kernel:** Inspired by neural networks.
   - Formula:
     $$
     K(x, y) = \tanh(\alpha x \cdot y + c)
     $$
   - Parameters: \( \alpha \) and \( c \).

### Advantages of SVM:
- Effective in high-dimensional spaces.
- Robust to overfitting, especially in cases where the number of dimensions is greater than the number of samples.
- Versatile due to the use of different kernel functions.


### Use Case Example:
- **Image Classification:** SVMs are often used in tasks like handwriting recognition and face detection, where they can efficiently handle high-dimensional data.


Take a look at this [short video series](https://www.youtube.com/watch?v=efR1C6CvhmE&t=743s) by StatQuest for more details.

---

## 2. Gradient Boosting Machines (GBM)

### Definition:
Gradient Boosting Machines (GBM) are ensemble learning techniques that build models sequentially, where each new model corrects the errors of the previous ones. This approach is particularly effective for structured data and is commonly used in competitions and industry.

### Key Concepts:
- **Boosting:** An ensemble method that combines the outputs of multiple weak learners (typically decision trees) to form a strong predictive model.
- **Learning Rate:** A hyperparameter that controls the contribution of each weak learner to the final model.
- **Loss Function:** The function minimized during the boosting process, which can vary depending on the task (e.g., mean squared error for regression).

### Mathematical Formulation:
The model is built in a stage-wise manner, with each new model $h_m(x)$ added to the previous ensemble to correct its errors:

$$ F_{m}(x) = F_{m-1}(x) + \alpha h_m(x) $$

Where:
- $F_{m}(x)$ is the model at stage $m$,
- $\alpha$ is the learning rate,
- $h_m(x)$ is the new model fitted to the residual errors of $F_{m-1}(x)$.

### Use Case Example:
- **Predictive Modeling:** GBM is widely used in tasks like credit scoring, where accurate predictions from structured data are crucial.

Take a look at this [short video series](https://www.youtube.com/watch?v=3CC4N4z3GJc) by StatQuest for more details.

---

## 3. Recurrent Neural Networks (RNN)

### Definition:
Recurrent Neural Networks (RNN) are a class of neural networks designed for sequential data. Unlike feedforward neural networks, RNNs have connections that form cycles, allowing them to maintain a memory of previous inputs, making them ideal for tasks like time series analysis and natural language processing.

![RNN](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Recurrent_neural_network_unfold.svg/2880px-Recurrent_neural_network_unfold.svg.png)
Anatomy of an RNN architecture.
_[Image Source](https://en.wikipedia.org/wiki/Recurrent_neural_network)_

### Key Concepts:
- **Hidden State:** The memory of the network that captures information from previous time steps.
- **Long Short-Term Memory (LSTM):** A variant of RNNs designed to overcome the problem of vanishing gradients, allowing the network to learn long-term dependencies.
- **Gated Recurrent Unit (GRU):** A simplified version of LSTM that uses fewer parameters and is computationally more efficient.

### Mathematical Formulation:
For a standard RNN, the hidden state $h_t$ at time step $t$ is computed as:

$$ h_t = g(W_h \cdot h_{t-1} + W_x \cdot x_t + b) $$

Where:
- $h_{t-1}$ is the hidden state from the previous time step,
- $x_t$ is the input at time $t$,
- $W_h$ and $W_x$ are weight matrices,
- $b$ is the bias vector,
- $g$ is the activation function (e.g., tanh or ReLU).

### Use Case:
- **Language Modeling:** RNNs, particularly LSTM and GRU, are used in tasks like machine translation, where understanding the context across a sequence of words is essential.

---

## 4. Generative Adversarial Networks (GANs)

### Definition:
Generative Adversarial Networks (GANs) are a class of neural networks used for unsupervised learning tasks. GANs consist of two networks, a generator and a discriminator, that are trained simultaneously. The generator creates fake data, while the discriminator tries to distinguish between real and fake data.

### Key Concepts:
- **Generator:** A neural network that generates synthetic data samples by mapping random noise to data space.
- **Discriminator:** A neural network that evaluates whether a given sample is real (from the training set) or fake (generated).
- **Adversarial Training:** The process in which the generator and discriminator are trained together in a zero-sum game until the generator produces data indistinguishable from real data.

### Mathematical Formulation:
The generator $G(z)$ maps noise $z$ from a latent space to the data space, while the discriminator $D(x)$ outputs the probability that a given sample $x$ is real. The objective function for GANs is:

$$ \min_G \max_D \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))] $$

Where:
- $p_{\text{data}}(x)$ is the distribution of the real data,
- $p_z(z)$ is the distribution of the input noise.

### Use Case:
- **Image Generation:** GANs are widely used for generating realistic images, such as creating high-resolution photos from low-resolution inputs or generating artwork.

---

### Recommended Reading:
- **["Pattern Recognition and Machine Learning" by Christopher M. Bishop](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)**
- **["An Introduction to Statistical Learning" by Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani](https://www.stat.berkeley.edu/users/rabbee/s154/ISLR_First_Printing.pdf)**
- **["Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville](https://www.deeplearningbook.org/)**

### Further Exploration:
- **SVM Tutorial:** Learn how to implement Support Vector Machines [here](https://scikit-learn.org/stable/modules/svm.html).
- **Implementing RNNs:** Explore a tutorial on Recurrent Neural Networks [here](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html).