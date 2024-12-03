# Advanced Machine Learning Models

### Topics Covered:
- **Support Vector Machines (SVM)**
- **Gradient Boosting Machines (GBM)**


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


### Types of Kernels:
SVM uses **kernels** to handle both linear and non-linear data. Kernels compute the similarity between two data points in the transformed space using the **kernel trick**, avoiding explicit computation of high-dimensional transformations.

Common kernel types include:
1. **Linear Kernel:** Suitable for linearly separable data. It works directly with the original feature space.
2. **Polynomial Kernel:** Captures polynomial relationships between features, allowing SVM to model complex patterns.
3. **Radial Basis Function (RBF) Kernel:** A widely used kernel that handles non-linear data effectively by focusing on the distance between data points.
4. **Sigmoid Kernel:** Inspired by neural networks, it models relationships in data using a sigmoidal function.

![Kernel Example](https://mldemystified.com/posts/basics-of-ml/support-vector-machines/images/kernel-svm-example.png#center)

[Image Source](https://mldemystified.com/)


- The **dotted lines** in each plot represent the **margins** around the hyperplane, which are determined by the support vectors. These margins indicate the regions where the classification confidence decreases. 

    - In the **Linear Kernel** plot, the margins are straight because the separation is linear.
    - In the **Polynomial**, **RBF**, and **Sigmoid Kernel** plots, the margins are curved, showcasing the more complex decision boundaries created by these kernels to handle non-linear data.

- The **solid black line** in each plot represents the main decision boundary (hyperplane) that separates the classes.



### Mathematical Formulation:
Given a set of training data $(x_i, y_i)$ where $x_i$ represents the feature vector and $y_i \in \{-1, 1\}$ represents the class label, SVM aims to find the hyperplane defined by:

$$ w \cdot x + b = 0 $$

Where $w$ is the weight vector and $b$ is the bias. The objective is to maximize the margin:

$$ \text{Maximize } \frac{2}{\|w\|} $$

Subject to the constraint:

$$ y_i(w \cdot x_i + b) \geq 1 \text{ for all } i $$

Remember that this mathematical formulation is not specific to any particular kernel. It represents the general optimization objective of SVM, applicable to both linear and non-linear data.

- For **linear SVM**, the formulation applies directly in the original feature space, as the data can be separated by a straight hyperplane.
- For **non-linear SVM**, kernels are used to map the data into a higher-dimensional space where it becomes linearly separable. The optimization still follows the same formulation but operates in the transformed feature space defined by the kernel.

The choice of kernel affects how the data is transformed, but the underlying mathematical objective and constraints remain consistent across all kernel types.


#### Further Learning Resources

- Watch this [short video series](https://www.youtube.com/watch?v=efR1C6CvhmE&t=743s) by StatQuest for more details about the math behind SVM.

- Take a look at [this notebook](https://github.com/Naviden/AI-and-Big-Data-Course/blob/main/Python%20Code/Algorithms/An%20Intro%20to%20SVM.ipynb) in which we see SVM in action

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

### Numerical Example of Gradient Boosting

To clarify the concept of Gradient Boosting, let's consider a simple regression example:

#### Problem:
We have a dataset with one feature $x$ and a target value $y$:

| $x$ | $y$ |
|------|------|
| 1    | 3    |
| 2    | 6    |
| 3    | 9    |

Our goal is to predict $y$ using Gradient Boosting.

#### Steps:

1. **Initialization**:
   Start with an initial model $F_0(x)$, which is usually the mean of the target values for regression:

$$
   F_0(x) = \text{mean}(y) = \frac{3 + 6 + 9}{3} = 6
$$

2. **Calculate Residuals**:
   Compute the residuals (errors) between the actual values $y$ and the predictions from $F_0(x)$:

$$
   r_i = y_i - F_0(x)
$$

   | $x$ | $y$ | $F_0(x)$ | Residual $(r = y - F_0(x))$ |
   |------|------|------------|-----------------------------|
   | 1    | 3    | 6          | -3                          |
   | 2    | 6    | 6          | 0                           |
   | 3    | 9    | 6          | 3                           |

3. **Fit a Weak Learner**:
   Fit a simple model $h_1(x)$ (e.g., a decision tree stump) to predict the residuals:
   
_(A decision tree stump is a very simple decision tree with only one split (a tree of depth 1). It is the simplest possible decision tree, making it a common choice for a weak learner in ensemble methods like Gradient Boosting or AdaBoost.)_

$$ 
   h_1(x) = \begin{cases} 
   -3 & \text{if } x = 1 \\
   0 & \text{if } x = 2 \\
   3 & \text{if } x = 3
   \end{cases}
$$

4. **Update the Model**:
   Add the weak learner $h_1(x)$ to the current model $F_0(x)$ with a learning rate $\alpha = 0.5$:
   
$$
   F_1(x) = F_0(x) + \alpha h_1(x)
$$

   For each $x$:
   
$$
   F_1(1) = 6 + 0.5(-3) = 4.5
$$

$$
   F_1(2) = 6 + 0.5(0) = 6
$$

$$
   F_1(3) = 6 + 0.5(3) = 7.5
$$

   | $x$ | $y$ | $F_1(x)$ |
   |------|------|------------|
   | 1    | 3    | 4.5        |
   | 2    | 6    | 6          |
   | 3    | 9    | 7.5        |

5. **Repeat**:
   Continue by calculating new residuals based on $F_1(x)$, fitting a new weak learner $h_2(x)$, and updating the model $F_2(x)$. Repeat this process for additional iterations until the residuals are minimized or a stopping criterion is met.

#### Key Takeaway:
Gradient Boosting builds the model iteratively by combining weak learners, each correcting the errors of the previous model. The learning rate $\alpha$ controls how much each weak learner contributes to the final model.
#### Further Learning Resources
- Watch this [short video series](https://www.youtube.com/watch?v=3CC4N4z3GJc) by StatQuest for more details about the math behind GBM.
- Take a look at [this notebook](https://github.com/Naviden/AI-and-Big-Data-Course/blob/main/Python%20Code/Algorithms/An%20Intro%20to%20GBM.ipynb) in which we see GBM in action
---
## 3. Neural Networks (NN)

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


---

### Recommended Reading:
- **["Pattern Recognition and Machine Learning" by Christopher M. Bishop](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)**
- **["An Introduction to Statistical Learning" by Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani](https://www.stat.berkeley.edu/users/rabbee/s154/ISLR_First_Printing.pdf)**
- **["Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville](https://www.deeplearningbook.org/)**

### Further Exploration:
- **SVM Tutorial:** Learn how to implement Support Vector Machines [here](https://scikit-learn.org/stable/modules/svm.html).
- **Implementing RNNs:** Explore a tutorial on Recurrent Neural Networks [here](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html).
