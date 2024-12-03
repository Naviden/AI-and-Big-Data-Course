# Classic Machine Leaning Algorithms

### Topics Covered:
- **Regression Models**
- **Classification and Regression Trees (CART)**

### Objectives:
1. Understand the fundamental concepts and applications of regression models and CART.
2. Learn how to implement and interpret regression models for continuous target predictions.
3. Explore the structure and decision-making process of CART for classification and regression tasks.
4. Recognize the advantages and limitations of each model, particularly in practical scenarios.

---

## Regression Models

### Definition:
Regression models are a class of machine learning models used to predict a continuous target variable based on one or more input features. These models learn the relationship between the independent variables (features) and the dependent variable (target) by fitting a function to the data.

Regression is one of the most fundamental techniques in supervised learning and is widely used in areas such as finance, healthcare, and engineering.

### Key Concepts:
- **Independent Variable (Feature):** The input variables used to make predictions.
- **Dependent Variable (Target):** The continuous variable the model aims to predict.
- **Loss Function:** A function that measures the difference between the predicted and actual values. Common loss functions in regression include:
  - **Mean Squared Error (MSE):** Penalizes large errors more heavily.
  - **Mean Absolute Error (MAE):** Penalizes all errors equally.
- **Coefficient:** A weight assigned to each feature, representing its contribution to the prediction.
- **Intercept:** A constant term that allows the model to adjust its prediction baseline.

### Mathematical Formulation:
In its simplest form, linear regression models predict the target $y$ as a weighted sum of the input features $x$:

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n + \epsilon
$$

Where:
- $y$ is the predicted value,
- $\beta_0$ is the intercept,
- $\beta_i$ are the coefficients for each feature $x_i$,
- $\epsilon$ is the error term.

The model learns the coefficients $\beta$ by minimizing a loss function, such as:

$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

Where:
- $N$ is the number of data points,
- $y_i$ is the actual value,
- $\hat{y}_i$ is the predicted value.

### Numerical Example of a Regression Model

#### Problem:
We have a simple dataset with one feature $x$ and a continuous target $y$:

| $x$ | $y$ |
|-----|-----|
| 1   | 3   |
| 2   | 5   |
| 3   | 7   |

#### Steps:

1. **Model Representation**:
   Assume a simple linear regression model:
   $$
   y = \beta_0 + \beta_1 x
   $$

2. **Initialize Parameters**:
   Start with initial guesses for the coefficients, e.g., $\beta_0 = 0$ and $\beta_1 = 1$.

3. **Make Predictions**:
   Use the current parameters to predict $y$ for each $x$:
   $$
   \hat{y} = \beta_0 + \beta_1 x
   $$

   Example for $x = 1$:
   $$
   \hat{y} = 0 + 1 \cdot 1 = 1
   $$

   Predicted values:
   | $x$ | $y$ | $\hat{y}$ |
   |-----|-----|-----------|
   | 1   | 3   | 1         |
   | 2   | 5   | 2         |
   | 3   | 7   | 3         |

4. **Compute Loss**:
   Calculate the Mean Squared Error (MSE):
   $$
   \text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
   $$

   Example:
   $$
   \text{MSE} = \frac{1}{3} \left[(3 - 1)^2 + (5 - 2)^2 + (7 - 3)^2\right] = \frac{1}{3} (4 + 9 + 16) = 9.67
   $$

5. **Update Coefficients**:
   Use gradient descent to update $\beta_0$ and $\beta_1$:
   $$
   \beta_0 \leftarrow \beta_0 - \eta \frac{\partial \text{MSE}}{\partial \beta_0}, \quad \beta_1 \leftarrow \beta_1 - \eta \frac{\partial \text{MSE}}{\partial \beta_1}
   $$

   Where $\eta$ is the learning rate.

6. **Repeat**:
   Iterate the process until the loss converges.

### Why Regression Models Are Important:
1. **Simplicity and Interpretability:** Regression models, especially linear regression, are simple and easy to interpret.
2. **Foundation for Advanced Models:** Regression is the basis for more complex models like Generalized Linear Models (GLMs) and regularized regression techniques.
3. **Wide Applicability:** Regression is used in diverse domains, from economics to biology, for modeling relationships between variables.

---
## Classification and Regression Trees (CART)

### Definition:
Classification and Regression Trees (CART) are a family of decision tree algorithms used for both classification and regression tasks. These models partition the data into subsets based on feature values, building a tree structure where each node represents a decision rule and each leaf represents an output (class or value).

CART models are part of the broader family of tree-based methods and form the foundation of ensemble methods like Random Forests and Gradient Boosting Machines.

### Key Concepts:
- **Tree Structure:** CART builds a binary tree where each internal node splits the data into two subsets based on a decision rule.
- **Splitting Criterion:** The choice of the split at each node is determined by minimizing a loss function:
  - For classification: Gini impurity or entropy.
  - For regression: Mean Squared Error (MSE).
- **Pruning:** To prevent overfitting, CART trees can be pruned to remove unnecessary splits, simplifying the model.
- **Recursive Partitioning:** The tree is built top-down, splitting the dataset recursively until a stopping criterion is met (e.g., maximum depth or minimum samples per leaf).

### Mathematical Formulation:
For a given dataset $D$, CART splits the data at each node to maximize homogeneity in the resulting subsets.

#### Splitting Criterion for Classification:
At a node, the Gini impurity is defined as:

$$
G = 1 - \sum_{i=1}^{C} p_i^2
$$

Where:
- $C$ is the number of classes,
- $p_i$ is the proportion of samples in class $i$.

The algorithm selects the split that minimizes the weighted Gini impurity of the child nodes.

#### Splitting Criterion for Regression:
For regression, the split minimizes the Mean Squared Error (MSE):

$$
MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y})^2
$$

Where:
- $N$ is the number of samples,
- $y_i$ is the actual value,
- $\hat{y}$ is the predicted value (mean of the values in the node).

### Numerical Example of CART

#### Problem:
We have a simple dataset with one feature $x$ and a binary target $y$:

| $x$ | $y$ |
|------|------|
| 1    | 0    |
| 2    | 0    |
| 3    | 1    |
| 4    | 1    |

#### Steps:

1. **Choose a Split**:
   Consider possible split points for $x$ (e.g., $x = 2.5$). Compute the Gini impurity for the resulting subsets:
   
   - Left subset: $x \leq 2.5 \rightarrow \{(1, 0), (2, 0)\}$
   - Right subset: $x > 2.5 \rightarrow \{(3, 1), (4, 1)\}$
   
   Compute Gini impurity for each subset:
   
   For the left subset:
   $$
   G_{left} = 1 - (1^2 + 0^2) = 0
   $$
   
   For the right subset:
   $$
   G_{right} = 1 - (0^2 + 1^2) = 0
   $$

2. **Evaluate Split Quality**:
   Compute the weighted Gini impurity for the split:
   $$
   G_{split} = \frac{N_{left}}{N} G_{left} + \frac{N_{right}}{N} G_{right}
   $$
   Since both subsets are pure ($G_{left} = G_{right} = 0$), the split is optimal.

3. **Continue Splitting**:
   Repeat the process recursively for each subset until the tree reaches a stopping criterion (e.g., maximum depth).

### Why CART Is Important:
1. **Interpretability:** CART models are easy to interpret and visualize, making them suitable for explaining predictions.
2. **Foundation for Ensembles:** CART forms the basis of ensemble models like Random Forests and Gradient Boosting Machines.
3. **Flexibility:** CART can handle both classification and regression tasks and is robust to outliers.
