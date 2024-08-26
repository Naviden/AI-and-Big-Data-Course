# Machine Learning Tasks


### Topics Covered:
- **Regression**
- **Classification**
- **Clustering**
- **Dimension Reduction**

### Objectives:
By the end of this segment, students will be able to:
1. Understand the key tasks in machine learning, including regression, classification, clustering, and dimension reduction.
2. Identify the appropriate algorithms for each task.
3. Apply these tasks to real-world data problems.

---

## 1. Regression

### Definition:
Regression is a supervised learning task where the goal is to predict a continuous output variable (dependent variable) based on one or more input features (independent variables).

### Key Concepts:
- **Dependent Variable (Target):** The continuous variable that we want to predict.
- **Independent Variables (Features):** The input variables that are used to make predictions.

### Mathematical Formulation:
The goal is to find a function $f(x)$ that maps the input features $x$ to the continuous target $y$:

$$ y = f(x) + \epsilon $$

Where $\epsilon$ is the error term representing the difference between the predicted and actual values.

### Common Algorithms:
- **Linear Regression:**
  - Predicts $y$ by fitting a linear equation to the observed data.
  - Formula: $y = \beta_0 + \beta_1x_1 + \dots + \beta_nx_n + \epsilon$
- **Polynomial Regression:**
  - Extends linear regression by adding polynomial terms to model non-linear relationships.
  - Formula: $y = \beta_0 + \beta_1x + \beta_2x^2 + \dots + \beta_nx^n + \epsilon$
- **Ridge and Lasso Regression:**
  - Linear regression with regularization to prevent overfitting.
  - Ridge: Minimize $\sum_{i=1}^{n} \left( y_i - f(x_i) \right)^2 + \lambda \sum_{j=1}^{p} \beta_j^2$
  - Lasso: Minimize $\sum_{i=1}^{n} \left( y_i - f(x_i) \right)^2 + \lambda \sum_{j=1}^{p} |\beta_j|$

### Examples:
- **House Price Prediction:** Estimating house prices based on features like size, location, and number of rooms.
- **Sales Forecasting:** Predicting future sales based on historical data.

---

## 2. Classification

### Definition:
Classification is a supervised learning task where the goal is to predict a categorical label (class) for given input features.

### Key Concepts:
- **Classes (Labels):** The distinct categories into which the data points are classified.
- **Decision Boundary:** The surface that separates different classes in the feature space.

### Mathematical Formulation:
Given input features $x$, the goal is to assign a class label $y \in \{1, 2, \dots, K\}$, where $K$ is the number of classes:

$$ P(y=k|x) = \frac{e^{\beta_0 + \beta_1x_1 + \dots + \beta_nx_n}}{\sum_{j=1}^{K} e^{\beta_{0_j} + \beta_{1_j}x_1 + \dots + \beta_{n_j}x_n}} $$

### Common Algorithms:
- **Logistic Regression:**
  - Used for binary classification.
  - Formula: $P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1x_1 + \dots + \beta_nx_n)}}$
- **Support Vector Machines (SVM):**
  - Finds the hyperplane that best separates the classes.
  - Formula: $f(x) = \text{sign}(w \cdot x + b)$
- **Decision Trees:**
  - Models decisions by splitting the data based on feature values.
  - Decision Rule: Split the data based on feature $x_j$ at threshold $t$.
- **K-Nearest Neighbors (KNN):**
  - Classifies a data point based on the majority class among its $k$ nearest neighbors.
- **Random Forest:**
  - An ensemble method that builds multiple decision trees and aggregates their predictions.

### Examples:
- **Spam Detection:** Classifying emails as spam or not spam.
- **Medical Diagnosis:** Predicting whether a patient has a particular disease based on symptoms and test results.

---

## 3. Clustering

### Definition:
Clustering is an unsupervised learning task where the goal is to group data points into clusters based on their similarity, without pre-labeled categories.

### Key Concepts:
- **Clusters:** Groups of similar data points.
- **Centroid:** The center of a cluster.

### Mathematical Formulation:
The goal is to assign each data point $x_i$ to a cluster $C_j$ such that:

$$ \text{minimize} \sum_{j=1}^{k} \sum_{x \in C_j} \| x - \mu_j \|^2 $$

Where $\mu_j$ is the centroid of cluster $C_j$.

### Common Algorithms:
- **K-Means Clustering:**
  - Partitions data into $k$ clusters by minimizing the variance within each cluster.
  - Objective: Minimize $\sum_{i=1}^{k} \sum_{x \in C_i} \| x - \mu_i \|^2$
- **Hierarchical Clustering:**
  - Builds a hierarchy of clusters using methods like single linkage or complete linkage.
- **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):**
  - Groups together points that are closely packed and marks points in low-density regions as outliers.
- **Gaussian Mixture Models (GMM):**
  - Assumes data is generated from a mixture of several Gaussian distributions and identifies the parameters of these distributions.

### Examples:
- **Customer Segmentation:** Grouping customers based on purchasing behavior.
- **Image Segmentation:** Dividing an image into different regions based on pixel similarities.

---

## 4. Dimension Reduction

### Definition:
Dimension reduction is an unsupervised learning task aimed at reducing the number of input features while preserving as much information as possible.

### Key Concepts:
- **Principal Components:** New features generated from the original features that are uncorrelated and ordered by the amount of variance they explain.
- **Variance:** The amount of information captured by the principal components.

### Mathematical Formulation:
Given data matrix $X$ with $n$ features, the goal is to project $X$ into a lower-dimensional space:

$$ Z = XW $$

Where $W$ is the matrix of eigenvectors (principal components).

### Common Algorithms:
- **Principal Component Analysis (PCA):**
  - Reduces the dimensionality of data by transforming it into principal components.
  - Objective: Maximize the variance captured by the first few principal components.
- **t-Distributed Stochastic Neighbor Embedding (t-SNE):**
  - A non-linear technique that reduces dimensionality while maintaining the local structure of data points.
- **Autoencoders:**
  - Neural networks used for unsupervised learning tasks like feature learning and dimension reduction.
  - Objective: Minimize the reconstruction error between input and output.

### Examples:
- **Data Visualization:** Reducing high-dimensional data to 2D or 3D for visualization.
- **Feature Extraction:** Identifying the most important features in a dataset for improving model performance.

---

### Recommended Reading:
- **"The Elements of Statistical Learning" by Trevor Hastie, Robert Tibshirani, and Jerome Friedman**
- **"Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville**

### Further Exploration:
- **Scikit-Learn Documentation:** Explore practical examples of regression, classification, clustering, and dimension reduction [here](https://scikit-learn.org/stable/).
- **Introduction to Clustering and Dimensionality Reduction:** Watch an explanatory video [YouTube link].

---

**Next Up: Machine Learning Algorithm Families**