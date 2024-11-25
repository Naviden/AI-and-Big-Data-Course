# Data Preprocessing and Feature Engineering

### Topics Covered:
- **Data Cleaning**
- **Feature Selection**
- **Feature Extraction**
- **Normalization**
- **Scaling**

### Objectives:
1. Understand the importance of data preprocessing in machine learning.
2. Learn techniques for cleaning and transforming raw data.
3. Apply feature engineering techniques to select and create features that improve model performance.
4. Normalize and scale features to prepare data for training ML models.

---

## 1. Data Cleaning

### Definition:
Data cleaning involves identifying and correcting (or removing) errors and inconsistencies in data to improve its quality. This step is crucial because the quality of your data directly impacts the performance of your machine learning models.

### Key Tasks:
- **Handling Missing Data:** 
  - Methods include removing rows/columns with missing values, imputing missing values using the mean, median, or mode, or using more advanced techniques like K-nearest neighbors (KNN) imputation.
  - Example: If a dataset has missing entries in the "Age" column, you can replace them with the mean age of the dataset. ([More details + Python  notebook](https://github.com/Naviden/Data-Quality-Issues/blob/main/Missing%20data.ipynb))
- **Outlier Detection and Removal:**
  - Outliers can be detected using statistical methods like the Z-score or IQR method and can be removed or corrected if they are erroneous.
  - Example: In a dataset of house prices, an outlier might be a house listed at a price significantly higher or lower than the rest, which could skew model predictions.
- **Data Consistency:**
  - Ensuring consistency in data format, units of measurement, and naming conventions.
  - Example: Converting all date formats to a standard format (e.g., YYYY-MM-DD).

---

## 2. Feature Selection

### Definition
Feature selection involves identifying and selecting the most relevant features from your dataset that contribute the most to the predictive power of the model. This process reduces the dimensionality of the data, improves computational efficiency, and helps in building simpler, faster, and more interpretable models. Additionally, it mitigates the risk of overfitting by removing irrelevant or redundant features.

### Importance
- **Improves Model Performance**: Reduces noise in the data and enhances the predictive accuracy of models.
- **Speeds Up Computation**: Decreases training and inference times by working with fewer features.
- **Increases Interpretability**: Simplifies the model, making it easier to understand and explain.
- **Reduces Overfitting**: Eliminates irrelevant features that might lead the model to capture noise rather than meaningful patterns.


### Techniques

#### 1. **Filter Methods**
- **Definition**: Select features based on their statistical properties, independent of the machine learning algorithm.
- **How It Works**:
  - Features are scored using metrics like correlation, mutual information, chi-squared tests, or variance thresholds.
  - Features with the highest scores are selected.
- **Examples**:
  - **Pearson Correlation Coefficient**: Selects features that have a strong linear relationship with the target variable.
  - **Chi-Squared Test**: Evaluates the dependence between categorical features and the target variable.
- **Use Case**: Quickly reducing dimensionality in datasets with many irrelevant features.


#### 2. **Wrapper Methods**
- **Definition**: Evaluate subsets of features by training a machine learning model and selecting the subset that yields the best performance.
- **How It Works**:
  - Iteratively adds or removes features based on the model’s performance (e.g., accuracy, F1-score).
  - Can be computationally expensive, especially with large datasets.
- **Techniques**:
  - **Forward Selection**: Starts with an empty feature set and adds features one at a time based on performance improvement.
  - **Backward Elimination**: Starts with all features and removes the least impactful features iteratively.
  - **Recursive Feature Elimination (RFE)**: Fits a model and recursively removes the least important features.
- **Examples**:
  - Using **RFE** with a decision tree to identify the most significant features in classification problems.
- **Use Case**: Best suited for smaller datasets or when computational resources are not a constraint.


#### 3. **Embedded Methods**
- **Definition**: Perform feature selection during the model training process by integrating it into the learning algorithm.
- **How It Works**:
  - Models inherently rank or penalize features during training, often using regularization techniques.
  - Features with low importance are automatically ignored or assigned low weights.
- **Techniques**:
  - **Lasso Regression (L1 Regularization)**: Shrinks coefficients of less important features to zero, effectively performing feature selection.
  - **Tree-Based Models**: Models like random forests and gradient boosting assign feature importance scores based on their contribution to splits.
- **Examples**:
  - **Lasso Regression**: Automatically selects features while training by penalizing large coefficients.
  - **Random Forest Feature Importance**: Ranks features based on their contribution to reducing impurity in decision trees.
- **Use Case**: Ideal for datasets with many features where regularization can improve generalization.


### Best Practices
- **Understand Your Data**: Use domain knowledge to select meaningful features.
- **Start Simple**: Begin with filter methods for a quick overview before diving into computationally expensive techniques.
- **Combine Techniques**: Use filter methods for initial screening and wrapper or embedded methods for fine-tuning.
- **Cross-Validation**: Always validate your feature selection pipeline to ensure robustness and avoid overfitting.


### Tools and Libraries
- **Python Libraries**:
  - `sklearn.feature_selection`: Includes RFE, mutual information, chi-squared tests, and variance thresholds.
  - `LIME` and `SHAP`: Help interpret feature importance and selection.
  - `xgboost` and `lightgbm`: Provide built-in feature importance tools for tree-based models.

---

## 3. Feature Extraction

### Definition:
Feature extraction is the process of transforming raw data into a set of features that can be used as inputs to a machine learning model. This process is particularly useful when the original data is too large or complex to be used directly.

### Techniques:
- **[Principal Component Analysis (PCA)](https://www.youtube.com/watch?v=FgakZw6K1QQ):**
  - PCA reduces the dimensionality of the data by transforming the original features into a new set of uncorrelated features (principal components) that capture the most variance in the data.
  - Mathematical Formulation:
      $$Z = XW$$
    - Where $Z$ is the matrix of principal components, $X$ is the original data matrix, and $W$ is the matrix of eigenvectors.
  - Example: In this scenario, a dataset containing numerous financial transaction features (e.g., transaction amount, merchant ID, location, time of day, device type) can be highly dimensional, with many features being correlated or redundant. PCA reduces the dataset to its principal components, capturing the most significant patterns in the data. This lower-dimensional representation enables faster and more accurate fraud detection by highlighting anomalies without the noise of irrelevant or redundant features.
- **Text Feature Extraction ([TF-IDF](https://www.youtube.com/watch?v=vZAXpvHhQow), [Word Embeddings](https://www.youtube.com/watch?v=viZrOnJclY0&t=11s)):**
  - Techniques like TF-IDF and word embeddings convert text data into numerical features that can be used by machine learning models.
  - Example: Converting a corpus of text into TF-IDF vectors to use in text classification tasks.
_[Python Code](https://github.com/Naviden/AI-and-Big-Data-Course/blob/main/Python%20Code/sklearn_PCA.py)_
---

## 4. Normalization

### Definition:
Normalization is the process of adjusting the values of features to a common scale, without distorting differences in the ranges of values. This is particularly important for algorithms that compute distances between data points, such as KNN or SVM.

### Techniques:
- **Min-Max Scaling:**
  - Rescales the feature values to a fixed range, typically [0, 1].
  - Mathematical Formulation:
      $$x' = \frac{x - x_{\text{min}}}{x_{\text{max}} - x_{\text{min}}}$$
    - Where $x$ is the original value, $x_{\text{min}}$ and $x_{\text{max}}$ are the minimum and maximum values of the feature.
  - Example: Scaling all pixel values in an image dataset to the range [0, 1].
- **Z-score Standardization:**
  - Centers the feature values around the mean and scales them based on standard deviation.
  - Mathematical Formulation:
      $$z = \frac{x - \mu}{\sigma}$$
    - Where $x$ is the original value, $\mu$ is the mean, and $\sigma$ is the standard deviation.
  - Example: Standardizing features before applying algorithms like PCA, which assume normally distributed data.

---

## 5. Scaling

### Definition:
Scaling involves adjusting the range of features to bring them to a similar scale. It is a critical step when the features have different units or magnitudes, which can affect the performance of machine learning algorithms.

### Techniques:
- **MaxAbs Scaling:**
  - Scales the data by the maximum absolute value of each feature, preserving the sign of the data.
  - Mathematical Formulation:
      $$x' = \frac{x}{|x_{\text{max}}|}$$
    - Where $|x_{\text{max}}|$ is the maximum absolute value of the feature.
  - Example: Useful for sparse data like text, where preserving the sparsity pattern is important.
- **Robust Scaling:**
  - Scales features using statistics that are robust to outliers, such as the median and interquartile range (IQR).
  - Mathematical Formulation:
      $$x' = \frac{x - \text{median}(x)}{\text{IQR}(x)}$$
    - Where IQR is the interquartile range, calculated as $Q3 - Q1$.
  - Example: Scaling features in datasets with outliers that might otherwise dominate the feature scaling.

---

## Difference Between Normalization and Scaling

Normalization and scaling are essential preprocessing techniques used to prepare data for machine learning algorithms. While they may seem similar, they serve different purposes and operate in distinct ways.

---

### Normalization
Normalization adjusts the values of features to a common scale, often between a specified range like [0, 1], without distorting the relative differences between the feature values. This process is particularly beneficial for algorithms sensitive to the scale of data, such as KNN, SVM, or PCA. Normalization often involves transforming the distribution of feature values.

- **Focus**: Brings features to a common scale by modifying their distribution.
- **Techniques**: Min-Max Scaling, Z-score Standardization.
- **Output**: Feature values are transformed to a specific range or distribution.
- **Example**: Rescaling pixel intensity values in images to the range [0, 1].

---

### Scaling
Scaling adjusts the magnitude of feature values to make them comparable. Unlike normalization, scaling does not constrain the data to a specific range but ensures that features with different units or magnitudes do not disproportionately influence the model.

- **Focus**: Adjusts the range of feature values to make them comparable by magnitude.
- **Techniques**: MaxAbs Scaling, Robust Scaling.
- **Output**: Data retains its structure but is scaled to a common magnitude.
- **Example**: Scaling features like income (in thousands) and age (in years) to prevent income from dominating due to its larger values.

---

### Key Distinction
- **Normalization** often involves transforming the distribution of data and focuses on statistical properties like the mean and standard deviation.
- **Scaling** focuses on the relative sizes of feature values and adjusts them to comparable magnitudes, often using robust measures to handle outliers.

---

## Numerical Example

Consider the following dataset with two features:

| Feature 1 (Age) | Feature 2 (Income in $) |
|------------------|-------------------------|
| 25               | 50,000                 |
| 30               | 60,000                 |
| 35               | 80,000                 |
| 40               | 100,000                |
| 45               | 150,000                |

---

### Normalization (Min-Max Scaling)
| Feature 1 (Age, normalized) | Feature 2 (Income, normalized) |
|-----------------------------|---------------------------------|
| 0.0                         | 0.0                            |
| 0.25                        | 0.1                            |
| 0.5                         | 0.3                            |
| 0.75                        | 0.5                            |
| 1.0                         | 1.0                            |

---

### Scaling (MaxAbs Scaling)
Scaling adjusts values by dividing them by the maximum absolute value of each feature:
| Feature 1 (Age, scaled) | Feature 2 (Income, scaled) |
|--------------------------|---------------------------|
| 0.556                    | 0.333                    |
| 0.667                    | 0.4                      |
| 0.778                    | 0.533                    |
| 0.889                    | 0.667                    |
| 1.0                      | 1.0                      |

---

## Are Both Normalization and Scaling Needed or Are They Alternatives?

Normalization and scaling are **alternatives**, not requirements to be used together in most preprocessing pipelines. The choice between the two depends on the nature of the data and the machine learning algorithm being used.

---

### Key Differences

- **Normalization**:
  - Focuses on transforming feature values into a specific range (e.g., [0, 1]) or distribution (e.g., mean = 0, standard deviation = 1).
  - Commonly used when the algorithm relies on distances or gradients and expects features to have similar ranges or distributions (e.g., KNN, SVM, Neural Networks, PCA).
  - **Example**: Normalizing pixel values in images to the range [0, 1].

- **Scaling**:
  - Adjusts the magnitude of features without necessarily altering their range or distribution.
  - Useful when the algorithm is sensitive to feature magnitudes but can tolerate variations in range or outliers (e.g., linear regression, gradient-based models like Logistic Regression, or Lasso).
  - **Example**: Scaling income values and age values to similar magnitudes for regression analysis.

---

### Are Both Needed?

No, normalization and scaling are not typically used together because they serve overlapping purposes:

1. **Use normalization** when:
   - The data contains features with varying ranges or distributions.
   - The algorithm is distance-based or dot-product-based (e.g., KNN, SVM, Neural Networks, PCA).

2. **Use scaling** when:
   - You need to standardize feature magnitudes without constraining them to a specific range.
   - The algorithm is less sensitive to distributions but requires uniform feature scaling (e.g., regression-based models or sparse data).

---

### When Both Might Be Considered

In rare cases, both might be used, but this is uncommon and task-specific. For example:
- You might **scale data robustly** to handle outliers and then **normalize** it to a range [0, 1] for visualization or a specific model requirement.


### Recommended Reading:
- **["Feature Engineering and Selection: A Practical Approach for Predictive Models" by Max Kuhn and Kjell Johnson](http://www.feat.engineering/)**


### Further Exploration:
- **Scikit-Learn Documentation:** Explore Python tools for preprocessing data and engineering features [here](https://scikit-learn.org/stable/modules/preprocessing.html).
