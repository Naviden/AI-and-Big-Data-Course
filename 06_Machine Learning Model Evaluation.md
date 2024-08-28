# Machine Learning Models


### Topics Covered:

- **Model Evaluation Methods**

### Objectives:
1. Learn about different methods to evaluate the performance of ML models.

## 1. Model Evaluation Methods

### 1.1 Cross-Validation
- **Purpose:** To assess how a model generalizes to an independent dataset.

- **Method:** The dataset is divided into $k$ subsets, and the model is trained on $k-1$ subsets while being tested on the remaining subset. This process is repeated $k$ times, with each subset used as the test set once.

- **Mathematical Notation:**
  $$\text{CV Error} = \frac{1}{k} \sum_{i=1}^{k} \text{Error}_i$$
  - Where $\text{Error}_i$ is the error on the $i$-th fold.

- **Use Case:** Ensures that the model's performance is consistent across different subsets of data.

### 1.2 Confusion Matrix
- **Purpose:** A performance measurement for classification models, showing the actual vs. predicted classifications.

- **Components:**
  - **True Positives (TP):** Correctly predicted positive cases.
  - **True Negatives (TN):** Correctly predicted negative cases.
  - **False Positives (FP):** Incorrectly predicted positive cases.
  - **False Negatives (FN):** Incorrectly predicted negative cases.

- **Mathematical Representation:**
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
  $$\text{Precision} = \frac{TP}{TP + FP}$$
  $$\text{Recall} = \frac{TP}{TP + FN}$$
  $$\text{F1 Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

- **Use Case:** Evaluating classification models, particularly in imbalanced datasets.

### 1.3 Precision, Recall, and F1-Score
- **Precision:** The proportion of true positives among all predicted positives.

- **Recall (Sensitivity):** The proportion of true positives among all actual positives.

- **F1-Score:** The harmonic mean of precision and recall, balancing the two metrics.

---

### Recommended Reading:
- **"An Introduction to Statistical Learning" by Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani**
- **"Pattern Recognition and Machine Learning" by Christopher M. Bishop**