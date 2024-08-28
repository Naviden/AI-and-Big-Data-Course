# Explainable AI (XAI)


### Topics Covered:
- **Principles of XAI**
- **Importance of Explainability**
- **Techniques for Explainability**
  - Local Models (LIME)
  - Global Models (SHAP)

### Objectives:
1. Understand the principles and importance of explainable AI.
2. Recognize the need for transparency in AI models, especially in critical applications.
3. Learn about specific techniques such as LIME and SHAP that are used to achieve explainability in machine learning models.

---

## 1. Principles of Explainable AI (XAI)

### Definition:
Explainable AI (XAI) refers to a set of processes and methods that allow human users to understand and trust the output of AI and machine learning models. The goal of XAI is to make AI systems more transparent, interpretable, and accountable.

### Key Principles:
- **Transparency:** The extent to which the inner workings of an AI model can be understood by humans.
- **Interpretability:** The degree to which a human can understand the cause of a decision made by an AI system.
- **Trustworthiness:** Ensuring that AI systems behave as expected and their decisions can be justified.
- **Fairness:** Avoiding bias and ensuring that AI systems treat all users or subjects equitably.

### Importance of XAI:
- **Accountability:** In many industries, particularly those involving high-stakes decisions (e.g., healthcare, finance), it is crucial to understand how and why a model makes certain predictions.
- **Ethical AI:** Ensuring that AI systems operate without unintended biases and with fairness to all individuals.
- **Regulatory Compliance:** Legal requirements, such as the General Data Protection Regulation (GDPR) in the EU, mandate the right to explanations for decisions made by automated systems.
- **User Trust:** For AI systems to be widely accepted and used, users must trust that the systems are making decisions in a fair and understandable way.

---

## 2. Techniques for Explainability

### Local vs. Global Explanations:
- **Local Models:** Explain individual predictions by approximating the model's behavior in a small, local region around a specific data point.
- **Global Models:** Explain the overall behavior of the model by examining its decision-making process across the entire dataset.

### 2.1 Local Interpretable Model-agnostic Explanations (LIME)

#### Description:
LIME is a technique used to explain the predictions of any machine learning classifier. It works by perturbing the input data and observing how the predictions change, thereby creating a simpler, interpretable model that approximates the complex model locally.

#### How LIME Works:
1. **Perturbation:** LIME generates new data points by slightly modifying the original input.
2. **Prediction:** The black-box model makes predictions for these new data points.
3. **Approximation:** LIME fits a simple, interpretable model (like a linear model) to these predictions, providing an explanation for the original prediction.

#### Mathematical Formulation:
Given a complex model $f$ and an input $x$, LIME approximates $f$ around $x$ using a simpler model $g$:

$$ g(z) \approx f(x) \text{ for inputs } z \text{ close to } x $$

Where $z$ represents perturbed versions of $x$.

#### Use Case:
- **Credit Scoring:** Explaining why a specific loan application was approved or denied by a machine learning model.

### 2.2 SHapley Additive exPlanations (SHAP)

#### Description:
SHAP is a method based on cooperative game theory that provides consistent and theoretically sound explanations for the output of any machine learning model. SHAP values represent the contribution of each feature to the prediction made by the model.

#### How SHAP Works:
1. **Shapley Values:** SHAP calculates the average marginal contribution of a feature across all possible combinations of features.
2. **Additive Explanation:** The model’s prediction is represented as a sum of the contributions from each feature.

#### Mathematical Formulation:
For a model $f(x)$ with $n$ features, the SHAP value for feature $i$ is given by:

$$ \phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right] $$

Where $S$ is a subset of all features $F$, and $f(S)$ is the model prediction based only on features in $S$.

#### Use Case:
- **Healthcare:** Explaining a model’s prediction of a patient’s likelihood of developing a disease, ensuring that medical practitioners can understand the contributing factors.

---

### Recommended Reading:
- **["Interpretable Machine Learning: A Guide for Making Black Box Models Explainable" by Christoph Molnar](https://christophm.github.io/interpretable-ml-book/)**
- **["A Survey on the Explainability of Supervised Machine Learning" by Burkart and Huber](https://dl.acm.org/doi/pdf/10.1613/jair.1.12228)**

### Further Exploration:
- **LIME Implementation:** Explore a Python implementation of LIME [here](https://github.com/marcotcr/lime).
- **SHAP Documentation:** Learn more about SHAP and how to apply it to your models [here](https://shap.readthedocs.io/).