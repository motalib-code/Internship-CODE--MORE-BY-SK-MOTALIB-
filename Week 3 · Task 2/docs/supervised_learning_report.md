# Supervised Learning Algorithms — A Comparative Report

## Overview

Supervised learning is a machine learning paradigm where models are trained on labeled data — each training example has an input feature vector and a known target output. The model learns a mapping from inputs to outputs, which it can then generalise to unseen data.

This report covers three fundamental supervised learning algorithms: **Linear Regression**, **Logistic Regression**, and **Decision Trees**.

---

## 1. Linear Regression

### Definition
Linear Regression models the relationship between a continuous target variable \( y \) and one or more predictor variables \( X \) by fitting a linear equation:

\[
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n + \varepsilon
\]

### How It Works
- The coefficients \( \beta \) are estimated by minimising the **Residual Sum of Squares (RSS)** — ordinary least squares (OLS).
- For high-dimensional or collinear data, regularised variants like Ridge (L2), Lasso (L1), or ElasticNet add a penalty term to prevent overfitting.

### Use Cases
- Predicting house prices, stock prices, sales revenue
- Forecasting continuous metrics (temperature, demand, energy consumption)
- Trend analysis and causal inference

### Advantages
- Simple, interpretable, and fast to train
- Closed-form solution (no iterative tuning required for OLS)
- Works well when the relationship is approximately linear

### Disadvantages
- Assumes linearity, independence of errors, homoscedasticity, and normally distributed errors
- Sensitive to outliers and multicollinearity
- Limited expressiveness — cannot capture complex non-linear patterns without feature engineering

---

## 2. Logistic Regression

### Definition
Logistic Regression estimates the probability that an instance belongs to a particular class. Despite its name, it is a **classification** algorithm.

\[
p(y=1 | X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \dots + \beta_n x_n)}}
\]

### How It Works
- The linear combination of features is passed through the **sigmoid (logistic) function**, which squashes the output to a probability between 0 and 1.
- The decision boundary is typically set at \( p \geq 0.5 \).
- Training uses **Maximum Likelihood Estimation (MLE)** via gradient descent or quasi-Newton methods.
- Multi-class variants exist (OvR, softmax / multinomial).

### Use Cases
- Binary classification: spam detection, churn prediction, disease diagnosis
- Credit scoring and risk assessment
- Multi-class: digit recognition, species classification

### Advantages
- Probabilistic output (confidence scores)
- Fast to train and easy to regularise
- Works well with linearly separable or near-separable data
- Coefficients can be interpreted as log-odds ratios

### Disadvantages
- Assumes a linear decision boundary — poor fit for complex, non-linear problems
- Sensitive to feature scaling
- Can struggle with class imbalance without weighting or resampling

---

## 3. Decision Trees

### Definition
Decision Trees partition the feature space into rectangular regions by asking a series of binary questions (e.g., "is age > 30?"). Each leaf node corresponds to a prediction — a mean value for regression or a majority class for classification.

### How It Works
- At each node, the algorithm selects the feature and threshold that best reduces impurity:
  - **Classification**: Gini impurity or entropy / information gain
  - **Regression**: Mean Squared Error (MSE) or Mean Absolute Error (MAE)
- Recursive partitioning stops when a maximum depth is reached, a minimum sample count is met, or no further gain is possible.
- Pruning (cost-complexity pruning) helps reduce overfitting.

### Use Cases
- Interpretable decision rules (e.g., loan approval, medical triage)
- Feature importance ranking
- Ensemble methods (Random Forest, Gradient Boosting, XGBoost) use trees as weak learners

### Advantages
- No feature scaling required
- Handles both numeric and categorical data
- Captures non-linear interactions automatically
- Transparent and interpretable (can be visualised)

### Disadvantages
- High variance — small changes in data can produce very different trees
- Prone to overfitting without careful pruning or ensemble aggregation
- Greedy splitting may miss globally optimal partitions

---

## Comparative Summary

| Property               | Linear Regression | Logistic Regression | Decision Trees        |
|------------------------|-------------------|---------------------|-----------------------|
| Type                   | Regression        | Classification      | Both                  |
| Output                 | Continuous value  | Class probability   | Class / mean value    |
| Decision boundary      | Linear            | Linear (in feature space) | Non-linear (axis-aligned) |
| Interpretability       | High (coefficients) | High (log-odds)   | Very high (tree rules) |
| Feature scaling        | Required          | Required            | Not required          |
| Overfitting risk       | Low (with reg.)   | Low (with reg.)     | High (needs pruning)  |
| Scalability            | Very high         | High                | Medium                |

---

## Conclusion

No single algorithm dominates across all problems. **Linear Regression** is the baseline for regression tasks with linear relationships. **Logistic Regression** remains a strong first choice for binary classification — fast, calibrated, and interpretable. **Decision Trees** shine when interpretability and non-linear patterns matter most, and they form the backbone of the most powerful ensemble methods in practice.

A principled approach is to start with the simplest model, establish a baseline, and increase complexity only when the performance gain justifies the loss of interpretability.
