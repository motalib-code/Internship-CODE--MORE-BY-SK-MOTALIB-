# Week 2 · Task 2 — Model Evaluation, Tuning & Performance Analysis

## Overview

This project evaluates and compares machine learning models for Titanic survival prediction. Two baseline classifiers (Logistic Regression, Decision Tree) are trained, evaluated, and compared. The better-performing model is then tuned via GridSearchCV.

## Dataset

- **Source:** Preprocessed Titanic dataset from Task 1 (22 features with OHE + feature engineering)
- **Target:** `survived` (binary)
- **Records:** 891 passengers (80/20 stratified split)

## Project Structure

```
├── README.md
├── requirements.txt
├── data/
│   └── cleaned_titanic_data.csv
├── notebooks/
│   └── model_evaluation_and_tuning.ipynb
├── reports/
│   └── performance_report.md
└── images/
    ├── confusion_matrix.png
    ├── confusion_matrix_tuned.png
    ├── roc_curve.png
    └── feature_importance.png
```

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/model_evaluation_and_tuning.ipynb
```

## Results Summary

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression (Baseline) | 0.8212 | 0.7601 | 0.8691 |
| Decision Tree (Baseline) | 0.7821 | 0.7254 | 0.8298 |
| Logistic Regression (Tuned) | 0.8212 | 0.7601 | 0.8730 |

**Best model:** Logistic Regression with `C=0.1`, `penalty='l2'`
**Key insight:** Hyperparameter tuning gave marginal improvement (CV F1 +1.3%). Logistic Regression consistently outperforms Decision Trees on this dataset.

## Visualizations

- `confusion_matrix.png` — Side-by-side confusion matrices for both baseline models
- `roc_curve.png` — ROC curves with AUC values
- `feature_importance.png` — Top 10 features from Decision Tree
- `confusion_matrix_tuned.png` — Tuned model confusion matrix
