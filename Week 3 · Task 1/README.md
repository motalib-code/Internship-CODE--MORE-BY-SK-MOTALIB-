# Week 3 · Task 1 — Basic ML Model Implementation

## Overview
End-to-end machine learning pipeline using Logistic Regression on the Titanic dataset. Covers EDA, feature selection, train/test split, model training, evaluation (Accuracy, Precision, Recall, F1-Score, ROC-AUC), and performance visualization.

## Dataset
- **Source:** Kaggle Titanic (preprocessed — OHE + feature engineering)
- **Target:** `survived` (binary classification)
- **Records:** 891 passengers
- **Features:** 21

## Project Structure
```
├── README.md
├── requirements.txt
├── data/
│   ├── cleaned_titanic_data.csv
│   └── dataset_source.txt
├── notebooks/
│   └── ml_model_implementation.ipynb
├── docs/
│   └── summary_report.md
└── images/
    ├── eda_correlations.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── feature_coefficients.png
    └── metrics_summary.png
```

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebooks/ml_model_implementation.ipynb
```

## Results Summary
| Metric | Score |
|--------|-------|
| Accuracy | 0.8212 |
| Precision | 0.7903 |
| Recall | 0.7321 |
| F1-Score | 0.7601 |
| ROC-AUC | 0.8691 |
| 5-Fold CV F1 | 0.7784 |

## Key Insight
Logistic Regression achieves good discrimination (ROC-AUC 0.87) but recall is the weakest metric (0.73) due to class imbalance. F1-Score (0.76) is the most reliable evaluation metric for this dataset.
