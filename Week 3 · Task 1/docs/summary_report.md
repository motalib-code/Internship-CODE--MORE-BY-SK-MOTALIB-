# Model Implementation Report — Logistic Regression on Titanic

## 1. Objective
Build and evaluate a baseline Logistic Regression classifier on the Titanic dataset to predict passenger survival. Demonstrate the end-to-end ML workflow: data loading, EDA, feature selection, train/test split, model training, evaluation, and visualization.

## 2. Dataset
- **Source:** Kaggle Titanic (preprocessed via Task 1)
- **Records:** 891 passengers (80/20 stratified split)
- **Features:** 21 (OHE-encoded categoricals + engineered features)
- **Target:** `survived` (0 = Not Survived, 1 = Survived)
- **Class balance:** 62% / 38% (imbalanced)

## 3. Model
- **Algorithm:** Logistic Regression
- **Hyperparameters:** `random_state=42`, `max_iter=1000`, default `C=1.0`, `penalty='l2'`
- **Scaling:** StandardScaler applied to numeric features (age, fare, sibsp, parch, family_size)

## 4. Performance Metrics

| Metric | Score |
|--------|-------|
| Accuracy | 0.8212 |
| Precision | 0.7903 |
| Recall | 0.7321 |
| F1-Score | 0.7601 |
| ROC-AUC | 0.8691 |
| 5-Fold CV F1 (mean ± std) | 0.7784 ± 0.0288 |

## 5. Key Findings
1. **Accuracy (0.82)** is reasonable but misleading due to class imbalance — F1-Score (0.76) is the better metric.
2. **Recall (0.73)** is the weakest metric — the model misses ~27% of actual survivors.
3. **ROC-AUC (0.87)** indicates good discrimination between classes.
4. **CV F1 (0.78)** is close to test F1 (0.76), confirming no overfitting.
5. **Top positive coefficients:** sex_female, fare, age_group_Middle_Aged.
6. **Top negative coefficients:** sex_male, pclass, age_group_Senior.

## 6. Visualizations
| File | Description |
|------|-------------|
| `eda_correlations.png` | Target distribution + top correlations with survived |
| `confusion_matrix.png` | Confusion matrix (TN=93, FP=15, FN=41, TP=30) |
| `roc_curve.png` | ROC curve with AUC = 0.869 |
| `feature_coefficients.png` | Top 12 logistic regression coefficients |
| `metrics_summary.png` | Bar chart of all evaluation metrics |

## 7. Limitations
1. **Linear decision boundary** — Logistic Regression cannot capture non-linear interactions without manual feature engineering.
2. **Class imbalance** — 62/38 split biases the model toward the majority class.
3. **Missing data** — Cabin (77% missing) was coarsely replaced with "Unknown".
4. **No hyperparameter tuning** — Default parameters used; GridSearchCV could yield marginal gains.

## 8. Recommendations
- Try non-linear models (Random Forest, Gradient Boosting)
- Apply class weighting or SMOTE to address imbalance
- Engineer additional features (ticket group size, cabin deck)
- Tune hyperparameters via GridSearchCV
- Threshold tuning to optimize precision-recall trade-off
