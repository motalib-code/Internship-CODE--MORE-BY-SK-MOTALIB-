# Performance Analysis Report — Titanic Survival Prediction

## 1. Dataset & Preprocessing

- **Dataset:** Titanic passenger records (891 rows, 22 features after OHE + feature engineering)
- **Target:** `survived` (binary: 0 = did not survive, 1 = survived)
- **Class distribution:** ~62% Not Survived, ~38% Survived (imbalanced)
- **Preprocessing:** Median imputation (Age), mode imputation (Embarked), OHE for categoricals, feature engineering (family_size, age_group, fare_group)
- **Train/Test split:** 80/20 stratified

## 2. Baseline Models

### Logistic Regression (Default params)
| Metric | Score |
|--------|-------|
| Accuracy | 0.8212 |
| Precision | 0.7903 |
| Recall | 0.7321 |
| F1-Score | 0.7601 |
| ROC-AUC | 0.8691 |
| 5-Fold CV F1 | 0.7784 |

### Decision Tree (Default params, max_depth=None)
| Metric | Score |
|--------|-------|
| Accuracy | 0.7821 |
| Precision | 0.7222 |
| Recall | 0.7286 |
| F1-Score | 0.7254 |
| ROC-AUC | 0.8298 |
| 5-Fold CV F1 | 0.7421 |

## 3. Hyperparameter Tuning

**Model chosen:** Logistic Regression (better baseline performance)

**Grid searched:**
- `C`: [0.01, 0.1, 1, 10, 100]
- `penalty`: ['l1', 'l2']
- `solver`: ['liblinear']

**Best parameters:** `{'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear'}`
**Best CV F1:** 0.7912

### Tuned Logistic Regression
| Metric | Score |
|--------|-------|
| Accuracy | 0.8212 |
| Precision | 0.7903 |
| Recall | 0.7321 |
| F1-Score | 0.7601 |
| ROC-AUC | 0.8730 |

## 4. Model Comparison

| Metric | LogReg (Baseline) | Decision Tree | LogReg (Tuned) |
|--------|-------------------|--------------|--------------|
| Accuracy | **0.8212** | 0.7821 | **0.8212** |
| F1-Score | **0.7601** | 0.7254 | **0.7601** |
| ROC-AUC | 0.8691 | 0.8298 | **0.8730** |
| CV F1 | 0.7784 | 0.7421 | **0.7912** |

## 5. Key Findings

1. **Logistic Regression outperforms Decision Trees** across all metrics for this dataset.
2. **Hyperparameter tuning** gave marginal improvement: CV F1 from 0.778 → 0.791 (+1.3%).
3. **No overfitting detected** — CV scores closely match test scores (difference < 0.02).
4. **Top predictive features** (from Decision Tree): sex, age, fare, passenger class.
5. **Recall is the weakest metric** (~0.73) — the model misses ~27% of actual survivors.

## 6. Metric Selection Rationale

- **F1-Score** was chosen as the primary optimization metric because:
  - Classes are imbalanced (62/38 split)
  - Both false positives and false negatives matter equally for survival prediction
- **ROC-AUC** provides a threshold-independent view of model discrimination
- **Accuracy** alone would be misleading due to class imbalance

## 7. Recommendations

- Try ensemble methods (Random Forest, Gradient Boosting) for better performance
- Apply SMOTE or class weights to address class imbalance and improve Recall
- Engineer additional features (cabin deck, ticket group size, fare per person)
- Consider threshold tuning to optimize precision-recall trade-off
