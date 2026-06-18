# Customer Churn Prediction - End-to-End ML Project

## Problem Statement

Customer churn is a critical business metric for telecom companies. When customers leave (churn), the company loses recurring revenue and incurs costs to acquire new customers. This project builds a machine learning pipeline to **predict whether a customer will churn** based on their demographic information, account details, and service usage patterns. The goal is to identify key factors driving churn and enable proactive retention strategies.

## Dataset

Since real telecom datasets have privacy constraints, this project uses a **realistic synthetic dataset** of 1,000 customers with 19 original features:

| Feature | Type | Description |
|---------|------|-------------|
| gender | Categorical | Male / Female |
| SeniorCitizen | Binary | 0 = No, 1 = Yes |
| Partner | Categorical | Has partner (Yes/No) |
| Dependents | Categorical | Has dependents (Yes/No) |
| tenure | Numeric | Months as customer (0-72) |
| PhoneService | Categorical | Has phone service (Yes/No) |
| MultipleLines | Categorical | Has multiple lines |
| InternetService | Categorical | DSL / Fiber optic / No |
| OnlineSecurity | Categorical | Has online security add-on |
| OnlineBackup | Categorical | Has online backup add-on |
| DeviceProtection | Categorical | Has device protection |
| TechSupport | Categorical | Has tech support |
| StreamingTV | Categorical | Has streaming TV |
| StreamingMovies | Categorical | Has streaming movies |
| Contract | Categorical | Month-to-month / One year / Two year |
| PaperlessBilling | Categorical | Uses paperless billing (Yes/No) |
| PaymentMethod | Categorical | Payment method type |
| MonthlyCharges | Numeric | Monthly billing amount |
| TotalCharges | Numeric | Total amount charged |
| **Churn** | **Binary** | **Target: 0 = No, 1 = Yes** |

## Approach

### 1. Data Preprocessing

- **Missing Values**: 5 missing values in TotalCharges filled with median
- **Binary Encoding**: LabelEncoder applied to 6 binary categorical features
- **One-Hot Encoding**: Applied to 10 multi-class categorical features (first category dropped to avoid multicollinearity)
- **Feature Scaling**: StandardScaler applied to all features for uniform scaling
- **Train/Test Split**: 80/20 stratified split preserving churn distribution

### 2. Exploratory Data Analysis

Key findings from EDA:

- **Churn Distribution**: Dataset shows ~45% churn rate (realistic for telecom)
- **Contract Impact**: Month-to-month contracts have the highest churn rate (~55%), while two-year contracts show lowest (~15%)
- **Internet Service**: Fiber optic customers churn significantly more than DSL customers
- **Tenure Effect**: Customers with < 12 months tenure churn at nearly 2x the rate of long-term customers
- **Security Features**: Customers without OnlineSecurity or TechSupport are 1.5-2x more likely to churn
- **Payment Method**: Electronic check users show the highest churn among payment methods

### 3. Model Building

Five classification models were trained and evaluated:

| Model | Key Hyperparameters |
|-------|-------------------|
| Logistic Regression | max_iter=1000 |
| Random Forest | n_estimators=100, max_depth=10 |
| Gradient Boosting | n_estimators=100, learning_rate=0.1, max_depth=5 |
| SVM | kernel='rbf', probability=True |
| KNN | n_neighbors=7 |

### 4. Model Evaluation

Each model was evaluated using:

- **Accuracy**: Overall correct prediction rate
- **Precision**: Of predicted churners, how many actually churned
- **Recall**: Of actual churners, how many were correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve (discrimination ability)
- **Cross-Validation**: 5-fold stratified CV for generalization estimate

## Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | CV Mean |
|-------|----------|-----------|--------|----------|---------|---------|
| Logistic Regression | 0.6450 | 0.5439 | 0.4079 | 0.4662 | 0.6622 | 0.6610 |
| Random Forest | 0.6400 | 0.5476 | 0.3026 | 0.3898 | 0.6267 | 0.6470 |
| Gradient Boosting | 0.6350 | 0.5224 | 0.4605 | 0.4895 | 0.6228 | 0.6030 |
| SVM | 0.6500 | 0.5789 | 0.2895 | 0.3860 | 0.6375 | 0.6430 |
| KNN | 0.6000 | 0.4655 | 0.3553 | 0.4030 | 0.5273 | 0.5990 |

**Best Model**: Logistic Regression with 64.50% accuracy and 0.6622 ROC-AUC

## Insights

### Key Business Insights

1. **Contract Type is the Strongest Predictor**: Month-to-month customers churn at 3x the rate of annual contract holders. Offering discounts for longer commitments can significantly reduce churn.

2. **Fiber Optic Service Needs Attention**: Despite being the premium service, fiber optic customers show the highest churn rates, suggesting potential pricing or quality issues.

3. **New Customers Are At-Risk**: Customers in their first 12 months represent the highest churn risk. Onboarding experience and early engagement are critical.

4. **Security & Support Services Retain Customers**: Customers with OnlineSecurity and TechSupport add-ons show measurably lower churn rates. Bundling these services could improve retention.

5. **Payment Method Matters**: Electronic check users churn more, possibly due to manual payment friction. Promoting auto-pay enrollment could reduce churn.

6. **Monthly Charges Drive Churn**: Higher monthly charges correlate with increased churn, suggesting price sensitivity among certain customer segments.

### Recommended Actions

- **Retention Campaigns**: Target month-to-month customers with contract upgrade offers
- **Service Bundles**: Include security and support services free for the first 6 months
- **New Customer Program**: Implement a 90-day check-in program for new subscribers
- **Auto-Pay Incentives**: Offer small discounts for automatic payment enrollment
- **Fiber Optic Review**: Investigate service quality and pricing for fiber optic plans

## How to Run

```bash
# Ensure dependencies are installed
pip install numpy pandas matplotlib seaborn scikit-learn

# Run the pipeline
cd "Week 4 · Task 2"
python customer_churn_prediction.py
```

The script will:
1. Generate synthetic dataset
2. Print EDA results
3. Train and evaluate 5 models
4. Generate all output files (CSVs and PNGs)

## Dependencies

- Python 3.11+
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

## Generated Output Files

| File | Description |
|------|-------------|
| `cleaned_churn_dataset.csv` | Preprocessed dataset ready for modeling |
| `model_comparison.csv` | All model metrics in tabular format |
| `churn_distribution.png` | Bar chart of target variable distribution |
| `correlation_heatmap.png` | Feature correlation heatmap |
| `feature_importance.png` | Top 15 feature importances from best model |
| `roc_curves.png` | ROC curves for all 5 models |
| `confusion_matrices.png` | Confusion matrices for all 5 models |
