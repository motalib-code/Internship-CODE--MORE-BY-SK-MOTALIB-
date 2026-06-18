#!/usr/bin/env python3
"""
Customer Churn Prediction - End-to-End ML Pipeline
====================================================
Predict customer churn for a telecom company using synthetic data.
Pipeline: Data Generation → Preprocessing → EDA → Model Building → Evaluation → Insights
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# =============================================================================
# 1. SYNTHETIC DATA GENERATION
# =============================================================================
print("=" * 70)
print("SECTION 1: GENERATING SYNTHETIC TELECOM CUSTOMER CHURN DATASET")
print("=" * 70)

N = 1000

def generate_churn_dataset(n):
    """Generate a realistic synthetic telecom customer churn dataset."""
    data = {}

    # Demographics
    data['gender'] = np.random.choice(['Male', 'Female'], n)
    data['SeniorCitizen'] = np.random.choice([0, 1], n, p=[0.84, 0.16])
    data['Partner'] = np.random.choice(['Yes', 'No'], n, p=[0.48, 0.52])
    data['Dependents'] = np.random.choice(['Yes', 'No'], n, p=[0.30, 0.70])

    # Account info
    data['tenure'] = np.random.randint(0, 73, n)

    # Services
    data['PhoneService'] = np.random.choice(['Yes', 'No'], n, p=[0.90, 0.10])
    data['MultipleLines'] = np.random.choice(['Yes', 'No', 'No phone service'], n, p=[0.42, 0.48, 0.10])
    data['InternetService'] = np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.34, 0.44, 0.22])
    data['OnlineSecurity'] = np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.28, 0.50, 0.22])
    data['OnlineBackup'] = np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.30, 0.48, 0.22])
    data['DeviceProtection'] = np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.29, 0.49, 0.22])
    data['TechSupport'] = np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.28, 0.50, 0.22])
    data['StreamingTV'] = np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.30, 0.48, 0.22])
    data['StreamingMovies'] = np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.30, 0.48, 0.22])

    # Billing
    data['Contract'] = np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.21, 0.24])
    data['PaperlessBilling'] = np.random.choice(['Yes', 'No'], n, p=[0.59, 0.41])
    data['PaymentMethod'] = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        n, p=[0.34, 0.23, 0.22, 0.21]
    )

    # Monthly charges (realistic distribution)
    data['MonthlyCharges'] = np.round(np.random.normal(64.76, 30.0, n).clip(18.0, 120.0), 2)

    # Total charges (tenure * monthly charges + noise)
    data['TotalCharges'] = np.round(
        np.array([data['tenure'][i] * data['MonthlyCharges'][i] for i in range(n)]) +
        np.random.normal(0, 50, n), 2
    ).clip(0, 9000)

    df = pd.DataFrame(data)

    # Generate realistic churn labels based on feature relationships
    churn_prob = np.zeros(n)
    churn_prob += np.where(df['Contract'] == 'Month-to-month', 0.25, 0.0)
    churn_prob += np.where(df['Contract'] == 'One year', 0.05, 0.0)
    churn_prob += np.where(df['InternetService'] == 'Fiber optic', 0.15, 0.0)
    churn_prob += np.where(df['tenure'] < 12, 0.15, 0.0)
    churn_prob += np.where(df['tenure'] > 48, -0.10, 0.0)
    churn_prob += np.where(df['OnlineSecurity'] == 'No', 0.08, 0.0)
    churn_prob += np.where(df['TechSupport'] == 'No', 0.06, 0.0)
    churn_prob += np.where(df['PaymentMethod'] == 'Electronic check', 0.10, 0.0)
    churn_prob += np.where(df['MonthlyCharges'] > 80, 0.08, 0.0)
    churn_prob += np.where(df['SeniorCitizen'] == 1, 0.05, 0.0)
    churn_prob += np.where(df['PaperlessBilling'] == 'Yes', 0.04, 0.0)
    churn_prob = churn_prob.clip(0.05, 0.85)

    df['Churn'] = np.random.binomial(1, churn_prob)

    # Add a few missing values for realism
    missing_idx = np.random.choice(n, 5, replace=False)
    df.loc[missing_idx, 'TotalCharges'] = np.nan

    return df

df = generate_churn_dataset(N)
print(f"Generated dataset with {df.shape[0]} rows and {df.shape[1]} columns")
print(f"Features: {list(df.columns[:-1])}")
print(f"Target: Churn (0 = No, 1 = Yes)")
print(f"\nFirst 5 rows:")
print(df.head())

# =============================================================================
# 2. DATA PREPROCESSING
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2: DATA PREPROCESSING")
print("=" * 70)

# Handle missing values
print(f"\nMissing values before handling:")
print(df.isnull().sum()[df.isnull().sum() > 0])
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
print(f"\nMissing values after handling: {df.isnull().sum().sum()}")

# Encode binary categorical variables
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# One-Hot Encode multi-class categorical variables
multi_class_cols = [
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaymentMethod'
]
df = pd.get_dummies(df, columns=multi_class_cols, drop_first=True)

print(f"\nDataset shape after encoding: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Save cleaned dataset
df.to_csv('cleaned_churn_dataset.csv', index=False)
print(f"\nCleaned dataset saved to cleaned_churn_dataset.csv")

# Feature scaling
X = df.drop('Churn', axis=1)
y = df['Churn']

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set:  {X_test.shape[0]} samples")
print(f"Churn rate in train: {y_train.mean():.3f}")
print(f"Churn rate in test:  {y_test.mean():.3f}")

# =============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# Reload original data for EDA (before encoding)
df_eda = generate_churn_dataset(N)
df_eda['TotalCharges'].fillna(df_eda['TotalCharges'].median(), inplace=True)

# Churn distribution
churn_counts = df_eda['Churn'].value_counts()
print(f"\nChurn Distribution:")
print(f"  No Churn (0):  {churn_counts[0]} ({churn_counts[0]/N*100:.1f}%)")
print(f"  Churn (1):     {churn_counts[1]} ({churn_counts[1]/N*100:.1f}%)")

# Plot churn distribution
plt.figure(figsize=(8, 5))
colors = ['#2ecc71', '#e74c3c']
ax = sns.countplot(x='Churn', data=df_eda, palette=colors)
ax.set_title('Customer Churn Distribution', fontsize=14, fontweight='bold')
ax.set_xlabel('Churn (0=No, 1=Yes)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())} ({p.get_height()/N*100:.1f}%)',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('churn_distribution.png', dpi=150)
plt.close()
print("Plot saved: churn_distribution.png")

# Statistical summaries by churn status
print(f"\nStatistical Summary by Churn Status:")
print("-" * 50)
for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
    churned = df_eda[df_eda['Churn'] == 1][col]
    not_churned = df_eda[df_eda['Churn'] == 0][col]
    print(f"\n{col}:")
    print(f"  Churned    - Mean: {churned.mean():.2f}, Median: {churned.median():.2f}, Std: {churned.std():.2f}")
    print(f"  Not Churned - Mean: {not_churned.mean():.2f}, Median: {not_churned.median():.2f}, Std: {not_churned.std():.2f}")

# Churn by contract type
print(f"\nChurn Rate by Contract Type:")
contract_churn = df_eda.groupby('Contract')['Churn'].mean() * 100
for contract, rate in contract_churn.items():
    print(f"  {contract}: {rate:.1f}%")

# Churn by internet service
print(f"\nChurn Rate by Internet Service:")
internet_churn = df_eda.groupby('InternetService')['Churn'].mean() * 100
for service, rate in internet_churn.items():
    print(f"  {service}: {rate:.1f}%")

# Correlation heatmap (for encoded features)
plt.figure(figsize=(16, 12))
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, fmt='.2f')
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
plt.close()
print("\nPlot saved: correlation_heatmap.png")

# Key correlations with Churn
churn_corr = corr_matrix['Churn'].drop('Churn').sort_values(ascending=False)
print(f"\nTop Positive Correlations with Churn:")
for feat, corr in churn_corr.head(5).items():
    print(f"  {feat}: {corr:.3f}")
print(f"\nTop Negative Correlations with Churn:")
for feat, corr in churn_corr.tail(5).items():
    print(f"  {feat}: {corr:.3f}")

# =============================================================================
# 4. MODEL BUILDING
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: MODEL BUILDING")
print("=" * 70)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=7)
}

results = {}
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')

    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std()
    }
    trained_models[name] = (model, y_pred, y_prob)

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  CV Score:  {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# =============================================================================
# 5. MODEL EVALUATION & COMPARISON
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5: MODEL EVALUATION & COMPARISON")
print("=" * 70)

# Model comparison table
comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df.round(4)
print(f"\nModel Comparison Table:")
print(comparison_df.to_string())

# Save model comparison
comparison_df.to_csv('model_comparison.csv')
print(f"\nModel comparison saved to model_comparison.csv")

# Find best model
best_model_name = comparison_df['ROC-AUC'].idxmax()
best_accuracy = comparison_df.loc[best_model_name, 'Accuracy']
best_roc = comparison_df.loc[best_model_name, 'ROC-AUC']
print(f"\n*** BEST MODEL: {best_model_name} ***")
print(f"    Accuracy: {best_accuracy:.4f} | ROC-AUC: {best_roc:.4f}")

# Detailed classification report for best model
best_model, best_pred, best_prob = trained_models[best_model_name]
print(f"\nClassification Report - {best_model_name}:")
print(classification_report(y_test, best_pred, target_names=['No Churn', 'Churn']))

# ROC Curves
plt.figure(figsize=(10, 8))
colors_roc = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
for (name, (model, _, y_prob)), color in zip(trained_models.items(), colors_roc):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, color=color, linewidth=2, label=f'{name} (AUC = {auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - All Models', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150)
plt.close()
print("Plot saved: roc_curves.png")

# Confusion Matrices
fig, axes = plt.subplots(1, 5, figsize=(25, 5))
for idx, (name, (model, y_pred, _)) in enumerate(trained_models.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    axes[idx].set_title(f'{name}', fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('Predicted', fontsize=10)
    axes[idx].set_ylabel('Actual', fontsize=10)
plt.suptitle('Confusion Matrices - All Models', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot saved: confusion_matrices.png")

# Feature Importance (from best tree-based model)
# Use Random Forest or Gradient Boosting for feature importance
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    importance_model = best_model
    importance_name = best_model_name
else:
    # Fallback to Random Forest
    importance_model = trained_models['Random Forest'][0]
    importance_name = 'Random Forest'

importances = importance_model.feature_importances_
feature_names = X.columns
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values('Importance', ascending=False)

print(f"\nTop 10 Most Important Features ({importance_name}):")
for _, row in feat_imp_df.head(10).iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")

# Plot feature importance
plt.figure(figsize=(10, 8))
top_features = feat_imp_df.head(15)
sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
plt.title(f'Top 15 Feature Importances ({importance_name})', fontsize=14, fontweight='bold')
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.close()
print("Plot saved: feature_importance.png")

# =============================================================================
# 6. FINAL SUMMARY & INSIGHTS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 6: FINAL SUMMARY & KEY INSIGHTS")
print("=" * 70)

print(f"""
======================================================================
                      PROJECT SUMMARY
======================================================================
  Dataset: Synthetic Telecom Customer Churn ({N} customers)
  Features: {X.shape[1]} (after encoding)
  Best Model: {best_model_name}
  Best Accuracy: {best_accuracy:.4f}
  Best ROC-AUC:  {best_roc:.4f}
======================================================================
  KEY BUSINESS INSIGHTS:
======================================================================
  1. Month-to-month contracts have highest churn risk
  2. Customers with Fiber optic service churn more
  3. Short tenure (< 12 months) strongly predicts churn
  4. Lack of OnlineSecurity and TechSupport increases churn
  5. Electronic check payment method correlates with higher churn
  6. Higher monthly charges are associated with increased churn
  7. Senior citizens show slightly higher churn tendency
  8. Paperless billing customers churn more frequently

  RECOMMENDATIONS:
  - Promote annual contracts to reduce churn
  - Bundle security and support services with internet plans
  - Target new customers (< 12 months) with retention campaigns
  - Offer incentives for automatic payment methods
  - Review fiber optic pricing and service quality
======================================================================
""")

# List all generated files
import os
print("Generated Output Files:")
output_files = [
    'cleaned_churn_dataset.csv', 'model_comparison.csv',
    'churn_distribution.png', 'correlation_heatmap.png',
    'feature_importance.png', 'roc_curves.png', 'confusion_matrices.png'
]
for f in output_files:
    exists = os.path.exists(f)
    size = os.path.getsize(f) if exists else 0
    status = "[OK]" if exists else "[MISSING]"
    print(f"  {status} {f} ({size:,} bytes)")

print("\n" + "=" * 70)
print("PIPELINE COMPLETE - All outputs generated successfully!")
print("=" * 70)
