"""
Task 1: Data Preparation for ML
Topics: Categorical Encoding, Feature Scaling, Missing Value Handling, Feature Engineering
Dataset: Titanic (built-in demo dataset from seaborn)
"""

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ============================================================
# 1. LOAD DEMO DATA
# ============================================================
print("=" * 60)
print("STEP 1: Loading Titanic Dataset")
print("=" * 60)

df = sns.load_dataset("titanic")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nData types:")
print(df.dtypes)

print(f"\nMissing values per column:")
print(df.isnull().sum())

# ============================================================
# 2. HANDLE MISSING VALUES
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Handling Missing Values")
print("=" * 60)

df_proc = df.copy()

# Age: fill with median (grouped by Sex and Pclass for better imputation)
age_median = df_proc.groupby(['sex', 'pclass'])['age'].transform('median')
df_proc['age'] = df_proc['age'].fillna(age_median)
print(f"Age missing after imputation: {df_proc['age'].isnull().sum()}")

# Embarked: fill with mode
embarked_mode = df_proc['embarked'].mode()[0]
df_proc['embarked'] = df_proc['embarked'].fillna(embarked_mode)
print(f"Embarked missing after imputation: {df_proc['embarked'].isnull().sum()}")

# Deck: drop (too many missing)
deck_missing = df_proc['deck'].isnull().sum()
df_proc.drop(columns=['deck'], inplace=True)
print(f"Dropped 'deck' column ({deck_missing} missing)")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Feature Engineering")
print("=" * 60)

# Family size (new feature: sibsp + parch + 1 for self)
df_proc['family_size'] = df_proc['sibsp'] + df_proc['parch'] + 1
df_proc['is_alone'] = (df_proc['family_size'] == 1).astype(int)
print(f"Created 'family_size' and 'is_alone' features")

# Title extraction from Name (if available)
title_created = False
if 'name' in df_proc.columns:
    title_created = True
    df_proc['title'] = df_proc['name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        'Mr': 'Mr', 'Mrs': 'Mrs', 'Miss': 'Miss', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
        'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
        'Capt': 'Rare', 'Sir': 'Rare'
    }
    df_proc['title'] = df_proc['title'].map(title_mapping).fillna('Rare')
    print(f"Created 'title' feature from names: {df_proc['title'].value_counts().to_dict()}")
else:
    print("'name' column not available — skipping title extraction")

# Age bins
df_proc['age_group'] = pd.cut(df_proc['age'], bins=[0, 12, 18, 35, 60, 100],
                              labels=['Child', 'Teen', 'Adult', 'Middle_Aged', 'Senior'])
print(f"Created 'age_group' bins: {df_proc['age_group'].value_counts().to_dict()}")

# Fare bins
df_proc['fare_group'] = pd.qcut(df_proc['fare'], q=4, labels=['Low', 'Medium', 'High', 'Very_High'],
                                duplicates='drop')
print(f"Created 'fare_group' quartile bins based on fare")

# ============================================================
# 4. SELECT RELEVANT COLUMNS
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Column Selection")
print("=" * 60)

drop_cols = [col for col in ['alive', 'who', 'adult_male', 'class', 'embark_town', 'name', 'alone'] if col in df_proc.columns]
df_proc.drop(columns=drop_cols, inplace=True, errors='ignore')
print(f"Dropped columns: {drop_cols}")
print(f"Remaining columns: {list(df_proc.columns)}")

# Separate target
target = 'survived'
y = df_proc[target]
X = df_proc.drop(columns=[target])

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# ============================================================
# 5. IDENTIFY COLUMN TYPES
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Identifying Column Types")
print("=" * 60)

categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")

# ============================================================
# 6. ENCODING CATEGORICALS
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Encoding Categorical Variables")
print("=" * 60)

# Option A: One-Hot Encoding
print("\n--- One-Hot Encoding ---")
X_ohe = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
print(f"Shape after OHE: {X_ohe.shape}")
print(f"New columns: {list(X_ohe.columns)}")

# Option B: We'll also build a scikit-learn pipeline below
print("\n--- Label Encoding (for ordinal features) ---")
df_le = X.copy()
for col in categorical_cols:
    le = LabelEncoder()
    df_le[col] = le.fit_transform(df_le[col].astype(str))
    print(f"  {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ============================================================
# 7. SCALING NUMERICALS
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Feature Scaling")
print("=" * 60)

# StandardScaler (Z-score normalization)
scaler_std = StandardScaler()
X_standardized = X_ohe.copy()
X_standardized[numerical_cols] = scaler_std.fit_transform(X_standardized[numerical_cols])
print("\nAfter StandardScaler (mean=0, std=1):")
for col in numerical_cols:
    print(f"  {col} -> mean={X_standardized[col].mean():.4f}, std={X_standardized[col].std():.4f}")

# MinMaxScaler (range [0,1])
scaler_mm = MinMaxScaler()
X_minmax = X_ohe.copy()
X_minmax[numerical_cols] = scaler_mm.fit_transform(X_minmax[numerical_cols])
print("\nAfter MinMaxScaler (range [0,1]):")
for col in numerical_cols:
    print(f"  {col} -> min={X_minmax[col].min():.4f}, max={X_minmax[col].max():.4f}")

# ============================================================
# 8. SCIKIT-LEARN PIPELINE (Production-ready approach)
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: sklearn Pipeline (Production-ready)")
print("=" * 60)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

X_pipeline = preprocessor.fit_transform(X)
feature_names = (
    numerical_cols +
    list(preprocessor.named_transformers_['cat']
         .named_steps['onehot'].get_feature_names_out(categorical_cols))
)

print(f"Pipeline output shape: {X_pipeline.shape}")
print(f"Total features: {len(feature_names)}")

# ============================================================
# 9. TRAIN/TEST SPLIT
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: Train/Test Split")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_pipeline, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]} samples")
print(f"Test:  {X_test.shape[0]} samples")
print(f"Target distribution (train): {y_train.value_counts().to_dict()}")

# ============================================================
# 10. SAVE CLEANED DATA
# ============================================================
print("\n" + "=" * 60)
print("STEP 10: Saving Cleaned Dataset")
print("=" * 60)

# Save the OHE DataFrame (most readable)
output_path = "C:\\Users\\91720\\Pictures\\Screenshots\\Internship CODE- MORE\\Task 1\\cleaned_titanic_data.csv"
df_final = pd.concat([X_ohe, y], axis=1)
df_final.to_csv(output_path, index=False)
print(f"Saved cleaned dataset to: {output_path}")
print(f"Final shape: {df_final.shape}")
print(f"Final columns ({len(df_final.columns)}): {list(df_final.columns)}")

print("\n" + "=" * 60)
print("TASK 1 COMPLETE: Data Preprocessing Done!")
print("=" * 60)
title_str = ", title" if title_created else ""
print(f"""
Summary of techniques applied:
  1. Missing Value Handling: median imputation (Age), mode imputation (Embarked), column drop (Deck; {deck_missing} missing)
  2. Feature Engineering: family_size, is_alone{title_str}, age_group, fare_group
  3. Categorical Encoding: One-Hot Encoding + Label Encoding
  4. Feature Scaling: StandardScaler + MinMaxScaler
  5. Production Pipeline: sklearn ColumnTransformer + Pipeline
  6. Train/Test Split (80/20, stratified)
""")
