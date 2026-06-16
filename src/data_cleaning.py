import pandas as pd
import numpy as np
import os

RAW_PATH = os.path.join("data", "raw_dataset.csv")
CLEAN_PATH = os.path.join("data", "cleaned_dataset.csv")


def load_data(path):
    df = pd.read_csv(path)
    print(f"Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}\n")
    return df


def inspect_missing(df):
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    summary = pd.DataFrame({"Missing Count": missing, "Percentage": missing_pct})
    print("Missing values summary:")
    print(summary[summary["Missing Count"] > 0])
    print()
    return summary


def handle_missing_values(df):
    df = df.copy()

    age_median = df["Age"].median()
    df["Age"] = df["Age"].fillna(age_median)
    print(f"Age missing values filled with median: {age_median:.1f}")

    fare_median = df["Fare"].median()
    df["Fare"] = df["Fare"].fillna(fare_median)
    print(f"Fare missing values filled with median: {fare_median:.2f}")

    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    print(f"Embarked missing values filled with mode: {df['Embarked'].mode()[0]}")

    df["Cabin"] = df["Cabin"].fillna("Unknown")
    print(f"Cabin missing values filled with 'Unknown' ({df['Cabin'].isnull().sum()} remaining)")

    print()
    return df


def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Duplicates removed: {before - after}")
    return df


def fix_data_types(df):
    df = df.copy()
    df["Survived"] = df["Survived"].astype("int64")
    df["Pclass"] = df["Pclass"].astype("int64")
    df["Age"] = df["Age"].astype("float64")
    df["Fare"] = df["Fare"].astype("float64")
    df["SibSp"] = df["SibSp"].astype("int64")
    df["Parch"] = df["Parch"].astype("int64")
    print("Data types corrected to int64/float64 where appropriate")
    print(f"Final data types:\n{df.dtypes}\n")
    return df


def save_clean_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Cleaned dataset saved to: {path}")
    print(f"Cleaned shape: {df.shape}")


def main():
    print("=" * 60)
    print("DATA CLEANING PIPELINE - Titanic Dataset")
    print("=" * 60)

    df = load_data(RAW_PATH)

    print("--- Missing Values (Before) ---")
    inspect_missing(df)

    print("--- Handling Missing Values ---")
    df = handle_missing_values(df)

    print("--- Removing Duplicates ---")
    df = remove_duplicates(df)

    print("--- Fixing Data Types ---")
    df = fix_data_types(df)

    print("--- Missing Values (After) ---")
    inspect_missing(df)

    save_clean_data(df, CLEAN_PATH)
    print("\nData cleaning complete!")


if __name__ == "__main__":
    main()
