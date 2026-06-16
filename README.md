# Data Preprocessing & Exploratory Data Analysis — Titanic Dataset

## Project Overview

This project demonstrates essential data preprocessing and exploratory data analysis (EDA) techniques using the famous **Titanic** dataset. The goal is to prepare raw data for machine learning by cleaning missing values, fixing data types, removing duplicates, and uncovering patterns through statistical analysis and visualizations.

## Dataset

- **Source:** [Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
- **Description:** Passenger manifest from the RMS Titanic, including survival status, class, age, sex, fare, and other attributes.
- **Records:** 891 passengers
- **Features:** 12 columns (PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked)

## Project Structure

```
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── data/
│   ├── raw_dataset.csv        # Original downloaded dataset
│   └── cleaned_dataset.csv    # Cleaned and preprocessed dataset
├── notebooks/
│   └── eda_analysis.ipynb     # Jupyter Notebook with full EDA
├── images/
│   ├── age_histogram.png      # Age distribution
│   ├── fare_vs_class.png      # Fare by passenger class
│   └── correlation_heatmap.png# Feature correlation matrix
└── src/
    └── data_cleaning.py       # Python script for data cleaning pipeline
```

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the data cleaning script

```bash
python src/data_cleaning.py
```

This loads `data/raw_dataset.csv`, handles missing values, removes duplicates, fixes data types, and saves the cleaned dataset to `data/cleaned_dataset.csv`.

### 3. Run the EDA notebook

```bash
jupyter notebook notebooks/eda_analysis.ipynb
```

Or open it in VS Code, JupyterLab, or Google Colab.

## Data Cleaning Steps

| Step | Action |
|------|--------|
| 1 | Load CSV and inspect shape, columns, dtypes |
| 2 | Fill missing `Age` with median |
| 3 | Fill missing `Embarked` with mode |
| 4 | Fill missing `Cabin` with "Unknown" |
| 5 | Remove duplicate rows |
| 6 | Convert columns to correct numeric types |
| 7 | Export cleaned dataset |

## Key Findings

- **Survival rate:** ~38% of passengers survived.
- **Class matters:** 1st class passengers had a significantly higher survival rate.
- **Age distribution:** Most passengers were between 20–40 years old.
- **Gender gap:** Female passengers survived at a much higher rate than males.
- **Fare correlation:** Higher fares correlated with higher survival (linked to class).

## Visualizations

- **Age Histogram:** Distribution of passenger ages.
- **Fare vs Passenger Class:** Box plot showing fare spread across classes.
- **Correlation Heatmap:** Numeric feature correlations including survival.

## Tools & Libraries

- Python 3.10+
- Pandas, NumPy (data manipulation)
- Matplotlib, Seaborn (visualization)
- Jupyter (notebook environment)
