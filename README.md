# Adult Income Classification + Vehicle Price Regression

This notebook contains two machine learning workflows:

1) **Adult Income (Binary Classification)** using tabular demographic/financial features, with preprocessing + SMOTE and model comparison (Logistic Regression, SVM, KNN).
2) **Vehicle Price Prediction (Regression)** using numeric + categorical vehicle attributes with a ColumnTransformer preprocessing pipeline.

## What’s inside
- `adult_income_prediction.ipynb` — EDA, feature engineering, preprocessing, modeling, and evaluation for both tasks.

## Methods used
**Classification**
- Numeric scaling (StandardScaler)
- Categorical encoding (OneHotEncoder)
- Class imbalance handling (SMOTE)
- Models: Logistic Regression, SVM (SVC), KNN
- Metrics: Accuracy, Precision, Recall, F1

**Regression**
- Mixed-type preprocessing (ColumnTransformer)
- Models include tree/ensemble regressors (see notebook)

## Data
This repository does **not** include dataset files. The notebook references:
- `data.csv` (Adult Income classification)
- `vehicles.csv` (vehicle price regression)

Update paths in the notebook to match your local setup.

## How to run
```bash
jupyter notebook adult_income_prediction.ipynb
