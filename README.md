# Multiclass Classification Project: Lifestyle Risk Category (Synthetic)

> **Educational Use Only.** This is a fully **synthetic**, interpretable tabular dataset created for teaching machine learning classification. Labels do **not** represent medical diagnoses or real medical risk; use this dataset only for ML practice.

## Overview

This project involves building a **multiclass classifier** to predict a person's **Lifestyle Risk Category** based on human-understandable features such as age, height, weight, BMI, blood pressure, resting heart rate, sleep, exercise, smoking, and alcohol use.

**Target Classes (`label`):**
- `Low`
- `Elevated`
- `High`
- `Very High`

The labels are derived from a latent risk score computed from standardized features with added noise, then bucketed into four categories. The class distribution is moderately imbalanced (approximately: Elevated 30%, Low 25%, High 25%, Very High 20%), so macro-averaged metrics are recommended for evaluation.

This repository provides a starter Jupyter notebook (`Lifestyle_Risk_Multiclass_Starter.ipynb`) to load the data, perform basic EDA, and guide model development.

## Files Provided

- `train.csv`: Training data with features and `label` (3750 rows).
- `test.csv`: Test data with features only (no label; 1250 rows).
- `sample_submission.csv`: Example submission format (`id,label`).
- `Lifestyle_Risk_Multiclass_Starter.ipynb`: Starter notebook with setup, data loading, and EDA.

**Note:** Place the CSV files in the same directory as the notebook or update `DATA_DIR` in the notebook accordingly.

## Feature List

| Column                  | Type    | Unit    | Allowed Values / Range | Description |
|-------------------------|---------|---------|------------------------|-------------|
| `id`                   | int     | —       | unique                 | Row identifier (must be preserved in submissions). |
| `sex`                  | category| —       | `female`, `male`       | Biological sex (categorical). Encode before modeling. |
| `age_years`            | int     | years   | 18–75 (approx.)        | Age in years. |
| `height_cm`            | float   | cm      | ~140–205               | Body height. |
| `weight_kg`            | float   | kg      | ~40–160                | Body weight. |
| `bmi`                  | float   | kg/m²   | ~15–50                 | Body Mass Index derived from height/weight. |
| `waist_cm`             | float   | cm      | ~55–160                | Waist circumference; correlated with BMI and height. |
| `sbp_mmHg`             | int     | mmHg    | ~90–200                | Systolic blood pressure (higher is worse). |
| `dbp_mmHg`             | int     | mmHg    | ~55–120                | Diastolic blood pressure. |
| `resting_hr_bpm`       | int     | bpm     | ~45–120                | Resting heart rate (beats per minute). |
| `exercise_hours_per_week` | float | hours   | ≥ 0                    | Self-reported average weekly exercise. |
| `smoker`               | int     | —       | 0 (non-smoker), 1 (smoker) | Smoking indicator. |
| `alcohol_units_per_week` | int   | units   | ≥ 0                    | Approximate weekly alcohol units. |
| `sleep_hours_per_night` | float  | hours   | ~3.5–10.5              | Average nightly sleep duration. |
| `label`                | category| —       | `Low`, `Elevated`, `High`, `Very High` | **Target** class (only in `train.csv`). |

## Objective

Train a model on `train.csv` to predict the `label` for every row in `test.csv`. Submit predictions as a CSV matching `sample_submission.csv` (columns: `id,label`).

## Rules

1. **Multiclass Prediction**: Output exactly one class (`Low|Elevated|High|Very High`) per test row.
2. **No Leakage**: Do not access private labels or use external data that could leak information.
3. **Reproducibility**: Set and report random seeds for all random operations.
4. **Write-up**: Include in your notebook or report:
   - Problem framing and baseline model.
   - Preprocessing (e.g., encoding categoricals, scaling features) and rationale.
   - Model(s) used, training details, and hyperparameters.
   - Validation strategy (e.g., stratified K-fold cross-validation).
   - Results with multiple metrics (accuracy, macro F1, precision, recall).
   - Error analysis (e.g., confusion matrix; discuss confused classes and potential reasons).

## Requirements

- Python 3.6+
- Libraries: `numpy`, `pandas`, `scikit-learn` (for modeling and metrics). Install via:
  ```
  pip install numpy pandas scikit-learn
  ```

## Usage

1. Clone the repository or download the files.
2. Place `train.csv`, `test.csv`, and `sample_submission.csv` in the project directory (or update paths in the notebook).
3. Open `Lifestyle_Risk_Multiclass_Starter.ipynb` in Jupyter Notebook or JupyterLab.
4. Run the cells to load data and perform initial EDA.
5. Implement your preprocessing, model training, and prediction pipeline.
6. Generate predictions for `test.csv` and save as `submission.csv`.

**Example Evaluation Code** (for validation split):
```python
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

print("Accuracy:", accuracy_score(y_val, y_pred))
print("F1 (macro):", f1_score(y_val, y_pred, average="macro"))
print("Precision (macro):", precision_score(y_val, y_pred, average="macro"))
print("Recall (macro):", recall_score(y_val, y_pred, average="macro"))

print("\nClassification Report:\n", classification_report(y_val, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))
```

## Recommended Workflow

- Use **stratified** train-validation splits or cross-validation (stratify by `label` to handle imbalance).
- Start with baselines: Logistic Regression, Linear SVM, or Decision Tree.
- Preprocess: One-hot encode `sex`; scale numerical features for linear models.
- Evaluate with macro-averaged metrics: Accuracy, F1, Precision, Recall.
- Experiment with advanced models (e.g., Random Forest, XGBoost) if baselines underperform.
- Analyze errors: Use confusion matrices to identify misclassifications (e.g., boundary classes like Elevated/High may overlap due to noise).

## Contributing

This is an educational project. Feel free to fork and experiment, but do not distribute real medical data or misuse the synthetic dataset.

## License

MIT License. Use for educational purposes only.
