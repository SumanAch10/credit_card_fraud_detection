# Credit Card Fraud Detection — Machine Learning Model Comparison

A machine learning project that detects fraudulent credit card transactions on the Kaggle Credit Card Fraud dataset. Five classifiers are trained, evaluated with fraud-appropriate metrics, and compared to identify the best model for a real-world financial setting.

---

## Table of Contents

- [Project Objective](#project-objective)
- [Dataset](#dataset)
- [Features](#features)
- [Key Challenges](#key-challenges)
- [Project Structure](#project-structure)
- [Notebooks](#notebooks)
- [Source Modules](#source-modules)
- [Models](#models)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Tech Stack](#tech-stack)

---

## Project Objective

Build a fraud detection system capable of identifying fraudulent transactions from highly imbalanced data. Since fraudulent transactions are rare, raw accuracy is misleading. This project focuses on **precision**, **recall**, **F1-score**, **ROC-AUC**, and **AUPRC** to evaluate how each model would perform in production, and analyses the tradeoff between catching fraud (recall) and reducing false alarms (precision).

---

## Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Total transactions | 284,807 |
| Fraudulent | 492 (≈ 0.17 %) |
| Legitimate | 284,315 |
| Target column | `Class` (`1` = fraud, `0` = legitimate) |

> Download `creditcard.csv` from Kaggle and place it at `data/raw/creditcard.csv` before running any notebook.

The dataset is severely imbalanced — fraud makes up less than 0.2 % of all transactions.

---

## Features

| Column | Description |
|---|---|
| `V1` – `V28` | PCA-transformed features (original features anonymised for privacy) |
| `Time` | Seconds elapsed since the first transaction in the dataset |
| `Amount` | Transaction amount in euros |
| `Class` | Target variable — `1` fraud, `0` legitimate |

`V1`–`V28` are not directly interpretable because they result from a PCA transformation applied to the original (confidential) transaction features.

---

## Key Challenges

| Challenge | Approach |
|---|---|
| Severe class imbalance (0.17 % fraud) | `class_weight="balanced"`, SMOTE, undersampling |
| Misleading accuracy score | Focus on AUPRC, F1, recall instead |
| Reducing false negatives | Tune decision threshold; optimise recall for fraud class |
| Model selection | Compare linear, tree-based, ensemble, and neural network |
| Threshold tradeoffs | Precision–Recall curves per model |

---

## Project Structure

```
credit_card_fraud_detection/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv          # original Kaggle dataset (not committed)
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_logistic_regression.ipynb
│   ├── 05_decision_tree.ipynb
│   ├── 06_random_forest.ipynb
│   ├── 07_xgboost.ipynb
│   ├── 08_model_comparison.ipynb
│   └── 09_neural_nets.ipynb
│
├── src/
│   ├── data_preprocessing.py       # load, scale, split, save/load processed data
│   ├── feature_engineering.py      # SMOTE, undersampling, class-weight helpers
│   ├── model_training.py           # train functions for all five models
│   ├── evaluation.py               # metrics computation and report saving
│   ├── visualization.py            # all plotting helpers
│   └── utils.py                    # save/load model, path constants, seed
│
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── neural_network.pkl
│
├── results/
│   ├── classification_reports/     # per-model .txt reports
│   ├── confusion_matrices/         # per-model .png heatmaps
│   ├── roc_curves/                 # ROC comparison plots
│   ├── pr_curves/                  # Precision-Recall comparison plots
│   └── metrics_comparison.csv      # combined metrics table
│
├── reports/                        # final written report
└── presentation/                   # slide deck
```

---

## Notebooks

Run the notebooks in order — each one builds on the previous step.

| Notebook | Description |
|---|---|
| `01_data_understanding.ipynb` | Load the dataset, inspect shape/types/nulls, review class distribution |
| `02_eda.ipynb` | Visualise feature distributions by class, correlation heatmap, amount/time analysis |
| `03_preprocessing.ipynb` | Scale `Time` and `Amount`, stratified train/test split, save processed files |
| `04_logistic_regression.ipynb` | Train Logistic Regression with `class_weight="balanced"`, evaluate, plot curves |
| `05_decision_tree.ipynb` | Train Decision Tree, evaluate, compare depth vs. performance |
| `06_random_forest.ipynb` | Train Random Forest ensemble, evaluate, feature importance |
| `07_xgboost.ipynb` | Train XGBoost with `scale_pos_weight`, evaluate, PR/ROC curves |
| `08_model_comparison.ipynb` | Side-by-side metric comparison across all trained models |
| `09_neural_nets.ipynb` | Build and train a dense neural network, evaluate with same metrics |

---

## Source Modules

### `src/data_preprocessing.py`
| Function | Description |
|---|---|
| `load_data(path)` | Read `creditcard.csv`; defaults to `data/raw/creditcard.csv` |
| `scale_features(df)` | `StandardScaler` on `Time` and `Amount` only (`V1`–`V28` are already PCA-scaled) |
| `split_data(df, test_size, random_state)` | Stratified 80/20 train-test split |
| `save_processed(...)` | Write split CSVs to `data/processed/` |
| `load_processed()` | Read split CSVs back from `data/processed/` |

### `src/feature_engineering.py`
| Function | Description |
|---|---|
| `apply_smote(X_train, y_train)` | Oversample the minority class with SMOTE |
| `apply_undersample(X_train, y_train, sampling_strategy)` | Randomly undersample the majority class |
| `apply_smote_undersampling(X_train, y_train)` | Combined pipeline: SMOTE then undersample |
| `compute_class_weight(y_train)` | Returns sklearn-compatible `class_weight` dict |

### `src/model_training.py`
| Function | Description |
|---|---|
| `train_logistic_regression(...)` | `LogisticRegression` with `class_weight="balanced"`, `solver="lbfgs"` |
| `train_decision_tree(...)` | `DecisionTreeClassifier` with `max_depth=10`, `class_weight="balanced"` |
| `train_random_forest(...)` | `RandomForestClassifier` with 100 estimators, `class_weight="balanced"` |
| `train_xgboost(...)` | `XGBClassifier` with `scale_pos_weight`, `eval_metric="aucpr"`, 200 estimators |
| `train_neural_network(...)` | Keras `Sequential` — Dense(64)→Dense(32)→Dense(1, sigmoid), binary cross-entropy |

All training functions call `save_model()` and persist the fitted model to `models/`.

### `src/evaluation.py`
| Function | Description |
|---|---|
| `evaluate_model(model, X_test, y_test, model_name)` | Computes ROC-AUC, AUPRC, classification report, confusion matrix; saves `.txt` report |
| `save_metrics_comparison(results)` | Writes all model metrics to `results/metrics_comparison.csv` |

### `src/visualization.py`
| Function | Description |
|---|---|
| `plot_class_distribution(y)` | Bar chart of fraud vs. legitimate counts |
| `plot_feature_distributions(df, features)` | Overlapping histograms per feature split by class |
| `plot_confusion_matrix(cm, model_name)` | Heatmap saved to `results/confusion_matrices/` |
| `plot_roc_curves(results)` | Multi-model ROC overlay saved to `results/roc_curves/` |
| `plot_pr_curves(results)` | Multi-model Precision-Recall overlay saved to `results/pr_curves/` |
| `plot_metrics_bar(metrics_df)` | Side-by-side bar charts of ROC-AUC and AUPRC |

### `src/utils.py`
| Item | Description |
|---|---|
| `save_model(model, name)` | Serialise model to `models/<name>.pkl` via `joblib` |
| `load_model(name)` | Deserialise model from `models/<name>.pkl` |
| `ensure_results_dirs()` | Create all `results/` subdirectories if missing |
| `set_seed(seed)` | Set `numpy` random seed for reproducibility |

---

## Models

Five classifiers are compared, each addressing class imbalance differently:

| Model | Imbalance Strategy | Key Hyperparameters |
|---|---|---|
| Logistic Regression | `class_weight="balanced"` | `solver="lbfgs"`, `max_iter=1000` |
| Decision Tree | `class_weight="balanced"` | `max_depth=10` |
| Random Forest | `class_weight="balanced"` | `n_estimators=100`, `n_jobs=-1` |
| XGBoost | `scale_pos_weight` (auto-computed) | `n_estimators=200`, `max_depth=6`, `learning_rate=0.05`, `eval_metric="aucpr"` |
| Neural Network | None (raw imbalance) | Dense(64, relu)→Dense(32, relu)→Dense(1, sigmoid), Adam, 20 epochs |

---

## Results

Evaluated on a held-out 20 % stratified test set (56,962 transactions, 98 fraud cases).

| Model | ROC-AUC | AUPRC | Fraud Precision | Fraud Recall | Fraud F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.9722 | 0.7189 | 0.06 | 0.92 | 0.11 |
| Decision Tree | 0.8915 | 0.4639 | 0.13 | 0.79 | 0.23 |
| Random Forest | 0.9573 | 0.8629 | 0.91 | 0.79 | 0.84 |
| **XGBoost** | **0.9769** | **0.8680** | 0.81 | 0.85 | **0.83** |

> Neural network results are available in `notebooks/09_neural_nets.ipynb` and `results/confusion_matrices/Neural_Network.png`.

**Key takeaways:**
- **XGBoost** achieves the highest ROC-AUC (0.9769) and AUPRC (0.8680), making it the best overall model.
- **Random Forest** is a close second with the highest fraud precision (0.91), meaning fewer false alarms.
- **Logistic Regression** has the highest fraud recall (0.92) but very low precision (0.06) — it flags almost everything as fraud.
- **Decision Tree** underperforms on AUPRC (0.4639), indicating it struggles with the probability calibration needed for imbalanced data.
- AUPRC is the most informative metric here because it focuses on the minority class and is not inflated by the large number of legitimate transactions.

---

## Installation

**Python 3.9+ recommended.**

```bash
# Clone the repository
git clone <repo-url>
cd credit_card_fraud_detection

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> `imbalanced-learn` is required by `src/feature_engineering.py` but is not in `requirements.txt`. Install it separately if you use SMOTE:
> ```bash
> pip install imbalanced-learn
> ```

---

## Usage

### Step 1 — Get the data

Download `creditcard.csv` from Kaggle and place it at:
```
data/raw/creditcard.csv
```

### Step 2 — Run notebooks in order

```bash
jupyter notebook
```

Open and run notebooks `01` through `09` sequentially. Each notebook imports from `src/` so the working directory must be the project root.

### Step 3 — Use source modules directly

```python
from src.data_preprocessing import load_data, scale_features, split_data, save_processed
from src.model_training import train_xgboost
from src.evaluation import evaluate_model
from src.visualization import plot_confusion_matrix, plot_roc_curves

# Load and preprocess
df = load_data()
df = scale_features(df)
X_train, X_test, y_train, y_test = split_data(df)
save_processed(X_train, X_test, y_train, y_test)

# Train
model = train_xgboost(X_train, y_train)

# Evaluate
result = evaluate_model(model, X_test, y_test, "xgboost")

# Visualise
plot_confusion_matrix(result["confusion_matrix"], "xgboost")
plot_roc_curves([result])
```

### Step 4 — Load a saved model

```python
from src.utils import load_model

model = load_model("xgboost")   # loads models/xgboost.pkl
```

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Logistic Regression, Decision Tree, Random Forest, preprocessing, metrics |
| `xgboost` | Gradient-boosted tree classifier |
| `tensorflow` / `keras` | Neural network model |
| `imbalanced-learn` | SMOTE and random undersampling |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualisations |
| `joblib` | Model serialisation |
| `jupyter` | Interactive notebooks |
