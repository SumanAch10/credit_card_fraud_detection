# Credit Card Fraud Detection — Machine Learning Model Comparison

A machine learning project focused on detecting fraudulent credit card transactions using the popular Kaggle credit card fraud dataset.  
This project compares multiple classification algorithms, evaluates them using fraud-appropriate metrics, and analyzes the tradeoff between catching fraud and reducing false alarms.

---

## Project Objective

The goal of this project is to build a fraud detection system that can identify fraudulent credit card transactions from highly imbalanced transaction data.

Fraud detection is not simply about achieving high accuracy. Since fraudulent transactions are rare, the project focuses on metrics such as precision, recall, F1-score, ROC-AUC, and AUPRC to evaluate how useful each model would be in a real-world financial setting.

---

## Dataset

The dataset used is the Kaggle Credit Card Fraud Detection dataset.

| Property | Description |
|---|---|
| Source | Kaggle Credit Card Fraud Detection |
| Transactions | 284,807 |
| Fraud cases | 492 |
| Legitimate cases | 284,315 |
| Target column | `Class` |
| Fraud label | `1` |
| Legitimate label | `0` |

The dataset is highly imbalanced, with fraud cases making up only a very small percentage of all transactions.

---

## Features

The dataset contains:

- `V1` to `V28`: PCA-transformed features
- `Time`: Seconds elapsed since the first transaction
- `Amount`: Transaction amount
- `Class`: Target variable

The original transaction features were transformed using PCA for privacy reasons, so the `V1`–`V28` features are not directly interpretable.

---

## Key Challenges

This project focuses on several real-world ML challenges:

- Severe class imbalance
- Misleading accuracy scores
- Choosing the right evaluation metrics
- Reducing false negatives without creating too many false positives
- Comparing linear, tree-based, and ensemble models
- Understanding threshold tradeoffs
- Evaluating model usefulness for fraud detection

---

## Project Workflow

```text
Load data
↓
Explore class imbalance
↓
Visualize feature distributions
↓
Scale Time and Amount
↓
Train multiple models
↓
Evaluate with classification metrics
↓
Compare ROC-AUC and AUPRC
↓
Analyze false positives and false negatives
↓
Tune hyperparameters
↓
Select best model

## The structure to follow

credit-card-fraud-detection/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── creditcard.csv
│   └── processed/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_logistic_regression.ipynb
│   ├── 05_decision_tree.ipynb
│   ├── 06_random_forest.ipynb
│   ├── 07_xgboost.ipynb
│   └── 08_model_comparison.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── utils.py
│
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
│
├── results/
│   ├── classification_reports/
│   │   ├── logistic_regression.txt
│   │   ├── decision_tree.txt
│   │   ├── random_forest.txt
│   │   └── xgboost.txt
│   │
│   ├── confusion_matrices/
│   │   ├── logistic_regression.png
│   │   ├── decision_tree.png
│   │   ├── random_forest.png
│   │   └── xgboost.png
│   │
│   ├── roc_curves/
│   │   └── roc_comparison.png
│   │
│   ├── pr_curves/
│   │   └── auprc_comparison.png
│   │
│   └── metrics_comparison.csv
│
├── reports/
│   └── final_report.md
│
└── presentation/
    └── project_summary.pptx