import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.evaluation import evaluate_model, save_metrics_comparison


def test_evaluate_model_returns_expected_keys_and_writes_report(
    synthetic_split, isolated_results_dir
):
    X_train, X_test, y_train, y_test = synthetic_split
    model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_train, y_train)

    result = evaluate_model(model, X_test, y_test, "unit_test_model")

    expected_keys = {
        "model_name",
        "roc_auc",
        "auprc",
        "confusion_matrix",
        "fpr",
        "tpr",
        "precision",
        "recall",
        "y_pred",
        "y_proba",
    }
    assert expected_keys.issubset(result.keys())
    assert 0.0 <= result["roc_auc"] <= 1.0
    assert 0.0 <= result["auprc"] <= 1.0
    assert result["confusion_matrix"].shape == (2, 2)

    report_path = os.path.join(
        str(isolated_results_dir), "classification_reports", "unit_test_model.txt"
    )
    assert os.path.exists(report_path)
    with open(report_path) as f:
        content = f.read()
    assert "unit_test_model" in content
    assert "ROC-AUC" in content


def test_save_metrics_comparison_writes_csv(isolated_results_dir):
    results = [
        {
            "model_name": "model_a",
            "roc_auc": 0.9,
            "auprc": 0.8,
            "confusion_matrix": np.array([[50, 2], [3, 10]]),
        },
        {
            "model_name": "model_b",
            "roc_auc": 0.95,
            "auprc": 0.85,
            "confusion_matrix": np.array([[48, 4], [1, 12]]),
        },
    ]

    df = save_metrics_comparison(results)

    csv_path = os.path.join(str(isolated_results_dir), "metrics_comparison.csv")
    assert os.path.exists(csv_path)

    reloaded = pd.read_csv(csv_path)
    assert list(reloaded["model"]) == ["model_a", "model_b"]
    assert df.loc[0, "tp"] == 10
    assert df.loc[1, "fn"] == 1
