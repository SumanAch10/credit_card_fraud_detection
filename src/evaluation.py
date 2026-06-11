import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

from src.utils import RESULTS_DIR, ensure_results_dirs


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    ensure_results_dirs()

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"])
    roc_auc = roc_auc_score(y_test, y_proba)
    auprc = average_precision_score(y_test, y_proba)

    report_path = os.path.join(RESULTS_DIR, "classification_reports", f"{model_name}.txt")
    with open(report_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"ROC-AUC : {roc_auc:.4f}\n")
        f.write(f"AUPRC   : {auprc:.4f}\n\n")
        f.write(report)

    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"ROC-AUC: {roc_auc:.4f}  |  AUPRC: {auprc:.4f}")
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    return {
        "model_name": model_name,
        "roc_auc": roc_auc,
        "auprc": auprc,
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def save_metrics_comparison(results: list):
    rows = [
        {
            "model": r["model_name"],
            "roc_auc": round(r["roc_auc"], 4),
            "auprc": round(r["auprc"], 4),
            "tn": r["confusion_matrix"][0, 0],
            "fp": r["confusion_matrix"][0, 1],
            "fn": r["confusion_matrix"][1, 0],
            "tp": r["confusion_matrix"][1, 1],
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "metrics_comparison.csv")
    df.to_csv(path, index=False)
    print(f"\nMetrics comparison saved to {path}")
    print(df.to_string(index=False))
    return df
