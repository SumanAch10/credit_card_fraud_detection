import os

import numpy as np

from src.visualization import (
    plot_confusion_matrix,
    plot_metrics_bar,
    plot_pr_curves,
    plot_roc_curves,
)


def _fake_result(model_name, seed=0):
    rng = np.random.RandomState(seed)
    fpr = np.sort(rng.rand(10))
    tpr = np.sort(rng.rand(10))
    precision = np.sort(rng.rand(10))[::-1]
    recall = np.sort(rng.rand(10))
    return {
        "model_name": model_name,
        "roc_auc": 0.9,
        "auprc": 0.8,
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
    }


def test_plot_confusion_matrix_saves_png(isolated_results_dir):
    cm = np.array([[50, 2], [3, 10]])

    plot_confusion_matrix(cm, "unit_test_model", save=True)

    path = os.path.join(str(isolated_results_dir), "confusion_matrices", "unit_test_model.png")
    assert os.path.exists(path)


def test_plot_roc_curves_saves_png(isolated_results_dir):
    results = [_fake_result("unit_test_model")]

    plot_roc_curves(results, save=True)

    path = os.path.join(
        str(isolated_results_dir), "roc_curves", "roc_comparison_unit_test_model.png"
    )
    assert os.path.exists(path)


def test_plot_pr_curves_saves_png(isolated_results_dir):
    results = [_fake_result("unit_test_model")]

    plot_pr_curves(results, save=True)

    path = os.path.join(
        str(isolated_results_dir), "pr_curves", "auprc_comparison_unit_test_model.png"
    )
    assert os.path.exists(path)


def test_plot_metrics_bar_saves_png(isolated_results_dir):
    import pandas as pd

    metrics_df = pd.DataFrame({"model": ["a", "b"], "roc_auc": [0.9, 0.95], "auprc": [0.8, 0.85]})

    plot_metrics_bar(metrics_df, save=True)

    path = os.path.join(str(isolated_results_dir), "metrics_bar_chart.png")
    assert os.path.exists(path)
