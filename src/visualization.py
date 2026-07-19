import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from src.utils import RESULTS_DIR, ensure_results_dirs


plt.rcParams.update({"figure.dpi": 120, "font.size": 11})


def plot_class_distribution(y, save: bool = False):
    counts = y.value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Legitimate (0)", "Fraud (1)"], counts.values, color=["steelblue", "tomato"])
    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 500, f"{v:,}", ha="center")
    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "class_distribution.png")
        fig.savefig(path)
        print(f"Saved: {path}")
    plt.show()


def plot_feature_distributions(df: pd.DataFrame, features: list = None, save: bool = False):
    if features is None:
        features = ["Amount", "Time"] + [f"V{i}" for i in range(1, 6)]
    n = len(features)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
    for i, feat in enumerate(features):
        axes[i].hist(df[df["Class"] == 0][feat], bins=50, alpha=0.6, label="Legit", color="steelblue", density=True)
        axes[i].hist(df[df["Class"] == 1][feat], bins=50, alpha=0.6, label="Fraud", color="tomato", density=True)
        axes[i].set_title(feat)
        axes[i].legend()
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Feature Distributions by Class", y=1.01)
    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "feature_distributions.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()


def plot_confusion_matrix(cm, model_name: str, save: bool = True):
    ensure_results_dirs()
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legitimate", "Fraud"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "confusion_matrices", f"{model_name}.png")
        fig.savefig(path)
        print(f"Saved: {path}")
    plt.show()


def plot_roc_curves(results: list, save: bool = True):
    ensure_results_dirs()
    fig, ax = plt.subplots(figsize=(7, 6))
    for r in results:
        ax.plot(r["fpr"], r["tpr"], label=f"{r['model_name']} (AUC={r['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "roc_curves", f"roc_comparison_{results[0]['model_name']}.png")
        fig.savefig(path)
        print(f"Saved: {path}")
    plt.show()


def plot_pr_curves(results: list, save: bool = True):
    print("Inside pr curve")
    ensure_results_dirs()
    fig, ax = plt.subplots(figsize=(7, 6))
    for r in results:
        ax.plot(r["recall"], r["precision"], label=f"{r['model_name']} (AUPRC={r['auprc']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Model Comparison")
    ax.legend(loc="upper right")
    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "pr_curves", f"auprc_comparison_{results[0]['model_name']}.png")
        fig.savefig(path)
        print(f"Saved: {path}")
    plt.show()


def plot_metrics_bar(metrics_df: pd.DataFrame, save: bool = True):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metrics_df.plot(x="model", y="roc_auc", kind="bar", ax=axes[0], color="steelblue", legend=False)
    axes[0].set_title("ROC-AUC by Model")
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_ylim(0.8, 1.0)
    axes[0].tick_params(axis="x", rotation=30)

    metrics_df.plot(x="model", y="auprc", kind="bar", ax=axes[1], color="tomato", legend=False)
    axes[1].set_title("AUPRC by Model")
    axes[1].set_ylabel("AUPRC")
    axes[1].set_ylim(0.5, 1.0)
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "metrics_bar_chart.png")
        fig.savefig(path)
        print(f"Saved: {path}")
    plt.show()
