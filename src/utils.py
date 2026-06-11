import os
import joblib
import numpy as np
import pandas as pd


MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def save_model(model, name: str):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"Model saved: {path}")


def load_model(name: str):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    return joblib.load(path)


def ensure_results_dirs():
    for sub in ("classification_reports", "confusion_matrices", "roc_curves", "pr_curves"):
        os.makedirs(os.path.join(RESULTS_DIR, sub), exist_ok=True)


def set_seed(seed: int = 42):
    np.random.seed(seed)
