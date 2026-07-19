import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_transactions_df():
    """Small imbalanced dataset shaped like the real creditcard.csv schema."""
    rng = np.random.RandomState(42)
    n_legit, n_fraud = 190, 10
    n = n_legit + n_fraud

    df = pd.DataFrame(
        {
            "Time": rng.uniform(0, 172_792, size=n),
            "V1": rng.normal(size=n),
            "V2": rng.normal(size=n),
            "V3": rng.normal(size=n),
            "V4": rng.normal(size=n),
            "V5": rng.normal(size=n),
            "Amount": rng.exponential(scale=50, size=n),
            "Class": np.array([0] * n_legit + [1] * n_fraud),
        }
    )
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


@pytest.fixture
def synthetic_split(synthetic_transactions_df):
    from sklearn.model_selection import train_test_split

    df = synthetic_transactions_df
    X = df.drop(columns=["Class"])
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture
def isolated_models_dir(tmp_path, monkeypatch):
    """Redirect src.utils.MODELS_DIR (and callers that imported save_model) to a tmp dir."""
    import src.utils as utils

    monkeypatch.setattr(utils, "MODELS_DIR", str(tmp_path / "models"))
    return tmp_path / "models"


@pytest.fixture
def isolated_results_dir(tmp_path, monkeypatch):
    """Redirect RESULTS_DIR everywhere it was bound via `from src.utils import RESULTS_DIR`."""
    import src.utils as utils

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    results_dir = str(results_dir)
    monkeypatch.setattr(utils, "RESULTS_DIR", results_dir)

    try:
        import src.evaluation as evaluation

        monkeypatch.setattr(evaluation, "RESULTS_DIR", results_dir)
    except ImportError:
        pass

    import src.visualization as visualization

    monkeypatch.setattr(visualization, "RESULTS_DIR", results_dir)

    return tmp_path / "results"


@pytest.fixture
def isolated_processed_dir(tmp_path, monkeypatch):
    import src.data_preprocessing as data_preprocessing

    monkeypatch.setattr(data_preprocessing, "PROCESSED_DIR", str(tmp_path / "processed"))
    return tmp_path / "processed"
