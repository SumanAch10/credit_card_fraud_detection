import os

import numpy as np

from src.utils import ensure_results_dirs, load_model, save_model, set_seed


def test_save_and_load_model_roundtrip(isolated_models_dir):
    model = {"coef": [1, 2, 3], "kind": "dummy"}
    save_model(model, "dummy_model")

    saved_path = os.path.join(str(isolated_models_dir), "dummy_model.pkl")
    assert os.path.exists(saved_path)

    loaded = load_model("dummy_model")
    assert loaded == model


def test_ensure_results_dirs_creates_all_subdirs(isolated_results_dir):
    ensure_results_dirs()

    for sub in ("classification_reports", "confusion_matrices", "roc_curves", "pr_curves"):
        assert os.path.isdir(os.path.join(str(isolated_results_dir), sub))


def test_set_seed_is_reproducible():
    set_seed(42)
    first = np.random.rand(5)

    set_seed(42)
    second = np.random.rand(5)

    np.testing.assert_array_equal(first, second)
