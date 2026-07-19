import pytest

pytest.importorskip("imblearn")

from src.feature_engineering import (  # noqa: E402
    apply_smote,
    apply_smote_undersampling,
    apply_undersample,
    compute_class_weight,
)


def test_apply_smote_balances_classes(synthetic_split):
    X_train, _, y_train, _ = synthetic_split

    X_res, y_res = apply_smote(X_train, y_train)

    assert (y_res == 0).sum() == (y_res == 1).sum()
    assert len(X_res) == len(y_res)


def test_apply_undersample_respects_sampling_strategy(synthetic_split):
    X_train, _, y_train, _ = synthetic_split

    X_res, y_res = apply_undersample(X_train, y_train, sampling_strategy=0.5)

    n_fraud = (y_res == 1).sum()
    n_legit = (y_res == 0).sum()
    assert n_fraud > 0
    assert n_fraud / n_legit == pytest.approx(0.5, abs=1e-6)


def test_apply_smote_undersampling_combined_pipeline(synthetic_split):
    X_train, _, y_train, _ = synthetic_split

    X_res, y_res = apply_smote_undersampling(X_train, y_train)

    n_fraud = (y_res == 1).sum()
    n_legit = (y_res == 0).sum()
    assert n_fraud / n_legit == pytest.approx(0.5, abs=1e-6)


def test_compute_class_weight_matches_manual_formula(synthetic_split):
    _, _, y_train, _ = synthetic_split

    weights = compute_class_weight(y_train)

    n_total = len(y_train)
    n_fraud = y_train.sum()
    n_legit = n_total - n_fraud
    assert weights[1] == pytest.approx(n_total / (2 * n_fraud))
    assert weights[0] == pytest.approx(n_total / (2 * n_legit))
