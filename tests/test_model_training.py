import os

import pytest

pytest.importorskip("xgboost")
pytest.importorskip("tensorflow")

from src.model_training import (  # noqa: E402
    train_decision_tree,
    train_logistic_regression,
    train_neural_network,
    train_random_forest,
    train_xgboost,
)


def _assert_saved(isolated_models_dir, name):
    assert os.path.exists(os.path.join(str(isolated_models_dir), f"{name}.pkl"))


def test_train_logistic_regression_fits_and_saves(synthetic_split, isolated_models_dir):
    X_train, X_test, y_train, _ = synthetic_split

    model = train_logistic_regression(X_train, y_train)

    assert len(model.predict(X_test)) == len(X_test)
    _assert_saved(isolated_models_dir, "logistic_regression")


def test_train_decision_tree_fits_and_saves(synthetic_split, isolated_models_dir):
    X_train, X_test, y_train, _ = synthetic_split

    model = train_decision_tree(X_train, y_train)

    assert len(model.predict(X_test)) == len(X_test)
    _assert_saved(isolated_models_dir, "decision_tree")


def test_train_random_forest_fits_and_saves(synthetic_split, isolated_models_dir):
    X_train, X_test, y_train, _ = synthetic_split

    model = train_random_forest(X_train, y_train, n_estimators=10)

    assert len(model.predict(X_test)) == len(X_test)
    _assert_saved(isolated_models_dir, "random_forest")


def test_train_xgboost_fits_and_saves(synthetic_split, isolated_models_dir):
    X_train, X_test, y_train, _ = synthetic_split

    model = train_xgboost(X_train, y_train)

    assert len(model.predict(X_test)) == len(X_test)
    _assert_saved(isolated_models_dir, "xgboost")


def test_train_neural_network_fits_and_saves(synthetic_split, isolated_models_dir):
    X_train, X_test, y_train, _ = synthetic_split

    model = train_neural_network(X_train.values, y_train.values)

    predictions = model.predict(X_test.values)
    assert len(predictions) == len(X_test)
    _assert_saved(isolated_models_dir, "neural_network")
