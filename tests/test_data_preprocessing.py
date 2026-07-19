import os

import numpy as np
import pandas as pd

from src.data_preprocessing import (
    load_data,
    load_processed,
    save_processed,
    scale_features,
    split_data,
)


def test_load_data_reads_csv_and_reports_shape(tmp_path, synthetic_transactions_df, capsys):
    csv_path = tmp_path / "creditcard.csv"
    synthetic_transactions_df.to_csv(csv_path, index=False)

    df = load_data(str(csv_path))

    assert df.shape == synthetic_transactions_df.shape
    out = capsys.readouterr().out
    assert "fraud" in out


def test_scale_features_standardizes_time_and_amount(synthetic_transactions_df):
    scaled = scale_features(synthetic_transactions_df)

    assert np.isclose(scaled["Time"].mean(), 0, atol=1e-8)
    assert np.isclose(scaled["Time"].std(ddof=0), 1, atol=1e-8)
    assert np.isclose(scaled["Amount"].mean(), 0, atol=1e-8)
    assert np.isclose(scaled["Amount"].std(ddof=0), 1, atol=1e-8)

    # V1-V28 columns must be left untouched
    pd.testing.assert_series_equal(scaled["V1"], synthetic_transactions_df["V1"])


def test_scale_features_does_not_mutate_input(synthetic_transactions_df):
    original = synthetic_transactions_df.copy()
    scale_features(synthetic_transactions_df)

    pd.testing.assert_frame_equal(synthetic_transactions_df, original)


def test_split_data_is_stratified_with_expected_sizes(synthetic_transactions_df):
    X_train, X_test, y_train, y_test = split_data(
        synthetic_transactions_df, test_size=0.25, random_state=42
    )

    n = len(synthetic_transactions_df)
    assert len(X_train) + len(X_test) == n
    assert len(X_test) == round(n * 0.25)

    full_rate = synthetic_transactions_df["Class"].mean()
    assert abs(y_train.mean() - full_rate) < 0.05
    assert abs(y_test.mean() - full_rate) < 0.05
    assert "Class" not in X_train.columns


def test_save_and_load_processed_roundtrip(synthetic_split, isolated_processed_dir):
    X_train, X_test, y_train, y_test = synthetic_split

    save_processed(X_train, X_test, y_train, y_test)
    for name in ("X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"):
        assert os.path.exists(os.path.join(str(isolated_processed_dir), name))

    X_train_l, X_test_l, y_train_l, y_test_l = load_processed()

    pd.testing.assert_frame_equal(X_train_l, X_train.reset_index(drop=True))
    pd.testing.assert_frame_equal(X_test_l, X_test.reset_index(drop=True))
    np.testing.assert_array_equal(y_train_l.values, y_train.values)
    np.testing.assert_array_equal(y_test_l.values, y_test.values)
