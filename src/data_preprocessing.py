import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


def load_data(path: str = None) -> pd.DataFrame:
    if path is None:
        path = os.path.join(RAW_DIR, "creditcard.csv")
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows — {df['Class'].sum()} fraud, {(df['Class'] == 0).sum()} legitimate")
    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scale Time and Amount; V1-V28 are already PCA-transformed."""
    df = df.copy()
    scaler = StandardScaler()
    df[["Time", "Amount"]] = scaler.fit_transform(df[["Time", "Amount"]])
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    X = df.drop(columns=["Class"])
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"Train fraud rate: {y_train.mean():.4%} | Test fraud rate: {y_test.mean():.4%}")
    return X_train, X_test, y_train, y_test


def save_processed(X_train, X_test, y_train, y_test):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    X_train.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)
    print(f"Processed splits saved to {PROCESSED_DIR}")


def load_processed():
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze()
    return X_train, X_test, y_train, y_test
