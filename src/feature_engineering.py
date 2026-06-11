import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline


def apply_smote(X_train, y_train, random_state: int = 42):
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE — fraud: {y_res.sum():,} | legitimate: {(y_res == 0).sum():,}")
    return X_res, y_res


def apply_undersample(X_train, y_train, sampling_strategy: float = 0.1, random_state: int = 42):
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    print(f"After undersampling — fraud: {y_res.sum():,} | legitimate: {(y_res == 0).sum():,}")
    return X_res, y_res


def apply_smote_undersampling(X_train, y_train, random_state: int = 42):
    """Combined SMOTE + undersampling pipeline for a balanced dataset."""
    over = SMOTE(sampling_strategy=0.1, random_state=random_state)
    under = RandomUnderSampler(sampling_strategy=0.5, random_state=random_state)
    pipeline = ImbPipeline([("over", over), ("under", under)])
    X_res, y_res = pipeline.fit_resample(X_train, y_train)
    print(f"After combined resampling — fraud: {y_res.sum():,} | legitimate: {(y_res == 0).sum():,}")
    return X_res, y_res


def compute_class_weight(y_train) -> dict:
    """Return class_weight dict suitable for sklearn estimators."""
    n_total = len(y_train)
    n_fraud = y_train.sum()
    n_legit = n_total - n_fraud
    weight_fraud = n_total / (2 * n_fraud)
    weight_legit = n_total / (2 * n_legit)
    return {0: weight_legit, 1: weight_fraud}
