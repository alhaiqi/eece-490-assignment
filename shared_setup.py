"""
shared_setup.py  -  EECE 490/690 Ensemble Learning, Dataset B (Heart Disease)
Owner: Member 1. Teammates import this so everyone uses the SAME split, CV folds and pipelines.

Usage in Colab (put heart.csv and this file in the same folder):
    from shared_setup import *
    X_train, X_test, y_train, y_test = load_and_split("heart.csv")
    pre = build_preprocessor("improved")   # or "basic"
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer, KNNImputer

# ---------------------------------------------------------------- constants
SEED = 42
TARGET = "HeartDisease"          # positive class = 1 (heart disease)
TEST_SIZE = 0.20

NUM_COLS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak", "FastingBS"]
CAT_COLS = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
ZERO_AS_MISSING = ["RestingBP", "Cholesterol"]   # a value of 0 is physiologically impossible

# One CV object for the whole team -> identical folds for every model
CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


# ---------------------------------------------------------------- data
def load_and_split(path="heart.csv"):
    """Load the CSV and return the fixed stratified 80/20 split.
    The test set must not be touched until the final evaluation (Part 5)."""
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)


# ---------------------------------------------------------------- preprocessing
def zeros_to_nan(X):
    """Stateless: mark impossible zeros in RestingBP / Cholesterol as missing.
    Learns nothing from data, so it cannot leak information."""
    X = X.copy()
    cols = [c for c in ZERO_AS_MISSING if c in X.columns]
    X[cols] = X[cols].replace(0, np.nan)
    return X


def build_numeric_pipeline(kind="improved"):
    """Numeric part only. Member 2 reuses this before SMOTENC
    (SMOTENC needs imputed, scaled numerics and NOT one-hot categoricals)."""
    if kind == "basic":
        # Keep zeros as real values; median imputation guards against any NaN.
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
        ])
    if kind == "improved":
        # zeros -> NaN, scale (KNN is distance-based), KNN-impute, add missingness flags
        return Pipeline([
            ("zeros_to_nan", FunctionTransformer(zeros_to_nan, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),          # ignores NaN when fitting
            ("impute", KNNImputer(n_neighbors=5, add_indicator=True)),
        ])
    raise ValueError("kind must be 'basic' or 'improved'")


def build_preprocessor(kind="improved"):
    """Full ColumnTransformer. Put it INSIDE a Pipeline with the model so it is
    refit on each training fold during cross-validation (no leakage)."""
    return ColumnTransformer([
        ("num", build_numeric_pipeline(kind), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])


def summarize_cv(results, name):
    """Turn sklearn cross_validate output into one tidy row (mean +/- std)."""
    row = {"model": name}
    for m in SCORING:
        for split in ("train", "test"):
            key = f"{split}_{m}"
            if key in results:
                label = "val" if split == "test" else "train"
                row[f"{label}_{m}"] = f"{results[key].mean():.3f} +/- {results[key].std():.3f}"
    return row
