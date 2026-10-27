"""Shared feature schema and preprocessing for training and inference."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

TARGET = "Machine failure"
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
LEAKAGE_COLUMNS = FAILURE_MODES.copy()
NUMERIC_FEATURES = [
    "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
    "Torque [Nm]", "Tool wear [min]",
]
CATEGORICAL_FEATURES = ["Type"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
FEATURE_ORDER = FEATURES.copy()

def make_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Return a transformer with a stable explicit feature schema."""
    numeric = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES),
         ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)],
        remainder="drop",
        verbose_feature_names_out=False,
    )

def make_pipeline(classifier, scale_numeric: bool = True) -> Pipeline:
    return Pipeline([("preprocessor", make_preprocessor(scale_numeric)), ("classifier", classifier)])
