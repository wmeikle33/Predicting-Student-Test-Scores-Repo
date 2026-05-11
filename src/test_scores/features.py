from __future__ import annotations
from typing import Iterable, Tuple, List
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from pandas.api.types import is_numeric_dtype

def split_features_label(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.Series]:
    y = df[label]
    X = df.drop(columns=[label])
    return X, y

def auto_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    ordinal_cols = ['exam_difficulty', 'facility_rating', 'internet_access', 'sleep_quality']

    oe_categories = [['easy', 'moderate', 'hard'],
     ['low', 'medium', 'high'],
     ['no', 'yes'],
     ['poor', 'average', 'good']]

    nominal_cols = [
        "study_method",
    ]

    numeric_cols = X.select_dtypes(include="number").columns.tolist()

    ordinal_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=oe_categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    onehot_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        ("ordinal", ordinal_pipe, ordinal_cols),
        ("onehot", onehot_pipe, nominal_cols),
        ("numeric", numeric_pipe, numeric_cols),
    ])

    return preprocessor
