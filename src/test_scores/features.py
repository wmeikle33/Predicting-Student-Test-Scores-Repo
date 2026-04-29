from __future__ import annotations
from typing import Iterable, Tuple, List
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from pandas.api.types import is_numeric_dtype

def split_features_label(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.Series]:
    y = df[label]
    X = df.drop(columns=[label])
    return X, y

def auto_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    oe_categories = [['easy', 'moderate', 'hard'],
    ['low', 'medium', 'high'],
    ['no', 'yes'],
    ['poor', 'average', 'good']]
    num_cols = X.select_dtypes(include='number').columns
    ohe_cols = ['study_method']
    oe_cols = X.select_dtypes(include='object').columns.difference(ohe_cols) 
    oe = OrdinalEncoder(categories=oe_categories)
    ohe = OneHotEncoder(handle_unknown='ignore')
    ss = StandardScaler()
    preprocess = ColumnTransformer([('Scaling',ss,num_cols),
                               ('Ordinal',oe,oe_cols),
                               ('Onehot',ohe,ohe_cols)])
    return preprocess
